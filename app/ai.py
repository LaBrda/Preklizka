"""AI funkce: odstranění pozadí, výběr postavy kliknutím, zvětšení obrázku, fotka -> kreslený obrázek.

Modely jsou ONNX (GitHub Release "modely"), stahují se při prvním použití.
Výpočet: ve Windows exe přes DirectML (jakákoli grafika: NVIDIA, AMD, Intel), jinak procesor.
Inference běží v samostatném procesu (viz server.py), tady jsou jen čisté funkce.
"""
from __future__ import annotations

import hashlib
import math
import io
import os
import sys
import threading
from pathlib import Path

import numpy as np

BASE_URL = "https://github.com/LaBrda/Preklizka/releases/download/modely/"
MODELS = {   # jméno -> přibližná velikost v MB
    "isnet-anime.onnx": 168, "isnet-general-use.onnx": 170, "birefnet.onnx": 973,
    "sam-encoder.onnx": 95, "sam-decoder.onnx": 9,
    "esrgan-anime.onnx": 17, "esrgan-fast.onnx": 3,
    "cartoon-portret.onnx": 9, "cartoon-krajina.onnx": 9,
}


def models_dir() -> Path:
    d = os.environ.get("PREKLIZKA_MODELS")
    if d:
        p = Path(d)
    elif getattr(sys, "frozen", False):
        p = Path(sys.executable).resolve().parent / "Preklizka_modely"     # přenosné: vedle exe
    else:
        p = Path.home() / ".cache" / "preklizka" / "modely"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------- stahování modelů (běží v hlavním procesu serveru) ----------------
_dl: dict[str, dict] = {}
_dl_lock = threading.Lock()


def _download(name: str) -> None:
    import httpx
    st = _dl[name]
    dst = models_dir() / name
    tmp = dst.with_suffix(".part")
    try:
        with httpx.stream("GET", BASE_URL + name, timeout=60, follow_redirects=True) as r:
            r.raise_for_status()
            st["total"] = int(r.headers.get("content-length") or MODELS[name] * 1_000_000)
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 18):
                    f.write(chunk)
                    st["done"] += len(chunk)
        os.replace(tmp, dst)
        st["ok"] = True
    except Exception as e:                       # noqa: BLE001
        st["error"] = str(e)
        try:
            tmp.unlink()
        except OSError:
            pass


def prepare(names: list[str]) -> dict:
    """Zajistí modely. Vrátí {ready, pct, mb, error}; chybějící začne stahovat na pozadí."""
    missing = [n for n in names if not (models_dir() / n).exists()]
    if not missing:
        return {"ready": True}
    with _dl_lock:
        for n in missing:
            st = _dl.get(n)
            if st is None or st.get("error"):
                _dl[n] = {"done": 0, "total": MODELS[n] * 1_000_000, "ok": False, "error": ""}
                threading.Thread(target=_download, args=(n,), daemon=True).start()
    err = next((_dl[n]["error"] for n in missing if _dl[n].get("error")), "")
    done = sum(_dl[n]["done"] for n in missing)
    total = sum(_dl[n]["total"] for n in missing)
    return {"ready": False, "pct": int(100 * done / max(1, total)), "mb": round(total / 1e6), "error": err}


def needed(op: str, p: dict, gpu: bool) -> list[str]:
    if op == "bg":
        if p.get("kind") == "white":
            return []
        if p.get("kind") in ("photo", "auto"):
            return ["birefnet.onnx"] if gpu else ["isnet-general-use.onnx"]   # na grafice nejlepší model BiRefNet
        return ["isnet-anime.onnx"]
    if op == "select":
        return ["sam-encoder.onnx", "sam-decoder.onnx"]
    if op == "upscale":
        return ["esrgan-anime.onnx" if (gpu and p.get("quality", "auto") != "fast") or p.get("quality") == "best" else "esrgan-fast.onnx"]
    if op == "cartoon":
        st = p.get("style", "portret")
        return [] if st in ("none", "poly") else ["cartoon-krajina.onnx" if st == "krajina" else "cartoon-portret.onnx"]
    raise ValueError("Neznámá AI funkce.")


def device_info() -> dict:
    try:
        import onnxruntime as ort
        prov = ort.get_available_providers()
    except Exception:                             # noqa: BLE001
        return {"available": False, "device": "nedostupné", "gpu": False}
    if "CUDAExecutionProvider" in prov:
        return {"available": True, "device": "grafická karta (CUDA)", "gpu": True}
    if "DmlExecutionProvider" in prov:
        return {"available": True, "device": "grafická karta (DirectML)", "gpu": True}
    return {"available": True, "device": "procesor", "gpu": False}


# ---------------- inference (běží v pracovním procesu) ----------------
_sess: dict[str, object] = {}


def _session(name: str, cpu: bool = False):
    import onnxruntime as ort
    key = name + ("#cpu" if cpu else "")
    if key in _sess:
        return _sess[key]
    prov = ort.get_available_providers()
    cpu = cpu or os.environ.get("PREKLIZKA_AI_CPU") == "1"
    use = [] if cpu else [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in prov]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if "DmlExecutionProvider" in use:
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if not use:
        so.enable_cpu_mem_arena = False          # menší špička paměti na serveru
        if name == "birefnet.onnx":
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    path = str(models_dir() / name)
    try:
        s = ort.InferenceSession(path, so, providers=use + ["CPUExecutionProvider"])
    except Exception:                             # noqa: BLE001
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    _sess[key] = s
    return s


def _run(name: str, feeds: dict):
    s = _session(name)
    try:
        return s.run(None, feeds)
    except Exception:                             # noqa: BLE001  grafika to nezvládla -> procesor
        if s.get_providers()[0] == "CPUExecutionProvider":
            raise
        return _session(name, cpu=True).run(None, feeds)


def _gpu() -> bool:
    return device_info()["gpu"]


def decode(png: bytes) -> np.ndarray:
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    im = im.convert("RGBA")
    return np.asarray(im).copy()


def encode(rgba: np.ndarray) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(rgba).save(b, "PNG", compress_level=3)
    return b.getvalue()


def _flatten(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)


# ---- pozadí (IS-Net) ----
def _guided(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He et al.): zarovná hranu masky s hranami v obrázku."""
    import cv2
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(guide, -1, k)
    mp = cv2.boxFilter(src, -1, k)
    cov = cv2.boxFilter(guide * src, -1, k) - mI * mp
    var = cv2.boxFilter(guide * guide, -1, k) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def bg_mask(rgb: np.ndarray, kind: str) -> np.ndarray:
    """Měkká maska popředí 0..1 v plném rozlišení."""
    import cv2
    h, w = rgb.shape[:2]
    o = None
    if kind == "photo" and _gpu() and (models_dir() / "birefnet.onnx").exists():
        try:
            o = _birefnet(rgb)
        except Exception:                                # noqa: BLE001  grafika to nezvládla -> IS-Net
            o = None
    if o is not None:
        return _refine(rgb, cv2.resize(o, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32), kind)
    if kind == "photo" and not (models_dir() / "isnet-general-use.onnx").exists():
        raise RuntimeError("Model pro fotky chybí, zkus to znovu (stáhne se).")
    name = "isnet-general-use.onnx" if kind == "photo" else "isnet-anime.onnx"
    inp = _session(name).get_inputs()[0].name
    x = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA if max(h, w) > 1024 else cv2.INTER_CUBIC).astype(np.float32)
    x = x / max(float(x.max()), 1e-6) - np.array([0.485, 0.456, 0.406], np.float32)
    o = _run(name, {inp: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    o = (o - o.min()) / max(float(o.max() - o.min()), 1e-6)
    m = cv2.resize(o, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return _refine(rgb, m, kind)


def _birefnet(rgb: np.ndarray) -> np.ndarray:
    import cv2
    x = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    s = _session("birefnet.onnx")
    o = _run("birefnet.onnx", {s.get_inputs()[0].name: np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)})[0][0, 0]
    return (1.0 / (1.0 + np.exp(-o))).astype(np.float32)


def _refine(rgb: np.ndarray, m: np.ndarray, kind: str) -> np.ndarray:
    import cv2
    h, w = rgb.shape[:2]
    # hrany dopočítat podle skutečného obrázku v plném rozlišení
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    r = max(2, round(max(h, w) / 350))
    m = np.clip(_guided(g, m, r, 1e-3), 0, 1)
    # objekt: zalepit uzavřené díry (okna, mezery) a zahodit odtržené drobky
    hard = (m > 0.5).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if n > 2:
        big = st[1:, cv2.CC_STAT_AREA].max()
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= big * 0.01
        drop = ~keep[cc] & (hard > 0)
        m[drop] = 0
        hard[drop] = 0
    inv = (1 - hard).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]])).tolist())
    holes = np.zeros(n, bool)
    for i in range(1, n):
        if i not in border and st[i, cv2.CC_STAT_AREA] < h * w * 0.02:
            holes[i] = True
    if kind == "photo":                                 # u kreslených jsou uzavřené díry skutečné pozadí (mezi rukou a tělem)
        m[holes[cc]] = 1.0
        m = _grabcut_refine(rgb, m)
    return m


def _grabcut_refine(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Nejisté pásmo kolem hrany rozhodnout podle barev (GrabCut). Pomůže u kol, nárazníků, zrcátek."""
    import cv2
    h, w = rgb.shape[:2]
    s = min(1.0, 900 / max(h, w))
    sw, sh = max(8, round(w * s)), max(8, round(h * s))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
    hard = (ms > 0.5).astype(np.uint8)
    if hard.sum() < 50 or hard.all():
        return m
    k = max(3, round(max(sw, sh) / 60))
    core = cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    far = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * k + 1, 4 * k + 1))) == 0
    gc = np.where(ms > 0.5, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[core > 0] = cv2.GC_FGD
    gc[(ms > 0.85) & (core == 0)] = cv2.GC_PR_FGD
    gc[far] = cv2.GC_BGD
    gc[ms < 0.03] = cv2.GC_BGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
    # GrabCut smí jen doplnit nejisté kousky (kola u země, nárazník), nikdy neubírá to, co síť našla
    add = (fg > 0.5) & (m > 0.12) & (m <= 0.5)
    out = np.where(add, np.maximum(m, fg), m)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.clip(_guided(g, out.astype(np.float32), max(2, round(max(h, w) / 400)), 1e-3), 0, 1)


def _flood_mask(rgb: np.ndarray, tol: int = 40) -> np.ndarray:
    """Záchrana pro kresby: pozadí = barva okraje zaplavená od krajů. Mezery v obrysu se dočasně uzavřou."""
    import cv2
    h, w = rgb.shape[:2]
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    ref = np.median(edge, 0)
    near = (np.abs(rgb.astype(np.int16) - ref).max(-1) < tol).astype(np.uint8)
    r = max(2, round(max(h, w) / 150))                 # mezery v obrysu do ~1,3 % velikosti
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    walls = cv2.dilate(1 - near, ker)                   # zesílené čáry jen pro hledání pozadí
    open_ = (near > 0) & (walls == 0)
    n, cc = cv2.connectedComponents(open_.astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
    bg = np.isin(cc, border[border > 0]) & open_
    bg = (cv2.dilate(bg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 3, 2 * r + 3))) > 0) & (near > 0)
    return (~bg).astype(np.float32)


def remove_white(rgba: np.ndarray) -> np.ndarray:
    """Odstranit všechnu bílou (i uvnitř kresby). Hrany čar zůstanou hladké, bílý lem se odečte."""
    rgb = rgba[..., :3].astype(np.float32)
    d = (255.0 - rgb).max(-1)                           # jak daleko je pixel od bílé
    a = np.clip((d - 12.0) / 70.0, 0, 1)
    safe = np.maximum(a, 1e-3)[..., None]
    col = np.clip((rgb - 255.0 * (1 - a[..., None])) / safe, 0, 255)   # odečíst příměs bílé na hranách
    a = a * (rgba[..., 3].astype(np.float32) / 255.0)
    return np.dstack([col.astype(np.uint8), (a * 255 + 0.5).astype(np.uint8)])


def remove_bg(rgba: np.ndarray, kind: str, soft: bool = False) -> np.ndarray:
    import cv2
    if kind == "white":
        return remove_white(rgba)
    if kind == "auto":                                  # 3D režim: kresba nebo fotka podle barevnosti
        f = _flatten(rgba)
        kind = "anime" if ((f.max(-1).astype(np.int16) - f.min(-1)) < 30).mean() > 0.97 else "photo"
    rgb = _flatten(rgba)
    sat = rgb.max(-1).astype(np.int16) - rgb.min(-1)
    lineart = kind != "photo" and (sat < 30).mean() > 0.97
    if lineart:                                         # černobílá kresba: síť ji nepochopí, stačí zaplavit pozadí
        m = _flood_mask(rgb)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])   # průhledný vnitřek kresby -> bílá plocha
    else:
        m = bg_mask(rgb, kind)
    if (m > 0.5).mean() < 0.02 and kind != "photo":     # kresba bez ploch: model nic nenašel
        m = bg_mask(rgb, "photo")
    if (m > 0.5).mean() < 0.02:
        m = _flood_mask(rgb)
    a = np.clip((m - 0.2) / 0.6, 0, 1) if soft else np.clip((m - 0.4) / 0.2, 0, 1)   # soft = jemné vlasy pro PNG
    # uvnitř objektu plná maska, měkce jen v úzkém pásu u okraje (žádné průsvitné plochy)
    hard = (m > 0.5).astype(np.uint8)
    r = max(2, round(max(m.shape) / 250))
    inner = cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    outer = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) == 0
    a[inner] = 1.0
    a[outer] = 0.0
    res = rgba.copy()
    res[..., 3] = (a * rgba[..., 3].astype(np.float32)).astype(np.uint8)
    return res


# ---- výběr kliknutím (Segment Anything, vit_b) ----
_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
_STD = np.array([58.395, 57.12, 57.375], np.float32)


def _sam_embedding(rgb: np.ndarray) -> np.ndarray:
    import cv2
    key = hashlib.sha1(rgb.tobytes() + b"v2").hexdigest()[:20]
    cache = models_dir() / "cache"
    cache.mkdir(exist_ok=True)
    f = cache / f"sam_{key}.npy"
    if f.exists():
        return np.load(f)
    h, w = rgb.shape[:2]
    s = 1024 / max(h, w)
    nh, nw = int(h * s + 0.5), int(w * s + 0.5)
    x = (cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR).astype(np.float32) - _MEAN) / _STD
    pad = np.zeros((1024, 1024, 3), np.float32)
    pad[:nh, :nw] = x
    emb = _run("sam-encoder.onnx", {"x": pad.transpose(2, 0, 1)[None]})[0]
    old = sorted(cache.glob("sam_*.npy"), key=lambda p: p.stat().st_mtime)
    for p in old[:-6]:                                   # držet jen pár posledních obrázků
        try:
            p.unlink()
        except OSError:
            pass
    np.save(f, emb)
    return emb


def select(rgba: np.ndarray, points: list, box: list | None = None) -> np.ndarray:
    """S obdélníkem: obdélník + kliky dohromady (klik přidá, Shift+klik ubere).
    Bez obdélníku: každý klik = jeden objekt, kliky se sčítají, Shift+klik objekt odebere."""
    rgb = _flatten(rgba)
    h, w = rgb.shape[:2]
    emb = _sam_embedding(rgb).astype(np.float32)
    s = 1024 / max(h, w)
    pos = [p for p in points if p[2]]
    neg = [p for p in points if not p[2]]
    if box:
        x0, y0, x1, y1 = min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])
        pts = np.array([[x0 * s, y0 * s], [x1 * s, y1 * s]] + [[p[0] * s, p[1] * s] for p in points], np.float32)[None]
        lab = np.array([2, 3] + [1 if p[2] else 0 for p in points], np.float32)[None]
        m = _run("sam-decoder.onnx", {
            "image_embeddings": emb, "point_coords": pts, "point_labels": lab,
            "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([h, w], np.float32)})[0][0, 0] > 0.0
        m = _clean_mask(m) if m.any() else m
        res = rgba.copy()
        res[..., 3] = np.where(m, rgba[..., 3], 0).astype(np.uint8)
        return res

    def one(p, others):
        pts = np.array([[p[0] * s, p[1] * s]] + [[q[0] * s, q[1] * s] for q in others] + [[0, 0]], np.float32)[None]
        lab = np.array([1] + [0] * len(others) + [-1], np.float32)[None]
        masks, iou, _ = _run("sam-decoder.onnx", {
            "image_embeddings": emb, "point_coords": pts, "point_labels": lab,
            "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([h, w], np.float32)})
        k = int(np.argmax(iou[0])) if masks.shape[1] > 1 else 0
        return masks[0, k] > 0.0

    m = np.zeros((h, w), bool)
    for p in pos:
        m |= one(p, neg)
    for q in neg:
        rm = one(q, pos)
        for p in pos:                                    # objekt, na který se klikalo, nikdy neodebrat
            if rm[min(h - 1, int(p[1])), min(w - 1, int(p[0]))]:
                rm = np.zeros_like(rm)
                break
        m &= ~rm
    m = _clean_mask(m) if m.any() else m
    res = rgba.copy()
    res[..., 3] = np.where(m, rgba[..., 3], 0).astype(np.uint8)
    return res


def _clean_mask(m: np.ndarray) -> np.ndarray:
    import cv2
    m8 = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n > 2:                                            # malé odtržené kousky pryč
        big = st[1:, cv2.CC_STAT_AREA].max()
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(30, big * 0.02)
        m8 = keep[lab].astype(np.uint8)
    inv = (1 - m8).astype(np.uint8)                      # malé díry zalepit
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    holes = np.zeros(n, bool)
    holes[1:] = st[1:, cv2.CC_STAT_AREA] < m8.size * 0.0005
    m8[holes[lab]] = 1
    return m8.astype(bool)


# ---- zvětšení (Real-ESRGAN anime) ----
def _tiled(name: str, x: np.ndarray, scale: int, tile: int, pad: int = 12) -> np.ndarray:
    _, _, h, w = x.shape
    out = np.zeros((1, 3, h * scale, w * scale), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ya, yb = max(0, y0 - pad), min(h, y0 + tile + pad)
            xa, xb = max(0, x0 - pad), min(w, x0 + tile + pad)
            o = _run(name, {"input": np.ascontiguousarray(x[:, :, ya:yb, xa:xb])})[0]
            ty, tx = (y0 - ya) * scale, (x0 - xa) * scale
            th, tw = (min(h, y0 + tile) - y0) * scale, (min(w, x0 + tile) - x0) * scale
            out[:, :, y0 * scale:y0 * scale + th, x0 * scale:x0 * scale + tw] = o[:, :, ty:ty + th, tx:tx + tw]
    return out


def upscale(rgba: np.ndarray, quality: str = "auto", max_in: int = 700) -> np.ndarray:
    import cv2
    h, w = rgba.shape[:2]
    if max(h, w) > max_in:                               # výsledek nad ~2800 px už appka stejně zmenší
        k = max_in / max(h, w)
        rgba = cv2.resize(rgba, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        h, w = rgba.shape[:2]
    gpu = _gpu()
    name = "esrgan-anime.onnx" if (gpu and quality != "fast") or quality == "best" else "esrgan-fast.onnx"
    x = _flatten(rgba).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    y = _tiled(name, x, 4, 192 if gpu else 128)
    rgb = (np.clip(y[0].transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)
    a = cv2.resize(rgba[..., 3], (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    return np.dstack([rgb, a])


# ---- fotka -> kreslený obrázek (AnimeGANv2 + zjednodušení barev + obrysy) ----
def _animegan(rgb: np.ndarray, style: str) -> np.ndarray:
    import cv2
    h, w = rgb.shape[:2]
    lim = 1024 if _gpu() else 768
    k = min(1.0, lim / max(h, w))
    nh, nw = max(8, int(h * k) // 8 * 8), max(8, int(w * k) // 8 * 8)
    x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1
    y = _run("cartoon-krajina.onnx" if style == "krajina" else "cartoon-portret.onnx", {"input": x.transpose(2, 0, 1)[None]})[0][0]
    y = ((np.clip(y.transpose(1, 2, 0), -1, 1) + 1) * 127.5).astype(np.uint8)
    return cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)


def _majority(idx: np.ndarray, k: int, r: int) -> np.ndarray:
    import cv2
    best = np.full(idx.shape, -1.0, np.float32)
    out = idx.copy()
    for c in range(k):
        v = cv2.blur((idx == c).astype(np.float32), (2 * r + 1, 2 * r + 1))
        upd = v > best
        best[upd] = v[upd]
        out[upd] = c
    return out


def _merge_shades(idx: np.ndarray, cent: np.ndarray, fg: np.ndarray):
    """Sloučit odstíny jedné barvy: osvětlená a stinná červená, šedé odlesky ve skle, skoro stejné barvy.
    Výsledná barva = barva největší části (nátěr má být čistý, ne kalný průměr)."""
    k = len(cent)
    cnt = np.bincount(idx[fg].ravel(), minlength=k).astype(np.float64) if fg.any() else np.bincount(idx.ravel(), minlength=k).astype(np.float64)
    L = cent[:, 0] * 100 / 255
    A = cent[:, 1] - 128
    B = cent[:, 2] - 128
    C = np.hypot(A, B)
    hue = np.degrees(np.arctan2(B, A))
    par = list(range(k))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            de = math.sqrt((L[i] - L[j]) ** 2 + (A[i] - A[j]) ** 2 + (B[i] - B[j]) ** 2)
            dh = abs((hue[i] - hue[j] + 180) % 360 - 180)
            same = de < 14
            same |= C[i] > 25 and C[j] > 25 and dh < 14 and abs(L[i] - L[j]) < 32       # světlá a stinná stejná barva
            same |= (C[i] < 12 and C[j] < 12 and abs(L[i] - L[j]) < 22 and min(L[i], L[j]) > 18   # šedé (sklo, kov), ne černá
                     and (max(L[i], L[j]) < 84 or min(L[i], L[j]) >= 84))                       # ani bílá (pruh, nárazník)
            if same:
                pairs.append((de, i, j))
    for _, i, j in sorted(pairs):
        a, b = find(i), find(j)
        if a != b:
            if cnt[a] < cnt[b]:
                a, b = b, a
            par[b] = a
            cnt[a] += cnt[b]
    root = np.array([find(i) for i in range(k)])
    return root[idx].astype(np.int32), cent


def cartoon(rgba: np.ndarray, style: str = "portret", colors: int = 8, lines: bool = True, detail: int = 5) -> np.ndarray:
    import cv2
    from scipy import ndimage as ndi
    H, W = rgba.shape[:2]
    d = int(np.clip(detail, 1, 10))
    fg_full = rgba[..., 3] > 127
    rgb = _flatten(rgba)
    if style != "none":
        rgb = _animegan(rgb, style)
    # plochy hledat v menším rozlišení, hranice pak hladce zvětšit
    lim = 500 + 50 * d
    s = min(1.0, lim / max(H, W))
    h, w = max(8, round(H * s)), max(8, round(W * s))
    small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    fg = cv2.resize(fg_full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    sm = cv2.pyrMeanShiftFiltering(small, 6 + (10 - d), 18 + 3 * (10 - d))
    lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    sel = fg.reshape(-1) if fg.any() else np.ones(h * w, bool)
    samp = lab[sel]
    if len(samp) > 40000:
        samp = samp[np.random.default_rng(1).choice(len(samp), 40000, replace=False)]
    k = int(np.clip(colors, 2, 24))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.3)
    _, _, cent = cv2.kmeans(samp, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    idx = ((lab[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1).astype(np.int32).reshape(h, w)
    idx, cent = _merge_shades(idx, cent, fg)
    idx = _majority(idx, k, 2)
    # dílky: malé plochy slučovat se sousedy, dokud jich nezbude cílový počet (posuvník Detail)
    target = 10 + 7 * d                                   # detail 1 -> ~17 ploch, 10 -> ~80
    min_area = max(12, int(h * w * 0.0004))
    for _ in range(14):
        comps = []
        bad = np.zeros((h, w), bool)
        total = 0
        for c in range(k):
            n, cc, st, _ = cv2.connectedComponentsWithStats(((idx == c) & fg).astype(np.uint8), connectivity=4)
            total += n - 1
            sm_ = np.zeros(n, bool)
            sm_[1:] = st[1:, cv2.CC_STAT_AREA] < min_area
            bad |= sm_[cc]
        if total <= target and not bad.any():
            break
        if bad.any() and not (bad | ~fg).all():
            _, (iy, ix) = ndi.distance_transform_edt(bad | ~fg, return_indices=True)
            nidx = idx[iy, ix]
            idx = np.where(bad, nidx, idx)
        if total <= target:
            break
        min_area = int(min_area * 1.6)
    # zpět na plné rozlišení s hladkými hranicemi (každá barva rozmazaná, vyhrává nejsilnější)
    sig = max(1.0, 1.2 / s)
    best = np.full((H, W), -1.0, np.float32)
    big = np.zeros((H, W), np.int32)
    for c in range(k):
        m = (idx == c).astype(np.float32)
        if not m.any():
            continue
        v = cv2.GaussianBlur(cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), sig)
        upd = v > best
        best[upd] = v[upd]
        big[upd] = c
    pal = cv2.cvtColor(cent.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    out = pal[big]
    fgF = fg_full if not fg_full.all() else np.ones((H, W), bool)
    if not fg_full.all():                                # hladký obrys postavy
        fgF = cv2.GaussianBlur(fg_full.astype(np.float32), (0, 0), max(1.0, 0.8 / s)) > 0.5
    if lines:
        # tmavé tahy z kresby (oči, obočí, ústa): tmavší než okolí, tenké, ne drobné tečky
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        loc = cv2.GaussianBlur(g, (0, 0), max(2.0, max(h, w) / 90))
        dark = ((loc - g) > (34 - 2 * d)) & fg
        dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, cc, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(14, h * w * (0.0005 - 0.00004 * d))
        dark = keep[cc].astype(np.float32)
        dark = cv2.GaussianBlur(cv2.resize(dark, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), sig) > 0.45
        t = max(2, round(max(H, W) / 420))
        edge = np.zeros((H, W), bool)
        edge[:, 1:] |= big[:, 1:] != big[:, :-1]
        edge[1:, :] |= big[1:, :] != big[:-1, :]
        fe = np.zeros((H, W), bool)
        fe[:, 1:] |= fgF[:, 1:] != fgF[:, :-1]
        fe[1:, :] |= fgF[1:, :] != fgF[:-1, :]
        edge = (edge & fgF) | fe
        edge = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))) > 0
        edge |= dark & fgF
        # drobné plošky uzavřené linkami zalít linkou (jinak z nich jsou mini dílky)
        n, cc, st, _ = cv2.connectedComponentsWithStats((fgF & ~edge).astype(np.uint8), connectivity=4)
        tiny = np.zeros(n, bool)
        tiny[1:] = st[1:, cv2.CC_STAT_AREA] < H * W * (0.0012 - 0.0001 * d)
        edge |= tiny[cc]
        out[edge] = (22, 20, 24)
        fgF = fgF | edge
    # odtržené drobky mimo postavu (zbytky pozadí) pryč
    n, cc, st, _ = cv2.connectedComponentsWithStats(fgF.astype(np.uint8), connectivity=8)
    if n > 2:
        big = st[1:, cv2.CC_STAT_AREA].max()
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= big * 0.01
        fgF = keep[cc]
    a = np.where(fgF, 255, 0).astype(np.uint8)
    return np.dstack([out, a])


def lowpoly(rgba: np.ndarray, colors: int = 8, detail: int = 5, lines: bool = True) -> np.ndarray:
    """Polygonový styl: trojúhelníky (Delaunay) s body hustěji u hran, barvy omezené na pár odstínů."""
    import cv2
    H, W = rgba.shape[:2]
    d = int(np.clip(detail, 1, 10))
    fg = rgba[..., 3] > 127
    if fg.all() or not fg.any():
        fg = np.ones((H, W), bool)
    rgb = _flatten(rgba)
    s = min(1.0, 900 / max(H, W))
    h, w = max(8, round(H * s)), max(8, round(W * s))
    small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    fgs = cv2.resize(fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    g = cv2.cvtColor(cv2.bilateralFilter(small, 7, 40, 7), cv2.COLOR_RGB2GRAY)
    n_pts = int(120 + 90 * d)                                  # detail 1 -> ~210 bodů, 10 -> ~1000
    rng = np.random.default_rng(3)
    pts = []
    # body hlavně tam, kde je v obrázku hrana (oči, ústa, vlasy), úměrně síle hrany
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.0) * fgs
    mag = mag ** 1.5
    pr = mag.reshape(-1) / max(float(mag.sum()), 1e-6)
    if pr.sum() > 0:
        k = rng.choice(h * w, int(n_pts * 0.6), replace=False, p=pr)
        pts += np.stack([k % w, k // w], 1).astype(float).tolist()
    ys, xs = np.nonzero(fgs)                                   # trochu náhodných bodů dovnitř
    if len(xs):
        k = rng.choice(len(xs), min(len(xs), int(n_pts * 0.15)), replace=False)
        pts += np.stack([xs[k], ys[k]], 1).astype(float).tolist()
    cnts, _ = cv2.findContours(fgs.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:                                             # body po obrysu postavy (silueta zůstane hladká)
        c = c.reshape(-1, 2)
        step = max(3, len(c) // max(8, int(n_pts * 0.3 * len(c) / max(1, sum(len(x) for x in cnts)))))
        pts += c[::step].astype(float).tolist()
    pts += [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]
    sub = cv2.Subdiv2D((0, 0, w, h))
    seen = set()
    for x, y in pts:
        x, y = float(min(w - 1, max(0, x))), float(min(h - 1, max(0, y)))
        key = (round(x), round(y))
        if key in seen:
            continue
        seen.add(key)
        sub.insert((x, y))
    tris = sub.getTriangleList().reshape(-1, 3, 2)
    tris = tris[(tris >= 0).all((1, 2)) & (tris[..., 0] < w).all(1) & (tris[..., 1] < h).all(1)]
    # průměrná barva každého trojúhelníku
    tid = np.full((h, w), -1, np.int32)
    for i, t in enumerate(tris):
        cv2.fillConvexPoly(tid, np.round(t).astype(np.int32), int(i))
    valid = tid >= 0
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    T = len(tris)
    cntv = np.bincount(tid[valid], minlength=T).astype(np.float64)
    mean = np.stack([np.bincount(tid[valid], weights=lab[valid.reshape(-1), c], minlength=T) for c in range(3)], 1) / np.maximum(cntv, 1)[:, None]
    fgt = np.bincount(tid[valid], weights=fgs[valid].astype(np.float64), minlength=T) / np.maximum(cntv, 1)
    inside = (fgt > 0.5) & (cntv > 0)
    k = int(np.clip(colors, 2, 24))
    samp = mean[inside].astype(np.float32)
    if len(samp) < k:
        k = max(1, len(samp))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.3)
    _, lbl, cent = cv2.kmeans(np.repeat(samp, np.maximum(1, (cntv[inside] / 50).astype(int)), axis=0) if len(samp) else samp, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    tcol = ((mean[:, None, :] - cent[None].astype(np.float64)) ** 2).sum(-1).argmin(1)
    pal = cv2.cvtColor(cent.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    cls = np.where(valid, tcol[np.maximum(tid, 0)], 0)
    ins = np.where(valid, inside[np.maximum(tid, 0)], False)
    # zpět na plné rozlišení: ostré hrany trojúhelníků (nejbližší soused)
    cls_f = cv2.resize(cls.astype(np.uint16), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    ins_f = cv2.resize(ins.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    out = pal[cls_f]
    if lines:
        t = max(2, round(max(H, W) / 420))
        edge = np.zeros((H, W), bool)
        edge[:, 1:] |= (cls_f[:, 1:] != cls_f[:, :-1]) | (ins_f[:, 1:] != ins_f[:, :-1])
        edge[1:, :] |= (cls_f[1:, :] != cls_f[:-1, :]) | (ins_f[1:, :] != ins_f[:-1, :])
        edge = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))) > 0
        n, cc, st, _ = cv2.connectedComponentsWithStats((ins_f & ~edge).astype(np.uint8), connectivity=4)
        tiny = np.zeros(n, bool)
        tiny[1:] = st[1:, cv2.CC_STAT_AREA] < H * W * 0.0008
        edge |= tiny[cc]
        out[edge] = (22, 20, 24)
        ins_f = ins_f | (edge & cv2.dilate(ins_f.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    a = np.where(ins_f, 255, 0).astype(np.uint8)
    return np.dstack([out, a])


def detect_wheels(rgba: np.ndarray) -> list:
    """Kola na fotce/kresbě z boku: tmavé kruhy dole na siluetě. Vrací [[cx, cy, r], ...] v pixelech, zleva doprava."""
    import cv2
    h, w = rgba.shape[:2]
    fg = rgba[..., 3] > 127
    if fg.all():                                        # bez průhlednosti: objekt = vše kromě barvy okraje
        fg = _flood_mask(_flatten(rgba)) > 0.5
    ys, xs = np.nonzero(fg)
    if len(xs) < 100:
        return []
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    H, W = y1 - y0 + 1, x1 - x0 + 1
    gray = cv2.cvtColor(_flatten(rgba), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(gray, (0, 0), max(1.0, H / 300))
    rmin, rmax = max(6, int(H * 0.10)), max(10, int(min(H * 0.42, W * 0.22)))
    cands = []
    for p2 in (0.85, 0.75):
        cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, 1.5, rmin * 1.6, param1=60, param2=p2, minRadius=rmin, maxRadius=rmax)
        if cs is not None:
            cands += cs[0].tolist()
        cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, rmin * 1.6, param1=100, param2=30, minRadius=rmin, maxRadius=rmax)
        if cs is not None:
            cands += cs[0][:20].tolist()
        if cands:
            break
    body = float(np.median(gray[fg]))
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for cx, cy, r in cands:
        if cy < y0 + H * 0.45 or cy + r < y1 - H * 0.2 or cy + r > y1 + r * 0.25:   # kolo je dole a dotýká se spodku
            continue
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < w and 0 <= iy < h) or not fg[iy, ix]:
            continue
        x_a, x_b, y_a, y_b = max(0, int(cx - r)), min(w, int(cx + r) + 1), max(0, int(cy - r)), min(h, int(cy + r) + 1)
        d = np.hypot(xx[y_a:y_b, x_a:x_b] - cx, yy[y_a:y_b, x_a:x_b] - cy)
        ring = (d > 0.72 * r) & (d < 0.97 * r)
        inside = fg[y_a:y_b, x_a:x_b][d < r].mean()
        tire = float(np.median(gray[y_a:y_b, x_a:x_b][ring])) if ring.any() else 255
        score = (body - tire) / 40.0 + inside
        if inside < 0.85 or tire > body + 10:
            continue
        out.append([float(cx), float(cy), float(r), score])
    # poloměr upřesnit: kolo sahá až na spodek siluety (bez tenkého stínu pod autem)
    kv = max(3, int(rmin * 0.35)) | 1
    solid = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8)) > 0
    for c in out:
        cx, cy, r = c[0], c[1], c[2]
        cols = [int(round(cx + dx * r)) for dx in (-0.25, 0, 0.25)]
        bots = []
        for x in cols:
            if 0 <= x < w:
                col = np.nonzero(solid[int(cy):, x])[0]
                if len(col):
                    # první mezera pod středem = konec kola
                    gaps = np.nonzero(np.diff(col) > 1)[0]
                    bots.append(int(cy) + (col[gaps[0]] if len(gaps) else col[-1]))
        if bots:
            rb = float(np.median(bots)) - cy
            if r * 0.95 <= rb <= r * 2.0:
                c[2] = rb
    # nepřekrývající se, nejlepší napřed
    out.sort(key=lambda c: -c[3])
    keep = []
    for c in out:
        if all(np.hypot(c[0] - k[0], c[1] - k[1]) > 0.9 * (c[2] + k[2]) for k in keep):
            keep.append(c)
    if keep:                                            # kola jednoho auta mají podobnou velikost
        rm = float(np.median([k[2] for k in keep]))
        keep = [k for k in keep if 0.7 * rm < k[2] < 1.4 * rm][:6]
    return [[round(c[0], 1), round(c[1], 1), round(c[2], 1)] for c in sorted(keep, key=lambda c: c[0])]


def run(op: str, png: bytes, p: dict) -> bytes:
    """Vstupní bod pro pracovní proces."""
    img = decode(png)
    if op == "bg":
        res = remove_bg(img, p.get("kind", "anime"), bool(p.get("soft", False)))
    elif op == "select":
        pts = p.get("points") or []
        if not pts and not p.get("box"):
            raise ValueError("Ohraničí postavu tažením myší.")
        res = select(img, pts, p.get("box"))
    elif op == "upscale":
        res = upscale(img, p.get("quality", "auto"))
    elif op == "cartoon" and p.get("style") == "poly":
        res = lowpoly(img, int(p.get("colors", 8)), int(p.get("detail", 5)), bool(p.get("lines", True)))
    elif op == "cartoon":
        res = cartoon(img, p.get("style", "portret"), int(p.get("colors", 8)), bool(p.get("lines", True)), int(p.get("detail", 5)))
    else:
        raise ValueError("Neznámá AI funkce.")
    return encode(res)
