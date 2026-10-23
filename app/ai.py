"""AI funkce: odstranění pozadí, výběr postavy kliknutím, zvětšení obrázku, fotka -> kreslený obrázek.

Modely jsou ONNX (GitHub Release "modely"), stahují se při prvním použití.
Výpočet: ve Windows exe přes DirectML (jakákoli grafika: NVIDIA, AMD, Intel), jinak procesor.
Inference běží v samostatném procesu (viz server.py), tady jsou jen čisté funkce.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
from pathlib import Path

import numpy as np

BASE_URL = "https://github.com/LaBrda/Preklizka/releases/download/modely/"
MODELS = {   # jméno -> přibližná velikost v MB
    "isnet-anime.onnx": 168, "isnet-general-use.onnx": 170,
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
        return ["isnet-general-use.onnx" if p.get("kind") == "photo" else "isnet-anime.onnx"]
    if op == "select":
        return ["sam-encoder.onnx", "sam-decoder.onnx"]
    if op == "upscale":
        return ["esrgan-anime.onnx" if (gpu and p.get("quality", "auto") != "fast") or p.get("quality") == "best" else "esrgan-fast.onnx"]
    if op == "cartoon":
        st = p.get("style", "portret")
        return [] if st == "none" else ["cartoon-krajina.onnx" if st == "krajina" else "cartoon-portret.onnx"]
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
    use = [] if cpu else [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in prov]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if "DmlExecutionProvider" in use:
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if not use:
        so.enable_cpu_mem_arena = False          # menší špička paměti na serveru
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
def remove_bg(rgba: np.ndarray, kind: str) -> np.ndarray:
    import cv2
    rgb = _flatten(rgba)
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA if max(h, w) > 1024 else cv2.INTER_CUBIC).astype(np.float32)
    x = x / max(float(x.max()), 1e-6) - np.array([0.485, 0.456, 0.406], np.float32)
    name = "isnet-general-use.onnx" if kind == "photo" else "isnet-anime.onnx"
    inp = _session(name).get_inputs()[0].name
    out = _run(name, {inp: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    m = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    a = np.clip((m - 0.35) / 0.3, 0, 1)                 # ostřejší hrana, žádné průsvitné závoje
    res = rgba.copy()
    res[..., 3] = (a * (rgba[..., 3].astype(np.float32))).astype(np.uint8)
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
    idx = _majority(idx, k, 2)
    # malé skvrny sloučit se sousedem
    min_area = max(12, int(h * w * (0.0002 + 0.0012 * (10 - d) / 9)))
    for _ in range(2):
        bad = np.zeros((h, w), bool)
        for c in range(k):
            n, cc, st, _ = cv2.connectedComponentsWithStats((idx == c).astype(np.uint8), connectivity=4)
            sm_ = np.zeros(n, bool)
            sm_[1:] = st[1:, cv2.CC_STAT_AREA] < min_area
            bad |= sm_[cc]
        if not bad.any() or bad.all():
            break
        _, (iy, ix) = ndi.distance_transform_edt(bad, return_indices=True)
        idx = idx[iy, ix]
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
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(10, h * w * 0.00008)
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
        out[edge] = (22, 20, 24)
        fgF = fgF | edge
    a = np.where(fgF, 255, 0).astype(np.uint8)
    return np.dstack([out, a])


def run(op: str, png: bytes, p: dict) -> bytes:
    """Vstupní bod pro pracovní proces."""
    img = decode(png)
    if op == "bg":
        res = remove_bg(img, p.get("kind", "anime"))
    elif op == "select":
        pts = p.get("points") or []
        if not pts and not p.get("box"):
            raise ValueError("Ohraničí postavu tažením myší.")
        res = select(img, pts, p.get("box"))
    elif op == "upscale":
        res = upscale(img, p.get("quality", "auto"))
    elif op == "cartoon":
        res = cartoon(img, p.get("style", "portret"), int(p.get("colors", 8)), bool(p.get("lines", True)), int(p.get("detail", 5)))
    else:
        raise ValueError("Neznámá AI funkce.")
    return encode(res)
