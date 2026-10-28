"""ComfyUI jako AI kreslíř: překreslení fotky do kreslené podoby, úprava podle textu, nový obrázek z textu.

ComfyUI běží u uživatele (výchozí http://127.0.0.1:8188) s jeho vlastními modely; appka mu pošle
graf (workflow) přes API, počká na výsledek a stáhne obrázek. Tvar drží ControlNet z obrysů appky.
"""
from __future__ import annotations

import io
import json
import time
import uuid

import httpx

DEFAULT_URL = "http://127.0.0.1:8188"
PROMPT = ("(flat vector illustration:1.3), (2d cartoon:1.2), children's animated series, (thick clean black outlines:1.3), "
          "(flat solid colors:1.4), (no shading:1.3), simple shapes, minimal details, sticker style, plain white background")
NEGATIVE = ("(shading:1.3), (gradient:1.4), shadows, highlights, reflections, photo, photorealistic, realistic, 3d render, "
            "texture, noise, grain, detailed, complex, text, letters, watermark, logo, blurry, messy lines, sketch")


def _url(u: str | None) -> str:
    u = (u or DEFAULT_URL).strip().rstrip("/")
    return u if u.startswith("http") else "http://" + u


def _options(c: httpx.Client, base: str, node: str, field: str) -> list:
    try:
        j = c.get(f"{base}/object_info/{node}", timeout=5).json()
        v = j[node]["input"]["required"][field]
        v = v[0] if isinstance(v[0], list) else v[1].get("options", []) if len(v) > 1 and isinstance(v[1], dict) else []
        return [x for x in v if isinstance(x, str)]
    except Exception:                                   # noqa: BLE001
        return []


def status(url: str | None = None) -> dict:
    base = _url(url)
    try:
        with httpx.Client() as c:
            c.get(base + "/system_stats", timeout=3).raise_for_status()
            return {"ok": True, "url": base,
                    "ckpts": _options(c, base, "CheckpointLoaderSimple", "ckpt_name"),
                    "unets": _options(c, base, "UNETLoader", "unet_name"),
                    "controlnets": _options(c, base, "ControlNetLoader", "control_net_name")}
    except Exception as e:                              # noqa: BLE001
        return {"ok": False, "url": base, "error": str(e)[:200]}


def is_xl(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("xl", "pony", "illustrious", "noob"))


def _upload(c: httpx.Client, base: str, png: bytes, name: str) -> str:
    r = c.post(base + "/upload/image", files={"image": (name, png, "image/png")}, data={"overwrite": "true"}, timeout=60)
    r.raise_for_status()
    j = r.json()
    return (j.get("subfolder") + "/" if j.get("subfolder") else "") + j["name"]


def graph(mode: str, ckpt: str, cn: str | None, img: str | None, ctrl: str | None, prompt: str, negative: str,
          w: int, h: int, strength: float, steps: int, cfg: float, seed: int, cn_strength: float = 1.0) -> dict:
    """Workflow pro API ComfyUI. ckpt "unet:<soubor>" = samostatné soubory unet/clip/vae (test), jinak checkpoint."""
    g: dict = {}
    if ckpt.startswith("unet:"):
        _, unet, clip, vae = ckpt.split(":")
        g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}
        g["1c"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "stable_diffusion"}}
        g["1v"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae}}
        model, clip_o, vae_o = ["1", 0], ["1c", 0], ["1v", 0]
    else:
        g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}
        model, clip_o, vae_o = ["1", 0], ["1", 1], ["1", 2]
    g["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_o}}
    g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip_o}}
    pos, neg = ["2", 0], ["3", 0]
    if mode == "new":
        g["4"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
        denoise = 1.0
    else:
        g["5"] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g["6"] = {"class_type": "ImageScale", "inputs": {"image": ["5", 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "disabled"}}
        g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": vae_o}}
        denoise = float(strength)
    if cn and ctrl:                                     # obrysy drží tvar a proporce
        g["7"] = {"class_type": "LoadImage", "inputs": {"image": ctrl}}
        g["8"] = {"class_type": "ImageScale", "inputs": {"image": ["7", 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "disabled"}}
        g["9"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cn}}
        g["10"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": pos, "negative": neg, "control_net": ["9", 0],
                                                                     "image": ["8", 0], "strength": cn_strength,
                                                                     "start_percent": 0.0, "end_percent": 1.0}}
        pos, neg = ["10", 0], ["10", 1]
    g["11"] = {"class_type": "KSampler", "inputs": {"model": model, "seed": seed, "steps": steps, "cfg": cfg,
                                                    "sampler_name": "dpmpp_2m", "scheduler": "karras",
                                                    "positive": pos, "negative": neg, "latent_image": ["4", 0], "denoise": denoise}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": vae_o}}
    g["13"] = {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "preklizka"}}
    return g


def run(url: str | None, mode: str, ckpt: str, cn: str | None, img_png: bytes | None, ctrl_png: bytes | None,
        prompt: str, negative: str, size: tuple, strength: float = 0.75, steps: int = 25, cfg: float = 7.0,
        seed: int | None = None, timeout: float = 900) -> bytes:
    base = _url(url)
    w, h = size
    with httpx.Client() as c:
        img = _upload(c, base, img_png, "preklizka_vstup.png") if img_png and mode != "new" else None
        ctrl = _upload(c, base, ctrl_png, "preklizka_obrysy.png") if ctrl_png and cn else None
        g = graph(mode, ckpt, cn, img, ctrl, prompt, negative, w, h, strength, steps, cfg,
                  int(seed if seed is not None else time.time()) % 2**31)
        cid = uuid.uuid4().hex
        r = c.post(base + "/prompt", json={"prompt": g, "client_id": cid}, timeout=30)
        if r.status_code != 200:
            try:
                j = r.json()
                det = j.get("error", {}).get("message", "") + " " + json.dumps(j.get("node_errors", {}), ensure_ascii=False)[:400]
            except Exception:                           # noqa: BLE001
                det = r.text[:400]
            raise RuntimeError("ComfyUI odmítl úlohu: " + det)
        pid = r.json()["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(1.0)
            hist = c.get(f"{base}/history/{pid}", timeout=10).json()
            if pid not in hist:
                continue
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                msgs = [m[1].get("exception_message", "") for m in st.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError("ComfyUI chyba: " + " ".join(msgs)[:400])
            for out in hist[pid].get("outputs", {}).values():
                for im in out.get("images", []):
                    v = c.get(base + "/view", params={"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                                      "type": im.get("type", "output")}, timeout=60)
                    v.raise_for_status()
                    return v.content
        raise RuntimeError("ComfyUI nestihl obrázek vytvořit.")


def size_for(W: int, H: int, xl: bool, side: int | None = None) -> tuple:
    """Rozměr pro model: SD 1.5 ~768 px na delší straně, SDXL ~1024; násobky 8."""
    side = side or (1024 if xl else 768)
    k = side / max(W, H)
    return max(256, int(W * k) // 8 * 8), max(256, int(H * k) // 8 * 8)


def to_png(arr) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()
