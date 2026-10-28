"""AI kreslíř: překreslení fotky do kreslené podoby (doplněk Preklizka.exe).

  python repaint.py --download     stáhne model (asi 2,9 GB)
  python repaint.py ukol.json      {"mode": "draw"|"new", "img": .., "ctrl": .., "out": .., "prompt": .., "strength": .., "w": .., "h": ..}

draw: fotka (průhledné = bílá) + obrysy (bílé čáry na černé) -> kreslený obrázek stejného tvaru.
new:  nový obrázek jen z textu.
Model: DreamShaper 8 + ControlNet lineart (Stable Diffusion 1.5, licence CreativeML OpenRAIL-M).
"""
import json
import os
import sys

# vložený Python ve Windows (soubor ._pth) ignoruje PYTHONPATH: knihovny kreslíře přidat ručně, před ostatní
if os.environ.get("PREKLIZKA_KRESBA_LIBS"):
    sys.path.insert(0, os.environ["PREKLIZKA_KRESBA_LIBS"])

BASE = "https://github.com/LaBrda/Preklizka/releases/download/modely/"
FILES = {
    "sd/model_index.json": "ds8-model_index.json",
    "sd/unet/config.json": "ds8-unet-config.json",
    "sd/unet/diffusion_pytorch_model.fp16.safetensors": "ds8-unet.fp16.safetensors",
    "sd/text_encoder/config.json": "ds8-te-config.json",
    "sd/text_encoder/model.fp16.safetensors": "ds8-te.fp16.safetensors",
    "sd/vae/config.json": "ds8-vae-config.json",
    "sd/vae/diffusion_pytorch_model.fp16.safetensors": "ds8-vae.fp16.safetensors",
    "sd/scheduler/scheduler_config.json": "ds8-scheduler-config.json",
    "sd/tokenizer/merges.txt": "ds8-tok-merges.txt",
    "sd/tokenizer/vocab.json": "ds8-tok-vocab.json",
    "sd/tokenizer/special_tokens_map.json": "ds8-tok-special.json",
    "sd/tokenizer/tokenizer_config.json": "ds8-tok-config.json",
    "sdi/unet/config.json": "ds8i-unet-config.json",
    "sdi/unet/diffusion_pytorch_model.fp16.safetensors": "ds8i-unet.fp16.safetensors",
    **{"sdxl/" + k: "sdxl-" + k.replace("/", "-") for k in (
        "model_index.json", "unet/config.json", "vae/config.json", "text_encoder/config.json", "text_encoder_2/config.json",
        "scheduler/scheduler_config.json", "tokenizer/merges.txt", "tokenizer/vocab.json", "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json", "tokenizer_2/merges.txt", "tokenizer_2/vocab.json", "tokenizer_2/special_tokens_map.json",
        "tokenizer_2/tokenizer_config.json")},
    **{"mt/" + k: "mt-csen-" + k for k in (
        "config.json", "generation_config.json", "pytorch_model.bin", "source.spm", "target.spm", "vocab.json", "tokenizer_config.json")},
    "cn/config.json": "cn-lineart-config.json",
    "cn/diffusion_pytorch_model.fp16.safetensors": "cn-lineart.fp16.safetensors",
}
STYLE = ("simple flat cartoon illustration for kids, thick uniform black outlines, few large flat color areas, "
         "no shading, minimal details, clean simple shapes, plain white background")
NEGATIVE = ("photo, photorealistic, realistic, 3d render, shading, gradient, shadows, highlights, reflections, "
            "text, letters, logo, watermark, hoses, cables, small details, texture, noise, grain, blurry, messy lines")


REAL = "photorealistic, highly detailed, sharp focus, natural light"
REAL_NEG = "cartoon, drawing, painting, illustration, lowres, blurry, deformed, text, watermark, logo"


def is_xl(path):
    """Vlastní model: SDXL (dva textové kodéry), jinak SD 1.5."""
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        return any(k.startswith("conditioner.embedders.1") for k in f.keys())


def model_dir():
    return os.environ.get("PREKLIZKA_KRESBA_MODEL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kresba_model")


def say(msg):
    print(msg, flush=True)


def download():
    import urllib.request
    d = model_dir()
    for local, remote in FILES.items():
        dst = os.path.join(d, local)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        say(f"STAV Stahuji {remote}…")
        last = [-1]

        def hook(n, bs, total):
            pct = min(100, n * bs * 100 // total) if total > 0 else 0
            if pct != last[0] and pct % 2 == 0:
                last[0] = pct
                say(f"{remote}: {pct} %")
        urllib.request.urlretrieve(BASE + remote, dst + ".part", hook)
        os.replace(dst + ".part", dst)
    say("STAV Model je stažený.")


def make(job):
    import numpy as np
    import torch
    from PIL import Image, ImageFilter
    from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetPipeline,
                           StableDiffusionInpaintPipeline, StableDiffusionPipeline, UNet2DConditionModel, UniPCMultistepScheduler)

    gpu = torch.cuda.is_available()
    dev = "cuda" if gpu else "cpu"
    dt = torch.float16 if gpu else (torch.bfloat16 if os.environ.get("PREKLIZKA_BF16") else torch.float32)
    say("STAV Počítá " + ("grafická karta " + torch.cuda.get_device_name(0) if gpu else "procesor (bez NVIDIA karty, potrvá to dlouho)"))
    d = model_dir()
    mode = job.get("mode", "draw")
    if job.get("model"):
        return make_custom(job, dev, dt)
    cn = None if mode in ("new", "inpaint") else ControlNetModel.from_pretrained(os.path.join(d, "cn"), torch_dtype=dt, variant="fp16", use_safetensors=True)
    base = StableDiffusionPipeline.from_pretrained(os.path.join(d, "sd"), torch_dtype=dt, variant="fp16", use_safetensors=True,
                                                   safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    parts = dict(base.components)
    parts["scheduler"] = UniPCMultistepScheduler.from_config(base.scheduler.config)
    if mode == "new":
        pipe = StableDiffusionPipeline(**parts)
    elif mode == "inpaint":
        parts["unet"] = UNet2DConditionModel.from_pretrained(os.path.join(d, "sdi", "unet"), torch_dtype=dt, variant="fp16", use_safetensors=True)
        pipe = StableDiffusionInpaintPipeline(**parts)
    elif mode == "sketch":
        pipe = StableDiffusionControlNetPipeline(controlnet=cn, **parts)
    else:
        pipe = StableDiffusionControlNetImg2ImgPipeline(controlnet=cn, **parts)
    text = (job.get("prompt") or "").strip()
    if job.get("lora"):                                   # naučená postava: LoRA + její slovo do zadání
        pipe.load_lora_weights(os.path.dirname(job["lora"]), weight_name=os.path.basename(job["lora"]))
        text = job.get("trigger", "") + (", " + text if text else "")
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    prompt = (text + ", " if text else "") + STYLE
    steps, cfg = int(job.get("steps", 25)), float(job.get("cfg", 8.0))
    g = torch.Generator(device="cpu").manual_seed(int(job.get("seed", 7)))
    w, h = int(job["w"]), int(job["h"])
    common = dict(negative_prompt=NEGATIVE, num_inference_steps=steps, guidance_scale=cfg, generator=g)
    say("STAV AI kreslí…")
    if mode == "new":
        res = pipe(prompt, width=w, height=h, **common).images[0]
    elif mode == "inpaint":                               # jen vybraná část; zbytek obrázku zůstane beze změny
        img = Image.open(job["img"]).convert("RGB")
        mask = Image.open(job["mask"]).convert("L")
        res = pipe(prompt, image=img.resize((w, h), Image.LANCZOS), mask_image=mask.resize((w, h), Image.LANCZOS),
                   width=w, height=h, strength=0.99, **common).images[0]
        res = res.resize(img.size, Image.LANCZOS)
        soft = mask.filter(ImageFilter.GaussianBlur(max(2, img.size[0] // 300)))
        res = Image.composite(res, img, soft)
    else:
        con = Image.open(job["ctrl"]).convert("RGB").resize((w, h), Image.LANCZOS)
        if mode == "sketch":                              # podle popisu: obrysy fotky jen volně vedou polohu
            res = pipe(prompt, image=con, width=w, height=h, controlnet_conditioning_scale=float(job.get("cn", 0.45)),
                       control_guidance_end=0.6, **common).images[0]
        else:
            img = Image.open(job["img"]).convert("RGB").resize((w, h), Image.LANCZOS)
            res = pipe(prompt, image=img, control_image=con, strength=float(job.get("strength", 0.9)),
                       controlnet_conditioning_scale=float(job.get("cn", 1.0)), **common).images[0]
    res.save(job["out"])
    say("STAV Hotovo")


def make_custom(job, dev, dt):
    """Vlastní model uživatele (.safetensors, SD 1.5 nebo SDXL, třeba realistický Juggernaut).
    Bez ControlNetu: nový obrázek z textu, překreslení obrázku (img2img) nebo vybrané části."""
    import torch
    from PIL import Image, ImageFilter
    from diffusers import (StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionPipeline,
                           StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLPipeline,
                           UniPCMultistepScheduler)
    path, mode = job["model"], job.get("mode", "draw")
    xl = is_xl(path)
    say("STAV Načítám vlastní model" + (" (SDXL)" if xl else "") + "…")
    cfg_dir = os.path.join(model_dir(), "sdxl" if xl else "sd")
    T2I, I2I, INP = ((StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline) if xl else
                     (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline))
    kw = dict(config=cfg_dir, torch_dtype=dt, local_files_only=True)
    if not xl:
        kw.update(safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    base = T2I.from_single_file(path, **kw)
    base.scheduler = UniPCMultistepScheduler.from_config(base.scheduler.config)
    pipe = base if mode == "new" else (INP if mode == "inpaint" else I2I)(**base.components)
    text = (job.get("prompt") or "").strip()
    real = job.get("style") == "real"
    if job.get("lora") and not xl:
        pipe.load_lora_weights(os.path.dirname(job["lora"]), weight_name=os.path.basename(job["lora"]))
        text = job.get("trigger", "") + (", " + text if text else "")
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    prompt = (text + ", " if text else "") + (REAL if real else STYLE)
    w, h = int(job["w"]), int(job["h"])
    if xl:                                                # SDXL chce ~1024 px
        k = 1024 / max(w, h)
        w, h = int(w * k) // 8 * 8, int(h * k) // 8 * 8
    g = torch.Generator(device="cpu").manual_seed(int(job.get("seed", 7)))
    common = dict(negative_prompt=REAL_NEG if real else NEGATIVE, num_inference_steps=int(job.get("steps", 25)),
                  guidance_scale=6.0 if xl else 7.0, generator=g)
    say("STAV AI kreslí…")
    if mode == "new":
        res = pipe(prompt, width=w, height=h, **common).images[0]
    elif mode == "inpaint":
        img = Image.open(job["img"]).convert("RGB")
        mask = Image.open(job["mask"]).convert("L")
        res = pipe(prompt, image=img.resize((w, h), Image.LANCZOS), mask_image=mask.resize((w, h), Image.LANCZOS),
                   width=w, height=h, strength=0.99, **common).images[0].resize(img.size, Image.LANCZOS)
        res = Image.composite(res, img, mask.filter(ImageFilter.GaussianBlur(max(2, img.size[0] // 300))))
    else:                                                 # překreslit / podle popisu: img2img z fotky
        img = Image.open(job["img"]).convert("RGB").resize((w, h), Image.LANCZOS)
        res = pipe(prompt, image=img, strength=float(job.get("strength", 0.9)) if mode == "draw" else 0.85, **common).images[0]
    res.save(job["out"])
    say("STAV Hotovo")


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    elif len(sys.argv) >= 2:
        with open(sys.argv[1], encoding="utf-8") as f:
            make(json.load(f))
    else:
        print(__doc__)
