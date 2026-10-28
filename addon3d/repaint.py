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
    "cn/config.json": "cn-lineart-config.json",
    "cn/diffusion_pytorch_model.fp16.safetensors": "cn-lineart.fp16.safetensors",
}
STYLE = ("flat vector illustration, 2d cartoon, children's animated series, thick clean black outlines, "
         "flat solid colors, no shading, simple shapes, minimal details, sticker style, plain white background")
NEGATIVE = ("shading, gradient, shadows, highlights, reflections, photo, photorealistic, realistic, 3d render, "
            "texture, noise, grain, detailed, complex, text, letters, watermark, logo, blurry, messy lines, sketch")


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
    from PIL import Image
    from diffusers import (ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline,
                           UniPCMultistepScheduler)

    gpu = torch.cuda.is_available()
    dt = torch.float16 if gpu else (torch.bfloat16 if os.environ.get("PREKLIZKA_BF16") else torch.float32)
    say("STAV Počítá " + ("grafická karta " + torch.cuda.get_device_name(0) if gpu else "procesor (bez NVIDIA karty, potrvá to dlouho)"))
    d = model_dir()
    cn = ControlNetModel.from_pretrained(os.path.join(d, "cn"), torch_dtype=dt, variant="fp16", use_safetensors=True)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        os.path.join(d, "sd"), controlnet=cn, torch_dtype=dt, variant="fp16", use_safetensors=True,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    text = (job.get("prompt") or "").strip()
    prompt = (text + ", " if text else "") + STYLE
    steps, cfg = int(job.get("steps", 25)), float(job.get("cfg", 8.0))
    g = torch.Generator(device="cpu").manual_seed(int(job.get("seed", 7)))
    w, h = int(job["w"]), int(job["h"])
    if job.get("mode") == "new":
        p2 = StableDiffusionPipeline(**{k: v for k, v in pipe.components.items() if k != "controlnet"})
        p2 = p2.to("cuda" if gpu else "cpu")
        p2.set_progress_bar_config(disable=True)
        say("STAV AI kreslí obrázek…")
        res = p2(prompt, negative_prompt=NEGATIVE, width=w, height=h, num_inference_steps=steps, guidance_scale=cfg, generator=g).images[0]
        res.save(job["out"])
        say("STAV Hotovo")
        return
    pipe = pipe.to("cuda" if gpu else "cpu")
    pipe.set_progress_bar_config(disable=True)
    img = Image.open(job["img"]).convert("RGB").resize((w, h), Image.LANCZOS)
    con = Image.open(job["ctrl"]).convert("RGB").resize((w, h), Image.LANCZOS)
    say("STAV AI překresluje obrázek…")
    res = pipe(prompt, negative_prompt=NEGATIVE, image=img, control_image=con, strength=float(job.get("strength", 0.9)),
               num_inference_steps=steps, guidance_scale=cfg, controlnet_conditioning_scale=float(job.get("cn", 1.0)),
               generator=g).images[0]
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
