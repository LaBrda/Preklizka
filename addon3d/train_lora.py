"""Naučit postavu: trénink LoRA pro DreamShaper 8 z několika obrázků (doplněk Preklizka.exe).

  python train_lora.py ukol.json   {"data": složka s obrázky, "out": soubor .safetensors, "trigger": slovo,
                                    "steps": kroky, "size": 512, "rank": 16}
Průběh píše na výstup řádky "KROK n/N", ať appka ukáže, kolik zbývá.
"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # vložený Python nepřidává složku skriptu
from repaint import model_dir, say  # noqa: E402


def train(job):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    gpu = torch.cuda.is_available()
    dev = "cuda" if gpu else "cpu"
    dt = torch.float16 if gpu else torch.float32
    say("STAV Trénuje " + ("grafická karta " + torch.cuda.get_device_name(0) if gpu else "procesor (bez NVIDIA karty, potrvá to velmi dlouho)"))
    sd = os.path.join(model_dir(), "sd")
    tok = CLIPTokenizer.from_pretrained(sd, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(sd, subfolder="text_encoder", torch_dtype=dt, variant="fp16").to(dev)
    vae = AutoencoderKL.from_pretrained(sd, subfolder="vae", torch_dtype=dt, variant="fp16").to(dev)
    unet = UNet2DConditionModel.from_pretrained(sd, subfolder="unet", torch_dtype=dt, variant="fp16").to(dev)
    noise_sched = DDPMScheduler.from_pretrained(sd, subfolder="scheduler")
    for m in (te, vae, unet):
        m.requires_grad_(False)
    rank = int(job.get("rank", 16))
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:                                      # váhy LoRA trénovat ve float32 (stabilní i s fp16 modelem)
        p.data = p.data.float()
    if gpu:
        unet.enable_gradient_checkpointing()

    size = int(job.get("size", 512))
    files = sorted(f for e in ("png", "jpg", "jpeg", "webp") for f in glob.glob(os.path.join(job["data"], "*." + e)))
    if not files:
        raise SystemExit("CHYBA Žádné obrázky.")
    say(f"STAV Připravuji {len(files)} obrázků…")
    lat, emb, close = [], [], []                          # close = čísla podkladů s tváří zblízka
    caps = {}

    def cond(text):                                       # popis -> vstup pro model (stejné popisy jen jednou)
        if text not in caps:
            ids = tok(text, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids.to(dev)
            caps[text] = te(ids)[0]
        return caps[text]

    def fit(im):                                          # zachovat poměr stran (čtvercový ořez by fotkám na výšku uřízl hlavu)
        w, h = im.size
        r = min(max(w / h, 0.5), 2.0)
        tw = max(256, int(round(size * r ** 0.5 / 64)) * 64)
        th = max(256, int(round(size / r ** 0.5 / 64)) * 64)
        return ImageOps.fit(im, (tw, th), Image.LANCZOS, centering=(0.5, 0.3))

    flips = (False, True) if len(files) < 15 else (False,)   # s dost obrázky nezrcadlit: tvář není souměrná
    with torch.no_grad():
        for f in files:
            im = Image.open(f)
            if im.mode in ("RGBA", "LA"):                  # průhledné pozadí = bílá
                bg = Image.new("RGB", im.size, "white")
                bg.paste(im, mask=im.convert("RGBA").getchannel("A"))
                im = bg
            im = fit(im.convert("RGB"))
            extra = ""
            if os.path.exists(os.path.splitext(f)[0] + ".txt"):    # popis výřezu (tvář zblízka, horní půlka těla, celá postava)
                with open(os.path.splitext(f)[0] + ".txt", encoding="utf-8") as t:
                    extra = t.read().strip()
            c = cond(job["trigger"] + (", " + extra if extra else ""))   # slovo postavy + jen druh záběru: model se učí postavu, ne styl
            for flip in flips:
                x = np.asarray(ImageOps.mirror(im) if flip else im).astype(np.float32) / 127.5 - 1
                x = torch.from_numpy(x).permute(2, 0, 1)[None].to(dev, dt)
                if "face" in extra:
                    close.append(len(lat))
                lat.append(vae.encode(x).latent_dist.sample() * vae.config.scaling_factor)
                emb.append(c)
    say(f"STAV Podkladů {len(lat)}, z toho tváří zblízka {len(close)}")
    del vae, te
    if gpu:
        torch.cuda.empty_cache()

    steps = int(job.get("steps", 1000))
    opt = torch.optim.AdamW(params, lr=float(job.get("lr", 1e-4)), weight_decay=1e-2)
    snr = noise_sched.alphas_cumprod / (1 - noise_sched.alphas_cumprod)
    w_snr = torch.clamp(snr, max=5.0) / snr               # min-SNR: skoro čisté kroky nepřehluší ty, kde vzniká podoba
    unet.train()
    for i in range(steps):
        k = random.choice(close) if close and random.random() < 0.4 else random.randrange(len(lat))   # tvář zblízka častěji
        x0 = lat[k]
        noise = torch.randn_like(x0)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (1,), device=dev).long()
        xt = noise_sched.add_noise(x0, noise, t)
        with torch.autocast(dev, dtype=dt, enabled=gpu):
            pred = unet(xt, t, emb[k]).sample
        loss = F.mse_loss(pred.float(), noise.float()) * float(w_snr[int(t)])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if i % 10 == 0 or i == steps - 1:
            say(f"KROK {i + 1}/{steps}")
    os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(os.path.dirname(job["out"]), weight_name=os.path.basename(job["out"]),
                                              unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)))
    say("STAV Hotovo")


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        train(json.load(f))
