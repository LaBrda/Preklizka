"""AI 3D doplněk Překližkových vrstev: jedna fotka (s průhledným pozadím) -> 3D model (GLB).

Spouští ho Preklizka.exe vlastním Pythonem doplňku:
  python run3d.py --download              stáhne model TripoSR (při instalaci)
  python run3d.py vstup.png vystup.glb    vytvoří model
Model: TripoSR (Stability AI + Tripo AI, licence MIT), kód ve složce tsr.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # vložený Python nepřidává složku skriptu

BASE = "https://github.com/LaBrda/Preklizka/releases/download/modely/"
FILES = {"model.ckpt": "triposr-model.ckpt", "config.yaml": "triposr-config.yaml", "dino/config.json": "dino-vitb16-config.json"}


def model_dir():
    return os.path.join(os.environ.get("PREKLIZKA_3D_MODEL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))


def say(msg):
    print(msg, flush=True)


def download():
    import urllib.request
    d = model_dir()
    os.makedirs(os.path.join(d, "dino"), exist_ok=True)
    for local, remote in FILES.items():
        dst = os.path.join(d, local)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
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


def load_model(dev):
    import torch
    from omegaconf import OmegaConf
    from tsr.system import TSR
    d = model_dir()
    cfg = OmegaConf.load(os.path.join(d, "config.yaml"))
    OmegaConf.resolve(cfg)
    cfg.image_tokenizer.pretrained_model_name_or_path = os.path.join(d, "dino")   # bez internetu
    model = TSR(cfg)
    model.load_state_dict(torch.load(os.path.join(d, "model.ckpt"), map_location="cpu"))
    return model


def make(inp, out, resolution=256):
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from tsr.utils import resize_foreground

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    say("STAV Počítá " + ("grafická karta " + torch.cuda.get_device_name(0) if dev != "cpu" else "procesor (bez NVIDIA karty, potrvá to déle)"))
    model = load_model(dev)
    model.renderer.set_chunk_size(8192 if dev != "cpu" else 4096)
    model.to(dev)

    img = Image.open(inp).convert("RGBA")
    a = np.asarray(img)
    if a[..., 3].min() > 250:
        raise SystemExit("CHYBA Fotka nemá odstraněné pozadí.")
    img = resize_foreground(img, 0.85)                  # objekt na 85 % obrázku, jako v TripoSR
    a = np.asarray(img).astype(np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:4] + (1 - a[..., 3:4]) * 0.5   # šedé pozadí
    img = Image.fromarray((rgb * 255).astype(np.uint8))

    say("STAV AI skládá 3D tvar…")
    with torch.no_grad():
        codes = model([img], device=dev)
    say("STAV Vytahuji povrch…")
    mesh = model.extract_mesh(codes, False, resolution=resolution)[0]
    # TripoSR má osu Z nahoru; GLB má nahoru Y
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    # jen největší kus (drobné úlomky pryč)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: len(m.faces))
    mesh.export(out)
    say(f"STAV Hotovo: {len(mesh.faces)} trojúhelníků")


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    elif len(sys.argv) >= 3:
        make(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 256)
    else:
        print(__doc__)
