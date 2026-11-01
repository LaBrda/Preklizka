"""AI 3D z jedné fotky – TripoSG (VAST-AI, licence MIT, kód ve složce sg). Nástupce TripoSR, výrazně přesnější tvar.

  python run_sg.py vstup.png vystup.glb [kroky]
vstup.png  objekt s průhledným pozadím (appka ho vyřízne předem).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                 # vložený Python nepřidává složku skriptu
sys.path.insert(0, os.path.join(HERE, "sg"))
if os.environ.get("PREKLIZKA_SG_LIBS"):                  # a ignoruje PYTHONPATH (soubor ._pth)
    sys.path.insert(0, os.environ["PREKLIZKA_SG_LIBS"])

BASE = "https://github.com/LaBrda/Preklizka/releases/download/modely/"
FILES = {
    "model_index.json": "sg-model_index.json",
    "scheduler/scheduler_config.json": "sg-scheduler__scheduler_config.json",
    "transformer/config.json": "sg-transformer__config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": "sg-transformer__diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": "sg-transformer__diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json": "sg-transformer__diffusion_pytorch_model.safetensors.index.json",
    "feature_extractor_dinov2/preprocessor_config.json": "sg-feature_extractor_dinov2__preprocessor_config.json",
    "image_encoder_dinov2/model.safetensors": "sg-image_encoder_dinov2__model.safetensors",
    "image_encoder_dinov2/config.json": "sg-image_encoder_dinov2__config.json",
    "vae/config.json": "sg-vae__config.json",
    "vae/diffusion_pytorch_model.safetensors": "sg-vae__diffusion_pytorch_model.safetensors",
}


def model_dir():
    return os.environ.get("PREKLIZKA_SG_MODEL") or os.path.join(HERE, "..", "sg_model")


def say(msg):
    print(msg, flush=True)


def prepare(path):
    """Jako TripoSG: objekt oříznout, doplnit na čtverec s 10 % okrajem, pozadí bílé."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    al = a[..., 3]
    ys, xs = np.nonzero(al > 0.5)
    if not len(xs):
        raise SystemExit("CHYBA Fotka nemá odstraněné pozadí.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb = a[y0:y1, x0:x1, :3] * al[y0:y1, x0:x1, None] + (1 - al[y0:y1, x0:x1, None])
    h, w = rgb.shape[:2]
    s = int(max(h, w) * 1.2)
    out = np.ones((s, s, 3), np.float32)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = rgb
    return Image.fromarray((out * 255).astype(np.uint8))


def make(inp, out, steps=50, seed=42):
    import numpy as np
    import torch
    import trimesh
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from run3d import largest_part

    gpu = torch.cuda.is_available()
    dev = "cuda" if gpu else "cpu"
    dt = torch.float16 if gpu else (torch.bfloat16 if os.environ.get("PREKLIZKA_BF16") else torch.float32)
    say("STAV Počítá " + ("grafická karta " + torch.cuda.get_device_name(0) if gpu else "procesor (bez NVIDIA karty, potrvá to dlouho)"))
    pipe = TripoSGPipeline.from_pretrained(model_dir(), torch_dtype=dt).to(dev, dt)
    img = prepare(inp)
    say("STAV AI skládá 3D tvar…")
    with torch.no_grad():
        res = pipe(image=img, generator=torch.Generator(device=dev).manual_seed(int(seed)), num_inference_steps=int(steps),
                   guidance_scale=7.0, use_flash_decoder=False,
                   dense_octree_depth=6 if os.environ.get("PREKLIZKA_SG_FAST") else 8,
                   hierarchical_octree_depth=7 if os.environ.get("PREKLIZKA_SG_FAST") else (9 if gpu else 8)).samples[0]
    mesh = trimesh.Trimesh(res[0].astype(np.float32), np.ascontiguousarray(res[1]))
    mesh = largest_part(mesh)
    mesh.export(out)
    say(f"STAV Hotovo: {len(mesh.faces)} trojúhelníků")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        make(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:4]))
    else:
        print(__doc__)
