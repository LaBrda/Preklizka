"""Zrcadlo TripoSG (VAST-AI, MIT) do GitHub Release "modely" (běží v Actions, workflow triposg.yml).

Release bere soubory do 2 GB: velké váhy se převedou na float16 a rozdělí na kusy (formát sharded safetensors,
diffusers ho načte sám). Soubory se jmenují sg-<cesta s __ místo />; sg-manifest.json říká, kam patří.
"""
import json
import os

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

OUT = "sg_out"
LIMIT = 1_900_000_000
os.makedirs(OUT, exist_ok=True)
src = snapshot_download("VAST-AI/TripoSG", local_dir="sg_src")
manifest = {}


def put(rel, path):
    name = "sg-" + rel.replace("/", "__")
    os.replace(path, os.path.join(OUT, name))
    manifest[rel] = name


for root, _, files in os.walk(src):
    if ".cache" in root or "/.git" in root:
        continue
    for f in files:
        p = os.path.join(root, f)
        rel = os.path.relpath(p, src).replace("\\", "/")
        if rel.startswith(".") or rel.endswith((".md", ".png", ".jpg", ".gitattributes")):
            continue
        if os.path.getsize(p) <= LIMIT:
            put(rel, p)
            continue
        assert rel.endswith(".safetensors"), rel
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in load_file(p).items()}
        base = rel[: -len(".safetensors")]
        shards, cur, size = [], {}, 0
        for k in sorted(sd):
            n = sd[k].numel() * sd[k].element_size()
            if cur and size + n > LIMIT:
                shards.append(cur)
                cur, size = {}, 0
            cur[k] = sd[k]
            size += n
        shards.append(cur)
        wmap = {}
        for i, sh in enumerate(shards, 1):
            fn = f"{os.path.basename(base)}-{i:05d}-of-{len(shards):05d}.safetensors"
            tmp = os.path.join(OUT, "tmp.safetensors")
            save_file(sh, tmp, metadata={"format": "pt"})
            put(os.path.join(os.path.dirname(rel), fn).replace("\\", "/"), tmp)
            wmap.update({k: fn for k in sh})
        idx = os.path.join(OUT, "tmp.json")
        json.dump({"metadata": {"total_size": sum(v.numel() * v.element_size() for v in sd.values())}, "weight_map": wmap}, open(idx, "w"))
        put(rel + ".index.json", idx)
        os.remove(p)
json.dump(manifest, open(os.path.join(OUT, "sg-manifest.json"), "w"), indent=1)
print(json.dumps(manifest, indent=1))
