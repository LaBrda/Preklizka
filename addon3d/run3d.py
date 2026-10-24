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


def largest_part(mesh):
    """Jen největší kus sítě (úlomky pryč); bez scipy/networkx, které doplněk nemá."""
    import numpy as np
    import trimesh
    f = mesh.faces
    lab = np.arange(len(mesh.vertices))
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    while True:                                          # šíření nejmenšího čísla po hranách + zkratky
        old = lab.copy()
        m = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
        np.minimum.at(lab, e[:, 0], m)
        np.minimum.at(lab, e[:, 1], m)
        lab = lab[lab]
        if np.array_equal(lab, old):
            break
    fl = lab[f[:, 0]]
    keep = fl == np.bincount(fl).argmax()
    out = trimesh.Trimesh(mesh.vertices, f[keep], process=False)
    out.remove_unreferenced_vertices()
    return out


def smooth(mesh, it=20):
    """Taubinovo vyhlazení (nezmenšuje model): pryč schody z mřížky, zůstanou kola a hrany kabiny."""
    import numpy as np
    v = mesh.vertices.copy()
    e = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=len(v)).astype(float)[:, None]
    deg[deg == 0] = 1
    for _ in range(it):
        for k in (0.5, -0.53):
            avg = np.zeros_like(v)
            np.add.at(avg, e[:, 0], v[e[:, 1]])
            v += k * (avg / deg - v)
    mesh.vertices = v
    return mesh


def _lower_tilt(a, b, n=200):
    """Sklon spodní opěrné hrany (kola na zemi) v průmětu a (vodorovně) × b (nahoru), pod těžištěm."""
    import numpy as np
    bins = np.clip(((a - a.min()) / (np.ptp(a) + 1e-9) * n).astype(int), 0, n - 1)
    low = np.full(n, np.inf)
    np.minimum.at(low, bins, b)
    xs = a.min() + (np.arange(n) + 0.5) / n * np.ptp(a)
    pts = [(x, y) for x, y in zip(xs, low) if np.isfinite(y)]
    hull = []
    for p in pts:                                        # spodní konvexní obal (monotone chain)
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0:
            hull.pop()
        hull.append(p)
    c = a.mean()
    for p, q in zip(hull, hull[1:]):
        if p[0] <= c <= q[0]:
            return np.arctan2(q[1] - p[1], q[0] - p[0])
    return 0.0


def align(mesh):
    """Délka do osy X (předek doleva, k fotoaparátu), šířka Z (viditelný bok +Z), nahoru Y; model postavit na kola."""
    import numpy as np
    import trimesh
    R = trimesh.transformations.rotation_matrix
    v = mesh.vertices
    xz = v[::max(1, len(v) // 20000), :][:, [0, 2]]
    t = np.radians(np.arange(0, 180, 0.5))
    c, s = np.cos(t), np.sin(t)
    pa, pb = xz[:, :1] * c - xz[:, 1:] * s, xz[:, :1] * s + xz[:, 1:] * c
    ang = t[np.argmin(np.ptp(pa, 0) * np.ptp(pb, 0))]   # nejmenší obdélník v půdorysu = osy auta
    M = R(-ang, [0, 1, 0])
    e = np.ptp(trimesh.transform_points(v, M), 0)
    if e[0] < e[2]:
        M = R(np.pi / 2, [0, 1, 0]) @ M
    if trimesh.transform_points([[1.9, 0, 0]], M)[0, 0] > 0:   # TripoSR fotí z +X: bližší konec = předek, ten doleva
        M = R(np.pi, [0, 1, 0]) @ M
    if trimesh.transform_points([[1.9, 0, 0]], M)[0, 2] < 0:   # viditelný bok (k fotoaparátu) vždy na +Z (appka ho zrcadlí na druhý bok)
        M = np.diag([1.0, 1.0, -1.0, 1.0]) @ M
    mesh.apply_transform(M)
    lim = np.radians(12)
    for ax, i in ((2, 0), (0, 2)):                       # sklon dopředu/dozadu, pak do strany
        a = _lower_tilt(mesh.vertices[:, i], mesh.vertices[:, 1])
        if abs(a) < lim:
            mesh.apply_transform(R(-a if i == 0 else a, [0, 0, 1] if i == 0 else [1, 0, 0]))
    return mesh


def make(inp, out, resolution=256, ratio=0.65):
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
    # objekt na 65 % obrázku (TripoSR má 85 %): u aut šikmo zepředu vyjde méně nafouklá hloubka a výška
    img = resize_foreground(img, ratio)
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
    mesh = align(smooth(largest_part(mesh)))
    mesh.export(out)
    say(f"STAV Hotovo: {len(mesh.faces)} trojúhelníků")


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    elif len(sys.argv) >= 3:
        make(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 256)
    else:
        print(__doc__)
