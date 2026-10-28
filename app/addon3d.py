"""Instalace a spouštění AI 3D doplňku (TripoSR) – jen Preklizka.exe ve Windows.

Doplněk je samostatný Python s PyTorchem ve složce Preklizka_3D vedle exe (nic se neinstaluje do systému).
S NVIDIA kartou PyTorch s CUDA (~2,5 GB), jinak menší verze pro procesor.
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

PY_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
TORCH = ["torch==2.3.1"]
LIBS = ["numpy<2", "omegaconf==2.3.0", "einops==0.7.0", "transformers==4.35.0", "huggingface-hub<0.18",
        "trimesh==4.0.5", "PyMCubes==0.1.4", "Pillow==10.4.0"]
# AI kreslíř: vlastní složka knihoven (3D potřebuje staré transformers, kreslíř nové); Python a PyTorch sdílí
LIBS_K = ["diffusers==0.30.3", "transformers==4.44.2", "huggingface-hub==0.24.6", "safetensors==0.4.5",
          "tokenizers==0.19.1", "numpy<2", "Pillow==10.4.0"]
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

_st = {"state": "none", "step": "", "log": "", "gpu": None}
_stk = {"state": "none", "step": "", "log": ""}
_lock = threading.Lock()


def base_dir() -> Path:
    d = os.environ.get("PREKLIZKA_3D")
    if d:
        return Path(d)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "Preklizka_3D"
    return Path.home() / ".cache" / "preklizka" / "Preklizka_3D"


def src_dir() -> Path:
    b = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return b / "addon3d"


def python() -> Path:
    return base_dir() / "python" / ("python.exe" if os.name == "nt" else "bin/python3")


def has_nvidia() -> bool:
    if shutil.which("nvidia-smi") or Path(r"C:\Windows\System32\nvidia-smi.exe").exists():
        try:
            r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15, creationflags=NO_WINDOW)
            return r.returncode == 0 and "GPU" in r.stdout
        except Exception:                                   # noqa: BLE001
            return False
    return False


def installed() -> bool:
    return (base_dir() / "hotovo.txt").exists() and python().exists()


def status() -> dict:
    s = dict(_st)
    s["available"] = os.name == "nt" or bool(os.environ.get("PREKLIZKA_3D_TEST"))
    if installed() and s["state"] != "installing":
        s["state"] = "ready"
    if s["gpu"] is None:
        s["gpu"] = has_nvidia() if s["available"] else False
        _st["gpu"] = s["gpu"]
    s["size"] = "zhruba 4,5 GB" if s["gpu"] else "zhruba 2,5 GB"
    return s


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("_PYI", "_MEIPASS", "PYTHON"))}
    env["HF_HOME"] = str(base_dir() / "hf")
    env["PREKLIZKA_3D_MODEL"] = str(base_dir() / "model")
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


_cur = _st                                              # do kterého stavu se píše průběh (3D / kreslíř)


def _run(args: list, step: str, env: dict | None = None) -> None:
    _cur["step"] = step
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env or _env(), cwd=str(base_dir()), creationflags=NO_WINDOW, bufsize=1)
    last = []
    for line in iter(p.stdout.readline, ""):
        line = line.strip().replace("\r", "")
        if line:
            last = (last + [line])[-20:]
            _cur["log"] = line[-200:]
    if p.wait() != 0:
        raise RuntimeError(step + " selhalo: " + " | ".join(last[-4:]))


def _download(url: str, dst: Path, step: str) -> None:
    import httpx
    _cur["step"] = step
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(dst, "wb") as f:
            for ch in r.iter_bytes(1 << 18):
                f.write(ch)
                done += len(ch)
                if total:
                    _cur["log"] = f"{done * 100 // total} %"


def _runtime(n: int) -> None:
    """Společný Python + pip + PyTorch (s CUDA, když je NVIDIA karta)."""
    b = base_dir()
    b.mkdir(parents=True, exist_ok=True)
    gpu = has_nvidia()
    _st["gpu"] = gpu
    if os.name == "nt":
        if not python().exists():
            z = b / "python.zip"
            _download(PY_URL, z, f"1/{n} Stahuji Python")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(b / "python")
            z.unlink()
            for pth in (b / "python").glob("python*._pth"):      # povolit site-packages (pip)
                pth.write_text(pth.read_text().replace("#import site", "import site"))
        if not (b / "python" / "Scripts" / "pip.exe").exists():
            gp = b / "get-pip.py"
            _download(PIP_URL, gp, f"2/{n} Instaluji pip")
            _run([str(python()), str(gp), "--no-warn-script-location"], f"2/{n} Instaluji pip")
            gp.unlink(missing_ok=True)
    else:                                                         # test v CI na Linuxu: virtuální prostředí
        if not python().exists():
            _run([sys.executable, "-m", "venv", str(b / "python")], f"1/{n} Python")
    if not (b / "torch.txt").exists():
        idx = "https://download.pytorch.org/whl/" + ("cu121" if gpu else "cpu")
        _run(_pip() + TORCH + ["--index-url", idx], f"3/{n} Stahuji PyTorch" + (" pro grafickou kartu (asi 2,5 GB)" if gpu else " (asi 200 MB)"))
        (b / "torch.txt").write_text("ok")


def _models(script: str, dst: Path, step: str) -> None:
    """Modely stáhne appka (httpx s vlastními certifikáty); urllib ve vloženém Pythonu ve Windows certifikátům GitHubu nevěří."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_m", str(src_dir() / script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for local, remote in mod.FILES.items():
        f = dst / local
        if f.exists() and f.stat().st_size > 0:
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        _download(mod.BASE + remote, f.with_name(f.name + ".part"), f"{step}: {remote}")
        os.replace(f.with_name(f.name + ".part"), f)


def _pip() -> list:
    return [str(python()), "-m", "pip", "install", "--no-warn-script-location", "--disable-pip-version-check"]


def _copy_app() -> Path:
    app = base_dir() / "app"
    if app.exists():
        shutil.rmtree(app)
    shutil.copytree(src_dir(), app)
    return app


def _install() -> None:
    global _cur
    _cur = _st
    try:
        (base_dir() / "hotovo.txt").unlink(missing_ok=True)
        _runtime(5)
        _run(_pip() + LIBS, "4/5 Instaluji knihovny")
        _copy_app()
        _models("run3d.py", base_dir() / "model", "5/5 Stahuji model TripoSR (asi 1,7 GB)")
        (base_dir() / "hotovo.txt").write_text("ok")
        _st.update(state="ready", step="Doplněk je nainstalovaný.", log="")
    except Exception as e:                                            # noqa: BLE001
        _st.update(state="error", step=f"Instalace se nepovedla: {e}")


# ---------------- AI kreslíř (překreslení do kreslené podoby) ----------------
def kresba_installed() -> bool:
    return (base_dir() / "kresba.txt").exists() and python().exists()


def _env_k() -> dict:
    env = _env()
    env["PYTHONPATH"] = str(base_dir() / "kresba_libs")
    env["PREKLIZKA_KRESBA_MODEL"] = str(base_dir() / "kresba_model")
    return env


def kresba_status() -> dict:
    s = dict(_stk)
    s["available"] = os.name == "nt" or bool(os.environ.get("PREKLIZKA_3D_TEST"))
    if kresba_installed() and s["state"] != "installing":
        s["state"] = "ready"
    if _st["gpu"] is None:
        _st["gpu"] = has_nvidia() if s["available"] else False
    s["gpu"] = _st["gpu"]
    s["size"] = ("zhruba 3,5 GB" if (base_dir() / "torch.txt").exists() else "zhruba 6 GB") if s["gpu"] else "zhruba 3,5 GB"
    return s


def _install_k() -> None:
    global _cur
    _cur = _stk
    try:
        (base_dir() / "kresba.txt").unlink(missing_ok=True)
        _runtime(5)
        _run(_pip() + ["--target", str(base_dir() / "kresba_libs"), "--upgrade"] + LIBS_K, "4/5 Instaluji knihovny kreslíře")
        _copy_app()
        _models("repaint.py", base_dir() / "kresba_model", "5/5 Stahuji model kreslíře (asi 2,9 GB)")
        (base_dir() / "kresba.txt").write_text("ok")
        _stk.update(state="ready", step="AI kreslíř je nainstalovaný.", log="")
    except Exception as e:                                            # noqa: BLE001
        _stk.update(state="error", step=f"Instalace se nepovedla: {e}")


def start_install_k() -> dict:
    with _lock:
        if _stk["state"] == "installing" or _st["state"] == "installing":
            return kresba_status()
        _stk.update(state="installing", step="Začínám…", log="")
        threading.Thread(target=_install_k, daemon=True).start()
    return kresba_status()


def run_kresba(job: dict, img_png: bytes | None, ctrl_png: bytes | None) -> bytes:
    """Spustí repaint.py v doplňku. job: mode, prompt, strength, w, h (bez cest k souborům)."""
    if not kresba_installed():
        raise RuntimeError("AI kreslíř není nainstalovaný.")
    app = base_dir() / "app"
    if not (app / "repaint.py").exists() or (src_dir() / "repaint.py").exists() and \
            (src_dir() / "repaint.py").read_bytes() != (app / "repaint.py").read_bytes():
        _copy_app()
    import json
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        job = dict(job, out=str(t / "out.png"))
        if img_png:
            (t / "img.png").write_bytes(img_png)
            job["img"] = str(t / "img.png")
        if ctrl_png:
            (t / "ctrl.png").write_bytes(ctrl_png)
            job["ctrl"] = str(t / "ctrl.png")
        (t / "job.json").write_text(json.dumps(job), encoding="utf-8")
        r = subprocess.run([str(python()), str(app / "repaint.py"), str(t / "job.json")], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=_env_k(), timeout=1800, creationflags=NO_WINDOW)
        if r.returncode != 0 or not (t / "out.png").exists():
            tail = (r.stdout + "\n" + r.stderr).strip().splitlines()[-6:]
            raise RuntimeError("AI kreslíř selhal: " + " | ".join(tail))
        return (t / "out.png").read_bytes()


def start_install() -> dict:
    with _lock:
        if _st["state"] == "installing":
            return status()
        _st.update(state="installing", step="Začínám…", log="")
        threading.Thread(target=_install, daemon=True).start()
    return status()


def run(png: bytes, resolution: int = 256) -> bytes:
    if not installed():
        raise RuntimeError("Doplněk není nainstalovaný.")
    app = base_dir() / "app"
    if not (app / "run3d.py").exists() or (src_dir() / "run3d.py").exists() and \
            (src_dir() / "run3d.py").read_bytes() != (app / "run3d.py").read_bytes():
        if src_dir().exists():                                        # nová verze appky = nová verze skriptu
            shutil.rmtree(app, ignore_errors=True)
            shutil.copytree(src_dir(), app)
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.png", Path(td) / "out.glb"
        inp.write_bytes(png)
        env = _env()
        env["HF_HUB_OFFLINE"] = "1"
        r = subprocess.run([str(python()), str(app / "run3d.py"), str(inp), str(out), str(resolution)], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env, timeout=1800, creationflags=NO_WINDOW)
        if r.returncode != 0 or not out.exists():
            tail = (r.stdout + "\n" + r.stderr).strip().splitlines()[-6:]
            raise RuntimeError("3D model se nepodařilo vytvořit: " + " | ".join(tail))
        return out.read_bytes()


def run_b64(img_b64: str) -> str:
    return base64.b64encode(run(base64.b64decode(img_b64.split(",")[-1]))).decode()
