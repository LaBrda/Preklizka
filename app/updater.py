"""Samoaktualizace Preklizka.exe z GitHub Releases (repozitář LaBrda/Preklizka).

Běžící exe ve Windows nejde přepsat, ale jde přejmenovat. Postup:
stáhnout novou verzi vedle jako .new -> běžící exe přejmenovat na .old -> .new na původní jméno
-> spustit novou verzi -> starou ukončit. Nová verze při startu smaže .old.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

REPO = "LaBrda/Preklizka"
ASSET = "Preklizka.exe"


def hard_exit(code: int = 0) -> None:
    """Okamžitý konec. Ve Windows TerminateProcess: běžné ukončení čeká na úklid DLL (DirectML, .NET) a umí se zaseknout."""
    srv = sys.modules.get("server")
    if srv is not None:
        try:
            srv.kill_workers()
        except Exception:                                # noqa: BLE001
            pass
    for f in (sys.stdout, sys.stderr):
        try:
            f.flush()
        except Exception:                                # noqa: BLE001
            pass
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.TerminateProcess(k.GetCurrentProcess(), code)
    os._exit(code)


def exe_path() -> Path | None:
    return Path(sys.executable).resolve() if getattr(sys, "frozen", False) else None


def cleanup_old() -> None:
    exe = exe_path()
    if not exe:
        return
    old = exe.with_suffix(".old")
    for _ in range(20):
        try:
            if old.exists():
                old.unlink()
            return
        except OSError:
            time.sleep(0.5)


def ver_tuple(v: str):
    out = []
    for part in v.strip().lstrip("vV").split("."):
        try:
            out.append(int(part))
        except ValueError:
            out.append(0)
    return tuple(out)


def latest_release():
    r = httpx.get(f"https://api.github.com/repos/{REPO}/releases/latest",
                  headers={"Accept": "application/vnd.github+json"}, timeout=15, follow_redirects=True)
    r.raise_for_status()
    j = r.json()
    url = next((a["browser_download_url"] for a in j.get("assets", []) if a.get("name") == ASSET), None)
    return j.get("tag_name", "").lstrip("vV"), url


def _swap_and_restart(new: Path, exe: Path, port: str) -> None:
    time.sleep(0.8)                                    # nechat odejít HTTP odpověď
    old = exe.with_suffix(".old")
    try:
        if old.exists():
            old.unlink()
    except OSError:
        pass
    os.replace(exe, old)                               # běžící exe jde přejmenovat
    os.replace(new, exe)
    env = dict(os.environ, PREKLIZKA_PORT=port, PREKLIZKA_NO_BROWSER="1")
    flags = 0x00000010 if os.name == "nt" else 0       # CREATE_NEW_CONSOLE
    subprocess.Popen([str(exe)], env=env, creationflags=flags, close_fds=True)
    hard_exit(0)


def start_update(current: str) -> dict:
    exe = exe_path()
    if not exe:
        raise RuntimeError("Aktualizace jde jen ve spuštěném Preklizka.exe.")
    latest, url = latest_release()
    if not latest or ver_tuple(latest) <= ver_tuple(current):
        return {"ok": True, "status": "current", "version": current}
    if not url:
        raise RuntimeError(f"Ve verzi {latest} chybí soubor {ASSET}.")
    new = exe.with_suffix(".new")
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
        r.raise_for_status()
        with open(new, "wb") as f:
            for chunk in r.iter_bytes(1 << 16):
                f.write(chunk)
    if new.stat().st_size < 1_000_000:
        new.unlink(missing_ok=True)
        raise RuntimeError("Stažený soubor je poškozený.")
    port = os.environ.get("PREKLIZKA_PORT", "8765")
    threading.Thread(target=_swap_and_restart, args=(new, exe, port), daemon=True).start()
    return {"ok": True, "status": "restarting", "version": latest}
