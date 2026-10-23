"""Preklizka.exe – spustí appku lokálně ve vlastním okně (Edge WebView2). Nic se neinstaluje.

Když okno nejde otevřít (chybí WebView2), otevře se appka v prohlížeči.
Parametr --prohlizec vynutí prohlížeč, --selftest ověří sestavení (CI).
"""
from __future__ import annotations

import multiprocessing
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def app_dir() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent


def quiet_streams() -> None:
    # exe bez konzole nemá stdout/stderr; logy jdou do souboru vedle exe
    if sys.stdout is None or sys.stderr is None:
        try:
            f = open(app_dir() / "Preklizka.log", "a", encoding="utf-8", buffering=1)
        except OSError:
            f = open(os.devnull, "w")
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f


def free_port(preferred: int) -> int:
    # stejný port i po aktualizaci; stará verze ho může chvíli držet
    for _ in range(40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                time.sleep(0.25)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Log(list):
    """Průběžně zapisuje do selftest.log (v CI je vidět, kde se to zaseklo)."""
    def append(self, x):
        super().append(x)
        try:
            (app_dir() / "selftest.log").write_text("\n".join(self), encoding="utf-8")
        except OSError:
            pass


def selftest() -> int:
    out = _Log()
    ok = True
    out.append("start")
    try:
        import numpy as np
        import vrstvy
        lab = -np.ones((120, 100), np.int16)
        lab[20:100, 20:80] = 1
        lab[40:60, 40:60] = 2
        res = vrstvy.build(lab, [[20, 20, 20], [200, 100, 50], [250, 250, 250]], ["outline", 2, 2],
                           {"layers": 3, "width": 100}, None, None)
        good = res["svg"].startswith("<?xml") and len(res["layers"]) == 3
        out.append(f"vrstvy {'OK' if good else 'FAIL'}")
        ok &= good
        import ai
        out.append(f"AI zařízení: {ai.device_info()}")
        import onnxruntime as ort
        out.append(f"onnxruntime {ort.__version__} {ort.get_available_providers()}")
        if sys.platform == "win32":
            out.append("import webview…")
            import webview  # noqa: F401
            out.append("pywebview OK")
        if "--ai" in sys.argv:                               # stáhnout malý model a zkusit zvětšení
            os.environ["PREKLIZKA_MODELS"] = str(app_dir() / "Preklizka_modely")
            out.append("stahuji model…")
            for _ in range(240):
                st = ai.prepare(["esrgan-fast.onnx"])
                if st["ready"] or st.get("error"):
                    break
                time.sleep(0.5)
            img = np.zeros((40, 50, 4), np.uint8)
            img[..., 3] = 255
            img[10:30, 10:40, 0] = 200
            out.append(f"model {st}")
            up = ai.upscale(img, "fast")
            good = up.shape == (160, 200, 4)
            out.append(f"AI zvětšení {'OK' if good else 'FAIL'} {up.shape}")
            ok &= good
            car = ai.cartoon(np.dstack([np.random.default_rng(0).integers(0, 255, (60, 80, 3), np.uint8), np.full((60, 80), 255, np.uint8)]), "none", 4)
            out.append(f"zjednodušení {car.shape}")
    except Exception as e:                                   # noqa: BLE001
        import traceback
        out.append("CHYBA " + repr(e))
        out.append(traceback.format_exc())
        ok = False
    out.append("selftest " + ("OK" if ok else "FAIL"))
    txt = "\n".join(out)
    try:
        (app_dir() / "selftest.log").write_text(txt, encoding="utf-8")
    except OSError:
        pass
    try:
        print(txt)
    except Exception:                                        # noqa: BLE001
        pass
    out.append("konec, ukončuji proces")
    return 0 if ok else 1


class Api:
    """Funkce volané ze stránky (window.pywebview.api.*)."""

    def __init__(self):
        self.window = None

    def save_file(self, name, content):
        import webview
        res = self.window.create_file_dialog(webview.SAVE_DIALOG, save_filename=name, file_types=("SVG (*.svg)", "Všechny soubory (*.*)"))
        if not res:
            return ""
        path = res if isinstance(res, str) else res[0]
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


def wait_up(port: int, timeout: float = 30) -> bool:
    t = time.time()
    while time.time() - t < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.15)
    return False


def main() -> None:
    multiprocessing.freeze_support()
    quiet_streams()
    if "--selftest" in sys.argv:
        code = selftest()
        sys.stdout.flush()
        import updater
        updater.hard_exit(code)
    os.environ["PREKLIZKA_DESKTOP"] = "1"
    import updater
    threading.Thread(target=updater.cleanup_old, daemon=True).start()
    port = free_port(int(os.environ.get("PREKLIZKA_PORT", "8765")))
    os.environ["PREKLIZKA_PORT"] = str(port)
    url = f"http://127.0.0.1:{port}/"

    use_window = "--prohlizec" not in sys.argv and os.environ.get("PREKLIZKA_BROWSER") != "1"
    webview = None
    if use_window:
        try:
            import webview  # type: ignore
        except Exception as e:                               # noqa: BLE001
            print("Okno nejde otevřít:", e)
            use_window = False
    os.environ["PREKLIZKA_WINDOW"] = "1" if use_window else "0"

    import uvicorn
    import server
    print("Překližkové vrstvy", server.VERSION, "na", url)

    if not use_window:
        if os.environ.get("PREKLIZKA_NO_BROWSER") != "1":
            threading.Timer(1.2, lambda: webbrowser.open(url)).start()
        uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")
        return

    cfg = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    srv.install_signal_handlers = lambda: None               # běží mimo hlavní vlákno
    threading.Thread(target=srv.run, daemon=True).start()
    wait_up(port)
    api = Api()
    try:
        win = webview.create_window(f"Překližkové vrstvy v{server.VERSION}", url, js_api=api,
                                    width=1440, height=920, min_size=(900, 600), text_select=True)
        api.window = win
        webview.start(private_mode=False)
    except Exception as e:                                   # noqa: BLE001  WebView2 chybí -> prohlížeč
        print("Okno selhalo, otevírám prohlížeč:", e)
        os.environ["PREKLIZKA_WINDOW"] = "0"
        webbrowser.open(url)
        while True:
            time.sleep(3600)
    updater.hard_exit(0)                                     # okno zavřeno -> konec appky i výpočtů


if __name__ == "__main__":
    main()
