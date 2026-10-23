"""Preklizka.exe – spustí appku lokálně a otevře ji v prohlížeči. Nic se neinstaluje."""
from __future__ import annotations

import multiprocessing
import os
import socket
import sys
import threading
import time
import webbrowser


def free_port(preferred: int) -> int:
    # stejný port i po aktualizaci (stránka v prohlížeči se jen obnoví); stará verze ho může chvíli držet
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


def selftest() -> int:
    import numpy as np
    import vrstvy
    lab = -np.ones((120, 100), np.int16)
    lab[20:100, 20:80] = 1
    lab[40:60, 40:60] = 2
    res = vrstvy.build(lab, [[20, 20, 20], [200, 100, 50], [250, 250, 250]], ["outline", 2, 2],
                       {"layers": 3, "width": 100}, None, None)
    ok = res["svg"].startswith("<?xml") and len(res["layers"]) == 3
    print("selftest", "OK" if ok else "FAIL")
    return 0 if ok else 1


def main() -> None:
    multiprocessing.freeze_support()
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    os.environ["PREKLIZKA_DESKTOP"] = "1"
    import updater
    threading.Thread(target=updater.cleanup_old, daemon=True).start()
    port = free_port(int(os.environ.get("PREKLIZKA_PORT", "8765")))
    os.environ["PREKLIZKA_PORT"] = str(port)
    url = f"http://127.0.0.1:{port}/"
    import uvicorn
    import server
    print("Překližkové vrstvy", server.VERSION)
    print("Appka běží na", url)
    print("Toto okno nech otevřené. Zavřením appku ukončíš.")
    if os.environ.get("PREKLIZKA_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
