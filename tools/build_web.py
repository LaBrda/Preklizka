"""Složí web/index.html z web/page.html a web/core.js (jeden soubor, funguje i offline)."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "web"
page = (root / "page.html").read_text(encoding="utf-8")
core = (root / "core.js").read_text(encoding="utf-8")
version = (root.parent / "VERSION").read_text(encoding="utf-8").strip()
html = page.replace("/*CORE*/", core).replace("__VERSION__", version)
(root / "index.html").write_text(
    '<!doctype html>\n<html lang="cs"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>\n'
    + html + "\n</body></html>\n", encoding="utf-8")
print("web/index.html:", len(html) // 1024, "kB")
