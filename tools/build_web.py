"""Složí web/index.html (z page.html + core.js) a web/3d.html (z model3d.html + knihoven ve web/vendor).
Oba soubory jsou samostatné a fungují offline."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "web"
version = (root.parent / "VERSION").read_text(encoding="utf-8").strip()
HEAD = '<!doctype html>\n<html lang="cs"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>\n'


THEME = (root / "theme.css").read_text(encoding="utf-8") if (root / "theme.css").exists() else ""


def write(name: str, html: str) -> None:
    if THEME:                                                # vzhled „oko“ za původní styly, ať je přebije
        i = html.find("</style>")
        html = html[:i] + "\n" + THEME + html[i:] if i >= 0 else "<style>" + THEME + "</style>" + html
    (root / name).write_text(HEAD + html + "\n</body></html>\n", encoding="utf-8")
    print(f"web/{name}:", len(html) // 1024, "kB")


page = (root / "page.html").read_text(encoding="utf-8")
core = (root / "core.js").read_text(encoding="utf-8")
write("index.html", page.replace("/*CORE*/", core).replace("__VERSION__", version))

# 3D režim (jen Preklizka.exe): three.js + loadery + JSZip vložené přímo do stránky
if "--bez-3d" in sys.argv:                                   # Docker: 3D režim se na serveru nepoužívá
    sys.exit(0)
vendor = ["three.min.js", "OrbitControls.js", "OBJLoader.js", "STLLoader.js", "GLTFLoader.js", "PLYLoader.js", "jszip.min.js"]
libs = "\n".join("<script>\n" + (root / "vendor" / f).read_text(encoding="utf-8").replace("</script", "<\\/script") + "\n</script>" for f in vendor)
m3d = (root / "model3d.html").read_text(encoding="utf-8")
write("3d.html", m3d.replace("<!--VENDOR-->", libs).replace("__VERSION__", version))
