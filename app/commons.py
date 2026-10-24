"""Hledání fotek na Wikimedia Commons pro 3D režim (jen Preklizka.exe): pohledy auta z boku, zepředu…

Commons u každé fotky vyžaduje uvést autora a licenci, proto je vracíme s obrázkem.
"""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse

API = "https://commons.wikimedia.org/w/api.php"
UA = "Preklizka/1.0 (https://github.com/LaBrda/Preklizka)"   # Wikimedia chce popisný User-Agent (jen ASCII)
THUMB, BIG = 330, 1280                                       # standardní velikosti náhledů Commons


def _text(v: str) -> str:
    """HTML z metadat Commons (autor bývá odkaz) -> čistý text."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", v or ""))).strip()


def parse(data: dict) -> list[dict]:
    """Odpověď API (generator=search + imageinfo) -> seznam fotek s náhledem, autorem a licencí."""
    pages = sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        if not ii.get("url") or not str(ii.get("mime", "image/jpeg")).startswith("image/"):
            continue
        meta = ii.get("extmetadata") or {}
        big = ii.get("thumburl") or ii["url"]
        thumb = big.replace(f"/{BIG}px-", f"/{THUMB}px-") if f"/{BIG}px-" in big else big
        out.append({
            "title": p.get("title", "").removeprefix("File:"),
            "thumb": thumb,
            "img": big,
            "w": ii.get("thumbwidth") or ii.get("width"),
            "h": ii.get("thumbheight") or ii.get("height"),
            "author": _text(meta.get("Artist", {}).get("value", "")) or "neznámý autor",
            "license": _text(meta.get("LicenseShortName", {}).get("value", "")) or "viz stránka souboru",
            "licenseUrl": meta.get("LicenseUrl", {}).get("value", ""),
            "page": ii.get("descriptionurl", ""),
        })
    return out


def search(q: str, limit: int = 24) -> list[dict]:
    import httpx
    params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": f"{q} filetype:bitmap",
              "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
              "iiurlwidth": BIG, "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl"}
    r = httpx.get(API, params=params, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    return parse(r.json())


def allowed(url: str) -> bool:
    u = urlparse(url)
    return u.scheme == "https" and (u.hostname or "").endswith(".wikimedia.org")


def fetch(url: str) -> tuple[bytes, str]:
    """Stáhne obrázek z Commons (stránka v exe ho kvůli CORS nemůže dát do canvasu sama)."""
    import httpx
    if not allowed(url):
        raise ValueError("Jen obrázky z Wikimedia Commons.")
    r = httpx.get(url, headers={"User-Agent": UA}, timeout=30, follow_redirects=False)
    r.raise_for_status()
    ct = r.headers.get("content-type", "")
    if not ct.startswith("image/") or len(r.content) > 25_000_000:
        raise ValueError("Tohle není obrázek.")
    return r.content, ct
