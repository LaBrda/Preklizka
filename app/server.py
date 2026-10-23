"""Webový server appky Překližkové vrstvy (běží v Dockeru)."""
from __future__ import annotations

import base64
import os
import threading
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import vrstvy

VERSION = os.environ.get("APP_VERSION", "dev")
WATCHTOWER_URL = os.environ.get("WATCHTOWER_URL", "")          # např. http://watchtower:8080
WATCHTOWER_TOKEN = os.environ.get("WATCHTOWER_TOKEN", "")
UPDATE_PASSWORD = os.environ.get("UPDATE_PASSWORD", "")
WEB = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Překližkové vrstvy", docs_url=None, redoc_url=None)
_lock = threading.Semaphore(int(os.environ.get("MAX_PARALLEL", "1")))   # nezahltit server


class BuildIn(BaseModel):
    W: int
    H: int
    cent: list
    roles: list
    opt: dict
    lab: str
    line: str
    hole: str


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    return {"ok": True, "version": VERSION, "update": bool(WATCHTOWER_URL and WATCHTOWER_TOKEN),
            "updateNeedsPassword": bool(UPDATE_PASSWORD)}


@app.post("/api/build")
def build(inp: BuildIn):
    n = inp.W * inp.H
    if n <= 0 or n > 12_000_000:
        raise HTTPException(400, "Obrázek má nepodporovanou velikost.")
    try:
        lab = np.frombuffer(base64.b64decode(inp.lab), np.int8).astype(np.int16).reshape(inp.H, inp.W)
        line = np.frombuffer(base64.b64decode(inp.line), np.uint8).reshape(inp.H, inp.W)
        hole = np.frombuffer(base64.b64decode(inp.hole), np.uint8).reshape(inp.H, inp.W)
    except Exception:
        raise HTTPException(400, "Poškozená data obrázku.")
    with _lock:
        try:
            res = vrstvy.build(lab, inp.cent, inp.roles, inp.opt, line, hole)
        except ValueError as e:
            raise HTTPException(422, str(e))
    st = res["stats"]
    res["stats"] = {k: (float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v) for k, v in st.items()}
    return JSONResponse(res)


@app.post("/api/update")
def update(x_update_password: str = Header(default="")):
    if not (WATCHTOWER_URL and WATCHTOWER_TOKEN):
        raise HTTPException(501, "Aktualizace z appky není nastavená (chybí Watchtower).")
    if UPDATE_PASSWORD and x_update_password != UPDATE_PASSWORD:
        raise HTTPException(403, "Špatné heslo.")
    try:
        # Watchtower stáhne novou verzi a appku restartuje; odpověď může přijít až po restartu
        r = httpx.post(f"{WATCHTOWER_URL}/v1/update", headers={"Authorization": f"Bearer {WATCHTOWER_TOKEN}"}, timeout=5)
        return {"ok": r.status_code < 400, "status": r.status_code}
    except httpx.TimeoutException:
        return {"ok": True, "status": "running"}
    except Exception as e:
        raise HTTPException(502, f"Watchtower neodpovídá: {e}")
