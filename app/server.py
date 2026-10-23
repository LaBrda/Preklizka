"""Webový server appky Překližkové vrstvy (běží v Dockeru)."""
from __future__ import annotations

import base64
import os
import threading
from pathlib import Path

import concurrent.futures as cf
import multiprocessing as mp

import httpx
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import vrstvy

_vf = Path(__file__).resolve().parent.parent / "VERSION"
VERSION = os.environ.get("APP_VERSION") or (_vf.read_text().strip() if _vf.exists() else "dev")
if VERSION == "dev" and _vf.exists():
    VERSION = _vf.read_text().strip()
WATCHTOWER_URL = os.environ.get("WATCHTOWER_URL", "")          # např. http://watchtower:8080
WATCHTOWER_TOKEN = os.environ.get("WATCHTOWER_TOKEN", "")
UPDATE_PASSWORD = os.environ.get("UPDATE_PASSWORD", "")
WEB = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Překližkové vrstvy", docs_url=None, redoc_url=None)
_lock = threading.Semaphore(int(os.environ.get("MAX_PARALLEL", "1")))   # nezahltit server


# Každý výpočet v novém procesu: po skončení se paměť vrátí systému (jinak by appka rostla).
_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), max_tasks_per_child=1)
    return _pool


def _run_build(lab, cent, roles, opt, line, hole):
    import vrstvy as v
    return v.build(lab, cent, roles, opt, line, hole)


_latest: dict[str, int] = {}          # klient -> nejnovější číslo požadavku
_latest_lock = threading.Lock()


class BuildIn(BaseModel):
    client: str = ""
    seq: int = 0
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
    with _latest_lock:
        if inp.client:
            _latest[inp.client] = max(_latest.get(inp.client, 0), inp.seq)
            if len(_latest) > 200:
                _latest.clear()
    with _lock:
        # než na nás přišla řada, mohla přijít novější změna od stejného klienta -> tenhle výpočet přeskočit
        if inp.client and _latest.get(inp.client, 0) > inp.seq:
            raise HTTPException(409, "Zastaralý požadavek.")
        try:
            res = _get_pool().submit(_run_build, lab, inp.cent, inp.roles, inp.opt, line, hole).result(timeout=300)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except cf.process.BrokenProcessPool:
            global _pool
            _pool = None
            raise HTTPException(503, "Výpočet spadl (nejspíš došla paměť). Zkus menší obrázek nebo méně barev.")
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
