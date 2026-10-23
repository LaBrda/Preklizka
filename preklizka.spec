# PyInstaller: jeden soubor Preklizka.exe
# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['app/desktop.py'],
    pathex=['app'],
    datas=[('web/index.html', 'web'), ('VERSION', '.')],
    hiddenimports=['server', 'vrstvy', 'updater', 'uvicorn.logging', 'uvicorn.loops.auto', 'uvicorn.protocols.http.auto',
                   'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan.on'],
    excludes=['tkinter', 'matplotlib', 'IPython', 'pandas'],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='Preklizka', console=True, upx=False,
          icon=None, version=None)
