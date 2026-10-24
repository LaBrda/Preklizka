# PyInstaller: jeden soubor Preklizka.exe (vlastní okno, bez konzole)
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules

binaries = collect_dynamic_libs('onnxruntime')
datas = [('web/index.html', 'web'), ('web/3d.html', 'web'), ('VERSION', '.'), ('addon3d', 'addon3d')] + collect_data_files('onnxruntime') + collect_data_files('webview')
hidden = ['server', 'vrstvy', 'updater', 'ai', 'addon3d', 'commons', 'commons', 'uvicorn.logging', 'uvicorn.loops.auto', 'uvicorn.protocols.http.auto',
          'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan.on', 'clr'] + collect_submodules('webview')

a = Analysis(
    ['app/desktop.py'],
    pathex=['app'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=['tkinter', 'matplotlib', 'IPython', 'pandas', 'torch', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'gi'],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='Preklizka', console=False, upx=False,
          icon=None, version=None)
