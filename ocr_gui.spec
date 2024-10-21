# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ocr_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('.\\env4simpleLatexOcr\\Lib\\site-packages\\latex2mathml','latex2mathml'),
	('.\\models','models')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FMatPix',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='guiicon.ico'
)
