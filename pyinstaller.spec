# -*- mode: python ; coding: utf-8 -*-
"""
Especificación personalizada de PyInstaller para empaquetar testcam.py
incluyendo todos los recursos y modelos necesarios.

Uso:
    pyinstaller --noconfirm pyinstaller.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# __file__ no está definido cuando PyInstaller ejecuta el spec mediante exec(),
# así que usamos el directorio actual desde el que se invoca el comando.
BASE_DIR = os.path.abspath(os.getcwd())
ICON_PATH = os.path.join(BASE_DIR, "icon.ico")

# Verificar que el icono existe, si no, usar None
if not os.path.exists(ICON_PATH):
    print(f"Advertencia: No se encontró {ICON_PATH}")
    ICON_PATH = None
else:
    # Convertir a ruta absoluta normalizada para evitar problemas
    # En Windows, usar barras normales y asegurar que sea absoluta
    ICON_PATH = os.path.normpath(os.path.abspath(ICON_PATH))
    # En Windows, convertir barras invertidas a barras normales para PyInstaller
    if os.name == 'nt':
        ICON_PATH = ICON_PATH.replace('\\', '/')
    print(f"Usando icono: {ICON_PATH}")

RESOURCE_FILES = [
    "best.pt",
    "drone_audio_model.h5",
    "audio_mean.npy",
    "audio_std.npy",
    "config_camara.json",
    "tinysa_advanced_intervals.json",
    "yolo_models_config.json",
    "__best.pt",
    # Archivos de Tailscale
    "tailscale-setup.exe",
    "tailscale-installer.sh",
    # Iconos de audio
    "vol.png",
    "mute.png",
    # settings.png se incluye después para evitar conflictos con el icono
    "settings.png",
]

datas = []
for resource in RESOURCE_FILES:
    src = os.path.join(BASE_DIR, resource)
    if os.path.exists(src):
        datas.append((src, "."))

# No recoger datos automáticamente - PyInstaller detectará automáticamente
# los archivos necesarios mediante análisis estático del código
# Esto evita problemas con archivos grandes o demasiados archivos

hiddenimports = [
    "matplotlib",
    "matplotlib.backends.backend_agg",
    "cv2",
    "ultralytics",
    "ultralytics.models",
    "ultralytics.utils",
    "librosa",
    "tensorflow",
    "pyaudio",
    "serial",
    "serial.tools.list_ports",
]

# No usar collect_submodules para evitar problemas con demasiados módulos
# PyInstaller debería detectar automáticamente las importaciones necesarias

binaries = []

block_cipher = None

a = Analysis(
    ["testcam.py"],
    pathex=[BASE_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# En Linux, usar onedir en lugar de onefile para evitar problemas con archivos grandes
# En Windows, se puede usar onefile
if os.name == 'nt':
    # Windows: onefile
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name="DetectorDrones",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=ICON_PATH if ICON_PATH and os.path.exists(ICON_PATH) else None,
    )
else:
    # Linux: onedir (carpeta con ejecutable y dependencias)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="DetectorDrones",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="DetectorDrones",
    )

