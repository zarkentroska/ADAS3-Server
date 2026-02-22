# -*- mode: python ; coding: utf-8 -*-
"""
Especificación personalizada de PyInstaller para empaquetar testcam.py
"""

import os
import glob
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

BASE_DIR = os.path.abspath(os.getcwd())
ICON_PATH = os.path.join(BASE_DIR, "adas3.ico")

if not os.path.exists(ICON_PATH):
    print(f"Advertencia: No se encontró {ICON_PATH}")
    ICON_PATH = None
else:
    ICON_PATH = os.path.normpath(os.path.abspath(ICON_PATH))
    if os.name == 'nt':
        ICON_PATH = ICON_PATH.replace('\\', '/')

# --- 1. INCLUSIÓN DE TUS MÓDULOS AUTOMÁTICAMENTE ---
modules_path = os.path.join(BASE_DIR, "modules", "*.py")
custom_modules = []
for file in glob.glob(modules_path):
    module_name = os.path.basename(file).replace(".py", "")
    if module_name != "__init__":
        custom_modules.append(f"modules.{module_name}")
print(f"[BUILD] Detectados {len(custom_modules)} módulos personalizados en la carpeta modules/")

torch_submodules = []
try:
    torch_submodules = collect_submodules('torch', recursive=True)
except Exception as e:
    torch_submodules = ['torch.distributed', 'torch.utils.data', 'torch.utils._python_dispatch']

# --- 2. RECOPILACIÓN DE LIBRERÍAS REBELDES (TF, Keras, SciPy, Librosa) ---
try:
    keras_subs = collect_submodules('keras', recursive=True)
    tf_subs = collect_submodules('tensorflow', recursive=True)
    scipy_subs = collect_submodules('scipy', recursive=True)
    librosa_subs = collect_submodules('librosa', recursive=True)
    
    # Librosa y Scipy necesitan sus propios archivos de datos (modelos internos, configs)
    librosa_datas = collect_data_files('librosa')
    scipy_datas = collect_data_files('scipy')
    
    print("[BUILD] Recopilados submódulos y datos de TF, Keras, SciPy y Librosa con éxito.")
except Exception as e:
    print(f"[BUILD] Advertencia al recopilar librerías complejas: {e}")
    keras_subs, tf_subs, scipy_subs, librosa_subs = [], [], [], []
    librosa_datas, scipy_datas = [], []

RESOURCE_FILES = [
    "models/best.pt",
    "models/drone_audio_model.h5",
    "models/audio_mean.npy",
    "models/audio_std.npy",
    "models/__best.pt",
    "installers/tailscale-setup.exe",
    "installers/tailscale-installer.sh",
    "assets/icons/vol.png",
    "assets/icons/mute.png",
    "assets/icons/settings.png",
    "assets/icons/ghlogo.png",
]

datas = []
if os.path.exists(os.path.join(BASE_DIR, "adas3.ico")):
    datas.append((os.path.join(BASE_DIR, "adas3.ico"), "."))

for resource in RESOURCE_FILES:
    src = os.path.join(BASE_DIR, resource)
    if os.path.exists(src):
        if resource.startswith("assets/icons/"):
            datas.append((src, "assets/icons"))
        elif resource.startswith("models/"):
            datas.append((src, "models"))
        elif resource.startswith("installers/"):
            datas.append((src, "installers"))
        else:
            datas.append((src, "."))

try:
    datas += collect_data_files('matplotlib', includes=['mpl-data/**'])
except:
    pass

# --- 3. AÑADIMOS LOS DATOS RECOPILADOS DE LAS LIBRERÍAS REBELDES ---
datas.extend(librosa_datas)
datas.extend(scipy_datas)

hiddenimports = [
    "matplotlib", "matplotlib.backends.backend_agg", "cv2", "ultralytics",
    "ultralytics.models", "ultralytics.utils", "soundfile",
    "numba", "pyaudio", "serial", "serial.tools.list_ports",
]

# --- 4. AÑADIMOS LOS SUBMÓDULOS RECOPILADOS ---
hiddenimports.extend(torch_submodules)
hiddenimports.extend(custom_modules)
hiddenimports.extend(keras_subs)
hiddenimports.extend(tf_subs)
hiddenimports.extend(scipy_subs)
hiddenimports.extend(librosa_subs)

if os.name == 'nt':
    excludes = []
else:
    excludes = ["triton", "polars", "_polars_runtime_32", "pytest", "IPython", "jupyter", "sklearn", "nvidia"]

a = Analysis(
    ["testcam.py"],
    pathex=[BASE_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

if os.name == 'nt':
    # WINDOWS: Creamos un ejecutable pero EN MODO DIRECTORIO (ideal para el instalador)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="ADAS3",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False, # <--- ¡CAMBIADO A TRUE PARA CAZAR ERRORES!
        icon=ICON_PATH,
    )
    # COLLECT agrupa todo en una carpeta en dist/ADAS3
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='ADAS3',
    )
else:
    # LINUX 
    exe = EXE(
        pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
        name="ADAS3", debug=False, bootloader_ignore_signals=False,
        strip=True, upx=False, upx_exclude=[], runtime_tmpdir=None,
        console=True, # <--- ¡También cambiado aquí por precaución!
        disable_windowed_traceback=False, argv_emulation=False,
        target_arch=None, codesign_identity=None, entitlements_file=None,
        icon=ICON_PATH,
    )
    coll = None
