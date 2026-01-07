# -*- mode: python ; coding: utf-8 -*-
"""
Especificación personalizada de PyInstaller para empaquetar testcam.py
incluyendo todos los recursos y modelos necesarios.

Uso:
    pyinstaller --noconfirm pyinstaller.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# Recopilar todos los submódulos de torch para asegurar inclusión completa (especialmente en Windows)
torch_submodules = []
try:
    torch_submodules = collect_submodules('torch', recursive=True)
    print(f"[BUILD] Recopilados {len(torch_submodules)} submódulos de torch")
except Exception as e:
    print(f"[BUILD] Advertencia al recopilar submódulos de torch: {e}")
    # Fallback: lista manual de módulos críticos
    torch_submodules = [
        'torch.distributed',
        'torch.distributed.*',
        'torch.utils.data',
        'torch.utils.data.*',
        'torch.utils._python_dispatch',
    ]

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
    "drone_audio_model.tflite",
    "audio_mean.npy",
    "audio_std.npy",
    "language_config.json",
    "tailscale_config.json",
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
    # Logo de GitHub para la UI
    "ghlogo.png",
]

datas = []
for resource in RESOURCE_FILES:
    src = os.path.join(BASE_DIR, resource)
    if os.path.exists(src):
        datas.append((src, "."))

# No recoger datos automáticamente - PyInstaller detectará automáticamente
# los archivos necesarios mediante análisis estático del código
# Esto evita problemas con archivos grandes o demasiados archivos

# Incluir datos necesarios de matplotlib (mpl-data) para evitar errores de fuentes/estilos
try:
    mpl_datas = collect_data_files('matplotlib', includes=['mpl-data/**'])
    datas += mpl_datas
except Exception as e:
    print(f"Advertencia: no se pudieron recopilar datos de matplotlib: {e}")

# Construir lista de hiddenimports
hiddenimports = [
    "matplotlib",
    "matplotlib.backends.backend_agg",
    "cv2",
    "ultralytics",
    "ultralytics.models",
    "ultralytics.utils",
    "librosa",
    "soundfile",
    "numba",
    "tensorflow",
    "pyaudio",
    "serial",
    "serial.tools.list_ports",
]

# Agregar todos los submódulos de torch (ya recopilados arriba)
hiddenimports.extend(torch_submodules)
print(f"[BUILD] Total de hiddenimports: {len(hiddenimports)} (incluyendo {len(torch_submodules)} de torch)")

# Excluir módulos innecesarios
# En Windows: NO excluir nada (mantener todo para máxima compatibilidad)
# En Linux: excluir módulos no usados para reducir tamaño (CPU-only)
if os.name == 'nt':  # Windows - NO excluir nada
    excludes = [
        # Solo herramientas de desarrollo opcionales que no afectan funcionalidad
        "pytest",
        "sphinx",
        "pydoc",
    ]
else:  # Linux - excluir módulos no usados
    excludes = [
        # Triton - compilador JIT de PyTorch, no necesario en runtime
        "triton",
        # Polars - no se usa
        "polars",
        "_polars_runtime_32",
        # Módulos de desarrollo y testing
        "pytest",
        "doctest",
        "test",
        "tests",
        # Jupyter/IPython
        "IPython",
        "jupyter",
        "notebook",
        # Documentación
        "sphinx",
        "pydoc",
        # Herramientas de desarrollo
        "setuptools",
        "wheel",
        # scipy - excluir módulos grandes no usados
        "scipy.sparse.csgraph",
        "scipy.spatial",
        "scipy.optimize",
        "scipy.integrate",
        "scipy.special",
        "scipy.stats",
        "scipy.ndimage",
        "scipy.signal",
        "scipy.io",
        # sklearn - no se usa en el código
        "sklearn",
        "sklearn.*",
        # matplotlib - excluir backends no usados (solo se usa 'Agg')
        "matplotlib.backends.backend_gtk3agg",
        "matplotlib.backends.backend_gtk3cairo",
        "matplotlib.backends.backend_gtk4agg",
        "matplotlib.backends.backend_gtk4cairo",
        "matplotlib.backends.backend_qt5agg",
        "matplotlib.backends.backend_qt5cairo",
        "matplotlib.backends.backend_tkagg",
        "matplotlib.backends.backend_webagg",
        "matplotlib.backends._backend_pdf_ps",
        "matplotlib.backends._backend_svg",
        # TensorFlow - excluir solo herramientas de desarrollo
        "tensorflow.python.tools",
        "tensorflow.python.debug",
        # PyTorch - excluir solo módulos de desarrollo
        "torch.testing",
        "torch.utils.tensorboard",
        "torch.utils.bottleneck",
        # OpenCV - excluir módulos no usados
        "cv2.qt",
        # CUDA/NVIDIA - excluir todas las librerías CUDA (CPU-only en Linux)
        "nvidia",
        "nvidia.cublas",
        "nvidia.cuda_cupti",
        "nvidia.cuda_nvcc",
        "nvidia.cuda_nvrtc",
        "nvidia.cuda_runtime",
        "nvidia.cudnn",
        "nvidia.cufft",
        "nvidia.curand",
        "nvidia.cusolver",
        "nvidia.cusparse",
        "nvidia.nccl",
        "nvidia.nvjitlink",
        "nvidia.nvtx",
        "nvidia.nvshmem",
        "nvidia.cufile",
    ]

binaries = []

block_cipher = None

# Los submódulos de torch ya se agregaron a hiddenimports arriba

a = Analysis(
    ["testcam.py"],
    pathex=[BASE_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filtrar librerías CUDA solo en Linux (CPU-only)
# En Windows: mantener todas las librerías CUDA (GPU completa)
if os.name != 'nt':  # Solo en Linux
    # Linux: CPU-only - mantener mínimas de PyTorch, excluir pesadas del sistema
    filtered_binaries = []
    for name, path, typ in a.binaries:
        name_lower = name.lower()
        path_lower = path.lower() if path else ""
        
        # Mantener librerías CUDA mínimas de PyTorch (necesarias para que no falle el import)
        if 'libtorch_cuda' in name_lower or 'libc10_cuda' in name_lower:
            # Incluir estas para que PyTorch pueda cargar
            filtered_binaries.append((name, path, typ))
            continue
        
        # Excluir librerías NVIDIA del sistema (pesadas y no necesarias para CPU)
        if 'nvidia' in path_lower and '/nvidia/' in path_lower:
            continue  # Excluir nvidia/cudnn, nvidia/cublas, etc.
        if 'cudnn' in name_lower or ('cudnn' in path_lower and 'nvidia' in path_lower):
            continue
        if 'cublas' in name_lower or ('cublas' in path_lower and 'nvidia' in path_lower):
            continue
        if 'cufft' in name_lower and 'nvidia' in path_lower:
            continue
        if 'libcuda.so' in name_lower:  # Librería del sistema, no necesaria
            continue
        
        # Incluir todo lo demás
        filtered_binaries.append((name, path, typ))
    
    a.binaries = filtered_binaries
# Windows: no filtrar, mantener todas las librerías CUDA (GPU completa)
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
    # Linux: onefile CPU-only (sin GPU, más pequeño y comprimido)
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
        strip=True,  # Stripping reduce el tamaño
        upx=False,  # UPX puede causar problemas, deshabilitado
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = None

