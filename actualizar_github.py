#!/usr/bin/env python3
"""
Script para actualizar GitHub: sube ejecutables y actualiza release notes
"""
import os
import sys
from pathlib import Path

# Intentar importar requests
try:
    import requests
except ImportError:
    print("Instalando requests...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
        import requests
    except Exception as e:
        print(f"Error instalando requests: {e}")
        print("Por favor instala requests manualmente: pip install requests")
        sys.exit(1)

# Configuración
PROJECT = Path("/home/zarkentroska/Documentos/adas3")
REPO = "zarkentroska/ADAS3-Server"
RELEASE_TAG = "v0.5Alpha"

# Leer token desde variable de entorno o archivo local (no versionado)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    # Intentar leer desde archivo local (no versionado)
    token_file = PROJECT / ".github_token"
    if token_file.exists():
        try:
            GITHUB_TOKEN = token_file.read_text().strip()
        except Exception:
            pass

if not GITHUB_TOKEN:
    print("ERROR: GITHUB_TOKEN no encontrado")
    print("Configura el token de una de estas formas:")
    print("  1. Variable de entorno: export GITHUB_TOKEN='tu_token'")
    print("  2. Archivo local: echo 'tu_token' > .github_token")
    sys.exit(1)

os.chdir(PROJECT)

print("="*70)
print("  ACTUALIZANDO GITHUB - ADAS3 Server")
print("="*70)

headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

# 1. Buscar ejecutables
print("\n[1/5] Buscando ejecutables...")
win_exe = None
lin_exe = None

for f in ["ADAS3-Server-0.5Alpha-win-x64.exe", "ADAS3-Server-0.5Alpha-win-x64.exe.exe", "dist/DetectorDrones.exe"]:
    p = PROJECT / f
    if p.exists():
        win_exe = p
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  ✓ Windows: {f} ({size_mb:.1f} MB)")
        break

for f in ["Adas3_Server_LInux_x64_v05", "dist/DetectorDrones"]:
    p = PROJECT / f
    if p.exists():
        lin_exe = p
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  ✓ Linux: {f} ({size_mb:.1f} MB)")
        break

if not win_exe and not lin_exe:
    print("  ✗ No se encontraron ejecutables")
    sys.exit(1)

# 2. Obtener release
print("\n[2/5] Obteniendo release...")
try:
    r = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{RELEASE_TAG}", 
                    headers=headers, timeout=15)
    if r.status_code != 200:
        print(f"  ✗ Error {r.status_code}: {r.text[:200]}")
        sys.exit(1)
    release_id = r.json()["id"]
    print(f"  ✓ Release ID: {release_id}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# 3. Eliminar assets antiguos
print("\n[3/5] Eliminando assets antiguos...")
try:
    assets = requests.get(f"https://api.github.com/repos/{REPO}/releases/{release_id}/assets",
                         headers=headers, timeout=15).json()
    for a in assets:
        name = a["name"]
        if name in ["ADAS3-Server-0.5Alpha-win-x64.exe", "ADAS3-Server-0.5Alpha-linux-x64", 
                   "adas3-server-0.5alpha_amd64.deb"]:
            r = requests.delete(f"https://api.github.com/repos/{REPO}/releases/assets/{a['id']}",
                          headers=headers, timeout=15)
            if r.status_code == 204:
                print(f"  ✓ Eliminado: {name}")
except Exception as e:
    print(f"  ⚠ Error eliminando: {e}")

# 4. Subir nuevos ejecutables
print("\n[4/5] Subiendo ejecutables...")
upload_hdrs = {**headers, "Content-Type": "application/octet-stream"}

if win_exe:
    try:
        print(f"  Subiendo {win_exe.name}... (esto puede tardar varios minutos)")
        with open(win_exe, "rb") as f:
            r = requests.post(
                f"https://uploads.github.com/repos/{REPO}/releases/{release_id}/assets?name=ADAS3-Server-0.5Alpha-win-x64.exe",
                headers=upload_hdrs, data=f, timeout=600
            )
        if r.status_code == 201:
            print(f"  ✓ Windows subido correctamente")
        else:
            print(f"  ✗ Error: {r.status_code} - {r.text[:200]}")
    except Exception as e:
        print(f"  ✗ Error subiendo Windows: {e}")

if lin_exe:
    try:
        print(f"  Subiendo {lin_exe.name}... (esto puede tardar varios minutos)")
        with open(lin_exe, "rb") as f:
            r = requests.post(
                f"https://uploads.github.com/repos/{REPO}/releases/{release_id}/assets?name=ADAS3-Server-0.5Alpha-linux-x64",
                headers=upload_hdrs, data=f, timeout=600
            )
        if r.status_code == 201:
            print(f"  ✓ Linux subido correctamente")
        else:
            print(f"  ✗ Error: {r.status_code} - {r.text[:200]}")
    except Exception as e:
        print(f"  ✗ Error subiendo Linux: {e}")

# 5. Actualizar notas
print("\n[5/5] Actualizando notas del release...")
notes = """## ADAS3 Server v0.5 Alpha

### 🚀 Descarga

#### Windows
- **ADAS3-Server-0.5Alpha-win-x64.exe** - Ejecutable único para Windows 10/11
  - Incluye todas las dependencias
  - GPU completa (CUDA/NVIDIA)
  - Ejecutable único (~650 MB)

#### Linux
- **ADAS3-Server-0.5Alpha-linux-x64** - Ejecutable único para Linux (Ubuntu/Debian)
  - Incluye todas las dependencias
  - Modo CPU-only (sin GPU, más pequeño ~1-1.5 GB)
  - Ejecutable único, arranque rápido
  - Dar permisos de ejecución: `chmod +x ADAS3-Server-0.5Alpha-linux-x64`

### ✨ Características principales

* **Detección de drones con YOLO** - Modelos personalizables
* **Análisis de audio con TensorFlow** - Detección de drones por sonido
* **Integración TinySA Ultra** - Análisis de espectro RF
* **Tailscale VPN** - Conexiones remotas seguras
* **Soporte multi-idioma** - 5 idiomas (ES, EN, FR, IT, PT)
* **Interfaz OpenCV** - Visualización en tiempo real

### 📋 Requisitos

* **Windows 10/11** o **Linux** (Ubuntu/Debian)
* **4GB RAM mínimo** (8GB+ recomendado)
* **GPU opcional** (aceleración en Windows)

### 🎮 Uso

1. Descarga el ejecutable para tu sistema
2. En Linux: `chmod +x ADAS3-Server-0.5Alpha-linux-x64`
3. Ejecuta el programa
4. Conecta con el cliente Android ADAS3

### 📝 Notas

* Los ejecutables incluyen todas las dependencias (no requiere Python instalado)
* En Linux, el ejecutable es CPU-only para reducir tamaño
* En Windows, incluye soporte GPU completo"""

try:
    r = requests.patch(
        f"https://api.github.com/repos/{REPO}/releases/{release_id}",
        headers={**headers, "Content-Type": "application/json"},
        json={"body": notes},
        timeout=15
    )
    if r.status_code == 200:
        print("  ✓ Notas actualizadas correctamente")
    else:
        print(f"  ✗ Error: {r.status_code} - {r.text[:200]}")
except Exception as e:
    print(f"  ✗ Error actualizando notas: {e}")

print("\n" + "="*70)
print("  ¡COMPLETADO!")
print("="*70)
print(f"\nRepositorio: https://github.com/{REPO}")
print(f"Release: https://github.com/{REPO}/releases/tag/{RELEASE_TAG}")
print("\n" + "="*70)
print("\nNOTA: Este script solo actualiza los ejecutables y las notas del release.")
print("Para actualizar el código en el repositorio, ejecuta:")
print("  cd /home/zarkentroska/Documentos/adas3")
print("  git add -A")
print("  git commit -m 'Actualización: mejoras en Tailscale, optimizaciones de build y nuevas características'")
print("  git push\n")


