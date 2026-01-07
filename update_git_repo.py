#!/usr/bin/env python3
"""
Actualiza el repositorio Git: agrega archivos nuevos y elimina duplicados innecesarios
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT = Path("/home/zarkentroska/Documentos/adas3")
os.chdir(PROJECT)

print("="*70)
print("  ACTUALIZANDO REPOSITORIO GIT - ADAS3 Server")
print("="*70)

def run_git_command(cmd, check=True):
    """Ejecuta un comando git y retorna el resultado"""
    try:
        result = subprocess.run(
            ["git"] + cmd,
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error ejecutando git {' '.join(cmd)}: {e.stderr}")
        return e
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

# 1. Verificar estado
print("\n[1/6] Verificando estado de Git...")
status = run_git_command(["status", "--porcelain"], check=False)
if status and status.stdout:
    print("  Archivos modificados/no rastreados encontrados")
else:
    print("  Sin cambios detectados")

# 2. Eliminar archivos duplicados innecesarios
print("\n[2/6] Limpiando archivos duplicados...")
files_to_remove = [
    "do_update.py",
    "execute_update.py", 
    "github_api_update.py",
    "run_update.py",
    "update_github_complete.py",
    "update_github.py",
    "upload_to_github.py",
    "actualizar_github.sh",
    "update_github.sh",
]

removed = []
for f in files_to_remove:
    p = PROJECT / f
    if p.exists():
        try:
            # Primero eliminarlo de git si está rastreado
            run_git_command(["rm", "--cached", f], check=False)
            # Luego eliminarlo del sistema
            p.unlink()
            removed.append(f)
            print(f"  ✓ Eliminado: {f}")
        except Exception as e:
            print(f"  ⚠ No se pudo eliminar {f}: {e}")

if not removed:
    print("  ℹ No hay archivos duplicados para eliminar")

# 3. Agregar archivos nuevos importantes
print("\n[3/6] Agregando archivos nuevos...")
files_to_add = [
    "actualizar_github.py",  # El script final
    "build_linux_onefile.sh",
    "build_windows.bat",
    "pyinstaller.spec",
    "testcam.py",
    ".gitignore",
    "ghlogo.png",
    "language_config.json",
    "tailscale_config.json",
    "yolo_models_config.json",
    "README.md",
    "LICENSE",
]

added = []
for f in files_to_add:
    p = PROJECT / f
    if p.exists():
        result = run_git_command(["add", f], check=False)
        if result and result.returncode == 0:
            added.append(f)
            print(f"  ✓ Agregado: {f}")

# 4. Agregar todos los archivos nuevos (excepto los ignorados)
print("\n[4/6] Agregando todos los cambios...")
result = run_git_command(["add", "-A"], check=False)
if result and result.returncode == 0:
    print("  ✓ Todos los cambios agregados")
else:
    print("  ⚠ Error agregando cambios")

# 5. Verificar qué se va a commitear
print("\n[5/6] Verificando cambios para commit...")
status = run_git_command(["status", "--short"], check=False)
if status and status.stdout and status.stdout.strip():
    print("  Cambios a commitear:")
    for line in status.stdout.strip().split('\n'):
        if line.strip():
            print(f"    {line}")
else:
    print("  ℹ No hay cambios para commitear")
    print("\n  ✓ Repositorio ya está actualizado")
    sys.exit(0)

# 6. Commit y Push
print("\n[6/6] Haciendo commit y push...")
commit_msg = "Actualización: mejoras en Tailscale, optimizaciones de build, nuevos scripts y configuración"

# Commit
result = run_git_command(["commit", "-m", commit_msg], check=False)
if result and result.returncode == 0:
    print("  ✓ Commit realizado")
    if result.stdout:
        print(f"    {result.stdout.strip()[:100]}")
else:
    if result and "nothing to commit" in result.stdout.lower():
        print("  ℹ No hay cambios para commitear")
    else:
        print("  ⚠ Error en commit")
        if result and result.stderr:
            print(f"    {result.stderr.strip()[:200]}")

# Push
print("\n  Haciendo push...")
result = run_git_command(["push"], check=False)
if result and result.returncode == 0:
    print("  ✓ Push realizado exitosamente")
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            if line.strip() and not line.startswith('Enumerating'):
                print(f"    {line}")
else:
    # Intentar con origin main/master
    for branch in ["main", "master"]:
        result = run_git_command(["push", "origin", branch], check=False)
        if result and result.returncode == 0:
            print(f"  ✓ Push realizado a origin/{branch}")
            break
    else:
        print("  ⚠ Error en push")
        if result and result.stderr:
            print(f"    {result.stderr.strip()[:200]}")

print("\n" + "="*70)
print("  ¡COMPLETADO!")
print("="*70)
print(f"\nRepositorio: https://github.com/zarkentroska/ADAS3-Server\n")

