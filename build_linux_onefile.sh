#!/usr/bin/env bash
set -euo pipefail

# Construye un ejecutable único (onefile) para Linux usando PyInstaller
# Uso:
#   chmod +x build_linux_onefile.sh
#   ./build_linux_onefile.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PY=""
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PY="$PROJECT_ROOT/.venv/bin/python"
  echo "[BUILD] Usando venv existente: $PROJECT_ROOT/.venv"
else
  # Venv de build (evita PEP668 / externally-managed-environment)
  BUILD_VENV="$PROJECT_ROOT/.pyinstaller_venv"
  if [[ ! -x "$BUILD_VENV/bin/python" ]]; then
    echo "[BUILD] Creando venv de build en: $BUILD_VENV"
    python3 -m venv "$BUILD_VENV"
  fi
  PY="$BUILD_VENV/bin/python"
  echo "[BUILD] Usando venv de build: $BUILD_VENV"
fi

echo "[BUILD] Python: $PY"
"$PY" -V

if ! "$PY" -c "import PyInstaller" >/dev/null 2>&1; then
  echo "[BUILD] Instalando PyInstaller en la venv..."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install pyinstaller
fi

echo "[BUILD] Limpiando artefactos previos..."
rm -rf build/ dist/

echo "[BUILD] Compilando con spec (onedir en Linux para arranque rápido)..."
"$PY" -m PyInstaller --noconfirm pyinstaller.spec

OUT_DIR="$PROJECT_ROOT/dist"
BIN_ONEFILE="$OUT_DIR/DetectorDrones"
BIN_ONEDIR="$OUT_DIR/DetectorDrones/DetectorDrones"

if [[ -f "$BIN_ONEFILE" ]]; then
  # onefile (Windows)
  chmod +x "$BIN_ONEFILE"
  echo "[BUILD] Ejecutable único generado:"
  echo "       $BIN_ONEFILE"
  echo
  echo "[BUILD] Puedes copiarlo donde quieras y abrirlo con doble click."
elif [[ -f "$BIN_ONEDIR" ]]; then
  # onedir (Linux - más rápido al arrancar)
  chmod +x "$BIN_ONEDIR"
  echo "[BUILD] Carpeta ejecutable generada (onedir - arranque rápido):"
  echo "       $OUT_DIR/DetectorDrones/"
  echo
  echo "[BUILD] Ejecutable: $BIN_ONEDIR"
  echo "[BUILD] Puedes ejecutar el programa directamente o crear un acceso directo al ejecutable."
  echo "[BUILD] Para distribuir, copia toda la carpeta 'DetectorDrones' completa."
else
  echo "[BUILD] Error: No se encontró ejecutable generado."
fi

echo "[BUILD] Hecho."


