#!/bin/bash
# Script para construir un paquete .deb de ADAS3

set -e  # Salir si hay algún error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Construyendo paquete .deb para ADAS3 ===${NC}"

# Directorio base
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# Información del paquete
PACKAGE_NAME="adas3"
VERSION="1.0.0"
ARCHITECTURE="amd64"
MAINTAINER="ADAS3 Team"
DESCRIPTION="Sistema de detección de drones ADAS3"

# Directorios temporales
BUILD_DIR="$BASE_DIR/deb_build"
DEB_ROOT="$BUILD_DIR/${PACKAGE_NAME}_${VERSION}"
DEBIAN_DIR="$DEB_ROOT/DEBIAN"
BIN_DIR="$DEB_ROOT/usr/bin"
SHARE_DIR="$DEB_ROOT/usr/share/${PACKAGE_NAME}"
APP_DIR="$DEB_ROOT/usr/share/applications"
ICONS_DIR="$DEB_ROOT/usr/share/pixmaps"

# Limpiar builds anteriores
echo -e "${YELLOW}Limpiando builds anteriores...${NC}"
rm -rf "$BUILD_DIR"
rm -rf "$BASE_DIR/dist"
rm -rf "$BASE_DIR/build"

# Compilar con PyInstaller
echo -e "${YELLOW}Compilando con PyInstaller...${NC}"
if ! command -v pyinstaller &> /dev/null; then
    echo -e "${RED}Error: pyinstaller no está instalado. Instálalo con: pip install pyinstaller${NC}"
    exit 1
fi

pyinstaller --clean --noconfirm pyinstaller.spec

# En Linux, PyInstaller crea una carpeta en lugar de un solo archivo
if [ -d "$BASE_DIR/dist/DetectorDrones" ]; then
    # Modo onedir (Linux)
    EXECUTABLE_PATH="$BASE_DIR/dist/DetectorDrones/DetectorDrones"
    EXECUTABLE_DIR="$BASE_DIR/dist/DetectorDrones"
elif [ -f "$BASE_DIR/dist/DetectorDrones" ]; then
    # Modo onefile (Windows)
    EXECUTABLE_PATH="$BASE_DIR/dist/DetectorDrones"
    EXECUTABLE_DIR=""
else
    echo -e "${RED}Error: No se generó el ejecutable${NC}"
    exit 1
fi

if [ ! -f "$EXECUTABLE_PATH" ]; then
    echo -e "${RED}Error: No se encontró el ejecutable en $EXECUTABLE_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Ejecutable compilado correctamente${NC}"

# Crear estructura de directorios
echo -e "${YELLOW}Creando estructura de directorios...${NC}"
mkdir -p "$DEBIAN_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$SHARE_DIR"
mkdir -p "$APP_DIR"
mkdir -p "$ICONS_DIR"

# Copiar ejecutable y dependencias
echo -e "${YELLOW}Copiando ejecutable y dependencias...${NC}"
if [ -n "$EXECUTABLE_DIR" ]; then
    # Modo onedir: copiar toda la carpeta
    cp -r "$EXECUTABLE_DIR"/* "$SHARE_DIR/"
    # Crear script wrapper que ejecuta desde el directorio correcto
    cat > "$BIN_DIR/${PACKAGE_NAME}" << 'WRAPPER_EOF'
#!/bin/bash
# Wrapper para ADAS3
cd /usr/share/adas3
exec ./DetectorDrones "$@"
WRAPPER_EOF
    chmod +x "$BIN_DIR/${PACKAGE_NAME}"
    chmod +x "$SHARE_DIR/DetectorDrones"
else
    # Modo onefile: copiar solo el ejecutable
    cp "$EXECUTABLE_PATH" "$BIN_DIR/${PACKAGE_NAME}"
    chmod +x "$BIN_DIR/${PACKAGE_NAME}"
fi

# Copiar icono si existe
if [ -f "$BASE_DIR/icon.ico" ]; then
    echo -e "${YELLOW}Copiando icono...${NC}"
    # Convertir .ico a .png si es necesario (requiere ImageMagick o similar)
    if command -v convert &> /dev/null; then
        convert "$BASE_DIR/icon.ico" "$ICONS_DIR/${PACKAGE_NAME}.png" 2>/dev/null || \
        cp "$BASE_DIR/icon.ico" "$ICONS_DIR/${PACKAGE_NAME}.ico"
    else
        # Si no hay convert, intentar copiar directamente
        cp "$BASE_DIR/icon.ico" "$ICONS_DIR/${PACKAGE_NAME}.ico" 2>/dev/null || true
    fi
fi

# Crear archivo .desktop
echo -e "${YELLOW}Creando archivo .desktop...${NC}"
cat > "$APP_DIR/${PACKAGE_NAME}.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ADAS3
Comment=${DESCRIPTION}
Exec=${PACKAGE_NAME}
Icon=${PACKAGE_NAME}
Terminal=false
Categories=Utility;Security;
EOF
chmod +x "$APP_DIR/${PACKAGE_NAME}.desktop"

# Crear archivo de control
echo -e "${YELLOW}Creando archivo de control...${NC}"
cat > "$DEBIAN_DIR/control" << EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCHITECTURE}
Maintainer: ${MAINTAINER}
Description: ${DESCRIPTION}
 Sistema de detección de drones ADAS3 con capacidades de:
 - Detección visual usando YOLO
 - Detección de audio usando IA
 - Análisis de espectro RF con TinySA
 - Integración con Tailscale para acceso remoto
Depends: libc6, libstdc++6
EOF

# Crear script de postinst (opcional)
cat > "$DEBIAN_DIR/postinst" << 'EOF'
#!/bin/bash
# Script post-instalación
set -e

# Actualizar base de datos de aplicaciones
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database /usr/share/applications 2>/dev/null || true
fi

# Actualizar iconos
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache /usr/share/pixmaps 2>/dev/null || true
fi

echo "ADAS3 instalado correctamente."
exit 0
EOF
chmod +x "$DEBIAN_DIR/postinst"

# Crear script de prerm (opcional)
cat > "$DEBIAN_DIR/prerm" << 'EOF'
#!/bin/bash
# Script pre-remoción
set -e
exit 0
EOF
chmod +x "$DEBIAN_DIR/prerm"

# Crear script de postrm (opcional)
cat > "$DEBIAN_DIR/postrm" << 'EOF'
#!/bin/bash
# Script post-remoción
set -e

# Actualizar base de datos de aplicaciones
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database /usr/share/applications 2>/dev/null || true
fi

# Actualizar iconos
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache /usr/share/pixmaps 2>/dev/null || true
fi

exit 0
EOF
chmod +x "$DEBIAN_DIR/postrm"

# Calcular el tamaño instalado (después de copiar todos los archivos)
echo -e "${YELLOW}Calculando tamaño instalado...${NC}"
INSTALLED_SIZE=$(du -sk "$DEB_ROOT" | cut -f1)
sed -i "/^Description:/i Installed-Size: ${INSTALLED_SIZE}" "$DEBIAN_DIR/control"

# Construir el paquete .deb
echo -e "${YELLOW}Construyendo paquete .deb...${NC}"
cd "$BUILD_DIR"

# Verificar que dpkg-deb está disponible
if ! command -v dpkg-deb &> /dev/null; then
    echo -e "${RED}Error: dpkg-deb no está instalado. Instálalo con: sudo apt-get install dpkg-dev${NC}"
    exit 1
fi

dpkg-deb --build "${PACKAGE_NAME}_${VERSION}"

# Mover el .deb al directorio base
DEB_FILE="${PACKAGE_NAME}_${VERSION}_${ARCHITECTURE}.deb"
mv "${PACKAGE_NAME}_${VERSION}.deb" "$BASE_DIR/$DEB_FILE"

echo -e "${GREEN}=== Paquete .deb creado exitosamente ===${NC}"
echo -e "${GREEN}Archivo: $BASE_DIR/$DEB_FILE${NC}"
echo -e "${YELLOW}Para instalar: sudo dpkg -i $DEB_FILE${NC}"
echo -e "${YELLOW}Si hay dependencias faltantes: sudo apt-get install -f${NC}"

# Limpiar (opcional, comentar si quieres mantener los archivos)
# echo -e "${YELLOW}Limpiando archivos temporales...${NC}"
# rm -rf "$BUILD_DIR"

