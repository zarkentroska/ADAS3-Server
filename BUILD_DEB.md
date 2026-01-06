# Construcción de paquete .deb para ADAS3

Este documento explica cómo construir un paquete `.deb` de ADAS3 que puede instalarse en cualquier sistema Linux sin necesidad de tener librerías Python instaladas.

## Requisitos

Para construir el paquete `.deb`, necesitas:

1. **PyInstaller**: `pip install pyinstaller`
2. **dpkg-dev**: `sudo apt-get install dpkg-dev`
3. **ImageMagick** (opcional): Para convertir iconos `.ico` a `.png`

## Construcción

Ejecuta el script de construcción:

```bash
./build_deb.sh
```

El script realizará los siguientes pasos:

1. Limpiará builds anteriores
2. Compilará el ejecutable con PyInstaller
3. Creará la estructura de directorios del paquete `.deb`
4. Copiará el ejecutable y recursos necesarios
5. Creará los archivos de control y scripts de instalación
6. Generará el paquete `.deb`

## Instalación

Una vez generado el paquete `.deb`, puedes instalarlo en cualquier sistema Linux:

```bash
sudo dpkg -i adas3_1.0.0_amd64.deb
```

Si hay dependencias faltantes (poco probable ya que el ejecutable es autocontenido), ejecuta:

```bash
sudo apt-get install -f
```

## Desinstalación

Para desinstalar el paquete:

```bash
sudo dpkg -r adas3
```

## Estructura del paquete

El paquete instala:

- `/usr/bin/adas3` - Ejecutable principal
- `/usr/share/applications/adas3.desktop` - Entrada en el menú de aplicaciones
- `/usr/share/pixmaps/adas3.png` o `.ico` - Icono de la aplicación

## Notas

- El ejecutable es autocontenido e incluye todas las dependencias Python necesarias
- El paquete está configurado para arquitectura `amd64` (x86_64)
- Puedes modificar la versión y otros metadatos editando las variables al inicio del script `build_deb.sh`


