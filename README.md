# ADAS3 Server

Sistema de detección de drones con análisis de audio y video.

**Versión: v0.5 Alpha**

## Repositorios del Proyecto ADAS3

Este proyecto está dividido en dos repositorios:

- **[ADAS3 Server](https://github.com/zarkentroska/ADAS3-Server)** (este repositorio) - Servidor del sistema
- **[ADAS3 Client](https://github.com/zarkentroska/ADAS3-Client)** - Cliente del sistema

## Actualizar el repositorio

Cada vez que modifiques archivos y quieras actualizar GitHub, ejecuta:

```bash
git add .
git commit -m "Descripción de los cambios"
git push
```

O usa el script helper: `./actualizar_github.sh "Descripción de los cambios"`

## Estructura del Proyecto

- `testcam.py` - Script principal
- `pyinstaller.spec` - Configuración de empaquetado
- Modelos y archivos de configuración en la raíz del proyecto

## Notas

- La carpeta `.venv/` está excluida del repositorio (ver `.gitignore`)
- Los archivos compilados de PyInstaller también están excluidos

