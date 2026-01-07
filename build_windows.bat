@echo off
REM Construye un ejecutable único (onefile) para Windows usando PyInstaller
REM Uso:
REM   build_windows.bat

set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

REM Verificar primero el venv del usuario (más común)
if exist "%USERPROFILE%\.venv\Scripts\python.exe" (
    set PYTHON_PATH=%USERPROFILE%\.venv\Scripts\python.exe
    echo [BUILD] Usando venv del usuario: %USERPROFILE%\.venv
    goto :found_python
)

REM Verificar si existe .venv en el directorio del proyecto
if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
    set PYTHON_PATH=%PROJECT_ROOT%.venv\Scripts\python.exe
    echo [BUILD] Usando venv del proyecto: %PROJECT_ROOT%.venv
    goto :found_python
)

REM Intentar usar el Python activo en la terminal (verificar que sea venv)
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM Verificar que el python activo sea de un venv
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.prefix)" 2^>nul') do set PYTHON_PREFIX=%%i
    python -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set PYTHON_PATH=python
        echo [BUILD] Usando Python del venv activo en la terminal: %PYTHON_PREFIX%
        goto :found_python
    )
)

REM Si no se encuentra, mostrar error
echo [BUILD] Error: No se encontró un venv con Python
echo [BUILD] Buscados en:
echo [BUILD]   - %USERPROFILE%\.venv\Scripts\python.exe
echo [BUILD]   - %PROJECT_ROOT%.venv\Scripts\python.exe
echo [BUILD]   - Python activo en terminal (no era un venv)
echo [BUILD]
echo [BUILD] Activa tu venv primero o crea uno con: python -m venv %USERPROFILE%\.venv
exit /b 1

:found_python

echo [BUILD] Python: %PYTHON_PATH%
"%PYTHON_PATH%" -V

REM Verificar PyInstaller
"%PYTHON_PATH%" -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [BUILD] Instalando PyInstaller en la venv...
    "%PYTHON_PATH%" -m pip install --upgrade pip
    "%PYTHON_PATH%" -m pip install pyinstaller
)

echo [BUILD] Limpiando artefactos previos...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [BUILD] Compilando con spec (onefile en Windows con GPU completa)...
"%PYTHON_PATH%" -m PyInstaller --noconfirm pyinstaller.spec

set OUT_DIR=%PROJECT_ROOT%dist
set BIN_ONEFILE=%OUT_DIR%\DetectorDrones.exe

if exist "%BIN_ONEFILE%" (
    echo [BUILD] Ejecutable único generado:
    echo        %BIN_ONEFILE%
    echo.
    echo [BUILD] Puedes ejecutarlo directamente o distribuir el .exe
) else (
    echo [BUILD] Error: No se encontró ejecutable generado.
    exit /b 1
)

echo [BUILD] Hecho.

