import os
import sys

# Forzar CPU-only para PyTorch cuando se ejecuta como ejecutable compilado sin GPU
# Solo en Linux (CPU-only). En Windows mantener GPU completa
# Esto evita que PyTorch intente cargar librerías CUDA que no están disponibles
# Debe hacerse ANTES de importar torch/ultralytics
if getattr(sys, 'frozen', False) and sys.platform != 'win32':  # Solo en Linux
    # Ejecutable compilado en Linux - verificar si las librerías CUDA existen
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    cuda_libs_exist = (
        os.path.exists(os.path.join(base_dir, 'torch', 'lib', 'libtorch_cuda.so')) or
        os.path.exists(os.path.join(base_dir, 'torch', 'lib', 'libc10_cuda.so'))
    )
    
    if not cuda_libs_exist:
        # No hay librerías CUDA, forzar CPU ANTES de importar PyTorch (solo Linux)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TensorFlow: deshabilitar GPU
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reducir logs de TensorFlow
        # PyTorch: forzar CPU evitando carga de librerías CUDA
        os.environ['TORCH_CUDA_ARCH_LIST'] = ''    # No compilar kernels CUDA
        # Evitar que PyTorch intente cargar CUDA
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        print("[CONFIG] Modo CPU-only activado (sin librerías CUDA en el ejecutable)")

import cv2
import time
import numpy as np
import requests
import pyaudio
import threading
import tkinter as tk
from tkinter import Tk, simpledialog, messagebox, filedialog
from tkinter import ttk
import json

# Nota: Las variables de entorno ya se configuraron arriba antes de importar torch

# Configurar XLA solo si no estamos forzando CPU
if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '-1':
    os.environ['XLA_FLAGS'] = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found=true'
    # --- FIX PARA TENSORFLOW / LIBDEVICE ---
    # Forzamos a XLA a usar el directorio del toolkit del sistema
    # que tiene la estructura correcta (nvvm/libdevice/libdevice.10.bc)
    possible_cuda_paths = [
        "/usr/lib/nvidia-cuda-toolkit",
        "/usr/lib/cuda",
        "/usr"
    ]

    for path in possible_cuda_paths:
        # Verificamos si existe la estructura que TensorFlow exige
        if os.path.exists(os.path.join(path, "nvvm/libdevice/libdevice.10.bc")):
            print(f"[CONFIG] Configurando XLA CUDA DIR a: {path}")
            os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={path}"
            break
else:
    # CPU-only: no configurar CUDA
    print("[CONFIG] Modo CPU-only activado (sin GPU)")
    os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
import subprocess
import shutil
import webbrowser
import matplotlib
# Configurar backend no interactivo para hilos (CRÍTICO para evitar crasheos)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ultralytics import YOLO
import librosa
import tensorflow as tf
import queue
import serial
import serial.tools.list_ports
import struct
import sys

# Obtener la ruta absoluta del directorio donde está este script
# Si se ejecuta desde un ejecutable de PyInstaller, usar sys._MEIPASS
# que contiene la ruta temporal donde se extraen los archivos
if getattr(sys, 'frozen', False):
    # Ejecutándose desde un ejecutable compilado
    BASE_DIR = sys._MEIPASS
else:
    # Ejecutándose como script normal
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_config_dir():
    """Obtiene el directorio persistente para archivos de configuración.
    En modo ejecutable, usa un directorio en el home del usuario.
    En modo desarrollo, usa el directorio del script."""
    if getattr(sys, 'frozen', False):
        # Ejecutable compilado: usar directorio de configuración persistente
        if os.name == 'nt':  # Windows
            config_dir = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'ADAS3')
        else:  # Linux/Mac
            config_dir = os.path.join(os.path.expanduser('~'), '.config', 'adas3')
        # Crear directorio si no existe
        os.makedirs(config_dir, exist_ok=True)
        return config_dir
    else:
        # Modo desarrollo: usar directorio del script
        return BASE_DIR

# Directorio para archivos de configuración (persistente)
CONFIG_DIR = get_config_dir()

# Rutas a archivos de configuración (se guardan en CONFIG_DIR, que es persistente)
CONFIG_FILE = os.path.join(CONFIG_DIR, "config_camara.json")
LANGUAGE_CONFIG_FILE = os.path.join(CONFIG_DIR, "language_config.json")
YOLO_MODELS_CONFIG = os.path.join(CONFIG_DIR, "yolo_models_config.json")
ADVANCED_INTERVALS_FILE = os.path.join(CONFIG_DIR, "tinysa_advanced_intervals.json")

# Rutas absolutas a los recursos (se cargan desde BASE_DIR, que está en el ejecutable)
# TAILSCALE_CONFIG_FILE eliminado - no guardamos credenciales por seguridad
TAILSCALE_INSTALLER_WIN = os.path.join(BASE_DIR, "tailscale-setup.exe")
TAILSCALE_INSTALLER_LINUX = os.path.join(BASE_DIR, "tailscale-installer.sh")
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "drone_audio_model.h5")
YOLO_DEFAULT_MODEL = os.path.join(BASE_DIR, "best.pt")
AUDIO_MEAN_PATH = os.path.join(BASE_DIR, "audio_mean.npy")
AUDIO_STD_PATH = os.path.join(BASE_DIR, "audio_std.npy")
SETTINGS_ICON_PATH = os.path.join(BASE_DIR, "settings.png")
MUTE_ICON_PATH = os.path.join(BASE_DIR, "mute.png")
VOL_ICON_PATH = os.path.join(BASE_DIR, "vol.png")

# Estado de modelos YOLO
yolo_model_path = YOLO_DEFAULT_MODEL
yolo_model_slots = []
yolo_default_slot = 0
yolo_options_thread = None
tailscale_options_thread = None
tailscale_message_lock = threading.Lock()
tailscale_message_shown = {'connecting': False, 'connected': False, 'disconnected': False}

# --- VARIABLES GLOBALES UI ---
mouse_x, mouse_y = -1, -1
click_event_pos = None
mouse_is_down = False
pending_ip_change = None
ip_dialog_thread = None
adb_connected = False
adb_message_timer = 0
last_adb_check = 0
ADB_TARGET_IP = "127.0.0.1:8080"
ADB_CHECK_INTERVAL = 5.0
last_wifi_ip = None

def mouse_handler(event, x, y, flags, param):
    """Callback para manejar eventos del ratón"""
    global mouse_x, mouse_y, click_event_pos, mouse_is_down, yolo_slider_active
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_is_down = True
        click_event_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_is_down = False
        yolo_slider_active = None


def update_stream_endpoints(ip_with_port, record_wifi=True):
    global ip_y_puerto, base_url, video_url, audio_url, last_wifi_ip
    if record_wifi and ip_with_port != ADB_TARGET_IP:
        last_wifi_ip = ip_with_port
    ip_y_puerto = ip_with_port
    base_url = f"http://{ip_y_puerto}"
    video_url = base_url + "/video"
    audio_url = base_url + "/audio"
    guardar_ip(ip_y_puerto)


def normalize_model_path(path):
    """Normaliza una ruta de modelo para que funcione tanto en desarrollo como en ejecutable compilado."""
    if not path:
        return ""
    
    # Si es una ruta absoluta que apunta a un archivo en BASE_DIR, convertirla a relativa
    if os.path.isabs(path):
        try:
            # Verificar si la ruta está dentro de BASE_DIR
            rel_path = os.path.relpath(path, BASE_DIR)
            if not rel_path.startswith('..'):
                # Está dentro de BASE_DIR, usar ruta relativa
                normalized = os.path.join(BASE_DIR, os.path.basename(path))
                if os.path.exists(normalized):
                    return normalized
        except:
            pass
        
        # Si la ruta absoluta existe, usarla
        if os.path.exists(path):
            return path
        # Si no existe, intentar buscar solo el nombre del archivo en BASE_DIR
        filename = os.path.basename(path)
        candidate = os.path.join(BASE_DIR, filename)
        if os.path.exists(candidate):
            return candidate
        return path
    
    # Si es relativa, construir la ruta completa
    full_path = os.path.join(BASE_DIR, path)
    if os.path.exists(full_path):
        return full_path
    
    # Si no existe, intentar solo el nombre del archivo
    filename = os.path.basename(path) if path else ""
    if filename:
        candidate = os.path.join(BASE_DIR, filename)
        if os.path.exists(candidate):
            return candidate
    
    return path

def load_yolo_models_config():
    """Carga o inicializa la configuración de modelos YOLO."""
    global yolo_model_slots, yolo_default_slot, yolo_model_path

    default_slots = [
        {"path": "best.pt", "description": "Modelo por defecto"},
    ] + [{"path": "", "description": ""} for _ in range(14)]

    if os.path.exists(YOLO_MODELS_CONFIG):
        try:
            with open(YOLO_MODELS_CONFIG, "r", encoding="utf-8") as f:
                data = json.load(f)
                slots = data.get("slots", [])
                while len(slots) < 15:
                    slots.append({"path": "", "description": ""})
                
                # Normalizar todas las rutas de los slots
                for slot in slots:
                    if slot.get("path"):
                        slot["path"] = normalize_model_path(slot["path"])
                        # Guardar solo el nombre del archivo si está en BASE_DIR
                        if os.path.isabs(slot["path"]):
                            try:
                                rel_path = os.path.relpath(slot["path"], BASE_DIR)
                                if not rel_path.startswith('..'):
                                    slot["path"] = os.path.basename(slot["path"])
                            except:
                                pass
                
                yolo_model_slots = slots[:15]
                yolo_default_slot = int(data.get("default_slot", 0))
        except Exception as e:
            print(f"[YOLO] No se pudo leer configuración de modelos: {e}")
            yolo_model_slots = default_slots
            yolo_default_slot = 0
    else:
        yolo_model_slots = default_slots
        yolo_default_slot = 0

    # Validar slot por defecto
    if not (0 <= yolo_default_slot < len(yolo_model_slots)):
        yolo_default_slot = 0

    default_path = yolo_model_slots[yolo_default_slot].get("path") or "best.pt"
    # Normalizar la ruta final
    default_path = normalize_model_path(default_path)
    if not default_path or not os.path.exists(default_path):
        # Fallback al modelo por defecto
        default_path = os.path.join(BASE_DIR, "best.pt")
    yolo_model_path = default_path


def save_yolo_models_config():
    """Guarda la configuración actual de modelos YOLO."""
    try:
        # Crear una copia de los slots normalizando las rutas antes de guardar
        slots_to_save = []
        for slot in yolo_model_slots:
            slot_copy = slot.copy()
            path = slot_copy.get("path", "")
            if path:
                # Si la ruta está en BASE_DIR, guardar solo el nombre del archivo
                if os.path.isabs(path):
                    try:
                        rel_path = os.path.relpath(path, BASE_DIR)
                        if not rel_path.startswith('..'):
                            # Está dentro de BASE_DIR, guardar solo el nombre
                            slot_copy["path"] = os.path.basename(path)
                        else:
                            # Está fuera, mantener la ruta absoluta
                            slot_copy["path"] = path
                    except:
                        slot_copy["path"] = os.path.basename(path) if os.path.exists(os.path.join(BASE_DIR, os.path.basename(path))) else path
                else:
                    # Ya es relativa, mantenerla
                    slot_copy["path"] = path
            slots_to_save.append(slot_copy)
        
        data = {
            "slots": slots_to_save,
            "default_slot": yolo_default_slot,
        }
        with open(YOLO_MODELS_CONFIG, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[YOLO] No se pudo guardar configuración de modelos: {e}")


load_yolo_models_config()


def get_yolo_settings_icon():
    """Carga y retorna el icono de ajustes para YOLO."""
    global yolo_settings_icon
    if yolo_settings_icon is None:
        if os.path.exists(SETTINGS_ICON_PATH):
            icon = cv2.imread(SETTINGS_ICON_PATH, cv2.IMREAD_UNCHANGED)
            if icon is not None:
                desired_size = 26
                icon = cv2.resize(icon, (desired_size, desired_size), interpolation=cv2.INTER_AREA)
                yolo_settings_icon = icon
        if yolo_settings_icon is None:
            size = 26
            fallback = np.zeros((size, size, 4), dtype=np.uint8)
            cv2.circle(fallback, (size // 2, size // 2), size // 2 - 2, (90, 90, 90, 255), -1, cv2.LINE_AA)
            yolo_settings_icon = fallback
    return yolo_settings_icon

def get_audio_volume_icon(muted=True):
    """Carga y retorna el icono de volumen (mute.png o vol.png)."""
    global mute_icon, vol_icon
    icon_path = MUTE_ICON_PATH if muted else VOL_ICON_PATH
    icon_cache = mute_icon if muted else vol_icon
    
    if icon_cache is None:
        if os.path.exists(icon_path):
            icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
            if icon is not None:
                desired_size = 24
                icon = cv2.resize(icon, (desired_size, desired_size), interpolation=cv2.INTER_AREA)
                if muted:
                    mute_icon = icon
                else:
                    vol_icon = icon
                return icon
        # Fallback si no existe el archivo
        size = 24
        fallback = np.zeros((size, size, 4), dtype=np.uint8)
        cv2.circle(fallback, (size // 2, size // 2), size // 3, (200, 200, 200, 255), 2)
        if muted:
            # Dibujar línea tachada
            cv2.line(fallback, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), (200, 200, 200, 255), 2)
            mute_icon = fallback
        else:
            vol_icon = fallback
        return fallback
    
    return icon_cache

def cargar_ip():
    """Carga la última IP guardada o retorna la por defecto"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('ip', '192.168.1.129:8080')
        except:
            pass
    return '192.168.1.129:8080'

def guardar_ip(ip):
    """Guarda la IP en el archivo de configuración"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({'ip': ip}, f)
    except Exception as e:
        print(f"Error al guardar IP: {e}")

# --- FUNCIONES TAILSCALE ---
tailscale_running = False

# NOTA: No guardamos credenciales de Tailscale porque:
# 1. Tailscale maneja la autenticación de forma persistente después del primer login
# 2. El usuario se autentica mediante OAuth en el navegador cuando ejecuta 'tailscale up'
# 3. Windows/Linux guardan la sesión automáticamente
# 4. Guardar credenciales en texto plano es un riesgo de seguridad

def verificar_estado_tailscale():
    """Verifica el estado real de Tailscale y actualiza tailscale_running"""
    global tailscale_running
    
    if not tailscale_installed():
        tailscale_running = False
        return False
    
    try:
        tailscale_cmd = get_tailscale_path()
        is_windows = os.name == 'nt'
        status_cmd = f'"{tailscale_cmd}" status' if is_windows else f'{tailscale_cmd} status'
        status_result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True, timeout=5)
        
        if status_result.returncode == 0:
            # Verificar si está conectado (tiene IP asignada o está logged in)
            if 'Logged in' in status_result.stdout or '100.' in status_result.stdout:
                tailscale_running = True
                return True
        
        tailscale_running = False
        return False
    except Exception as e:
        print(f"Error verificando estado de Tailscale: {e}")
        tailscale_running = False
        return False

def get_tailscale_username():
    """Obtiene el nombre de usuario de Tailscale"""
    try:
        tailscale_cmd = get_tailscale_path()
        is_windows = os.name == 'nt'
        cmd = f'"{tailscale_cmd}" whoami' if is_windows else f'{tailscale_cmd} whoami'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except:
        pass
    return None

def get_tailscale_ip():
    """Obtiene la IP de Tailscale de este dispositivo"""
    try:
        tailscale_cmd = get_tailscale_path()
        is_windows = os.name == 'nt'
        cmd = f'"{tailscale_cmd}" ip -4' if is_windows else f'{tailscale_cmd} ip -4'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except:
        pass
    return None

def get_tailscale_connected_devices():
    """Obtiene la lista de dispositivos conectados (online) de Tailscale"""
    devices = []
    try:
        tailscale_cmd = get_tailscale_path()
        is_windows = os.name == 'nt'
        cmd = f'"{tailscale_cmd}" status' if is_windows else f'{tailscale_cmd} status'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                # Ignorar líneas vacías o que no contengan IPs
                if not line.strip() or not line[0].isdigit():
                    continue
                
                # Verificar que la línea no contenga "offline"
                if 'offline' in line.lower():
                    continue
                
                # Extraer IP y nombre del dispositivo
                # Formato: "100.66.87.40     zarkentroska-2       kadifer1993@  linux    -"
                parts = line.split()
                if len(parts) >= 2:
                    ip = parts[0].strip()
                    device_name = parts[1].strip()
                    # Verificar que la IP tiene formato válido
                    if '.' in ip and len(ip.split('.')) == 4:
                        devices.append({'ip': ip, 'name': device_name})
    except Exception as e:
        print(f"Error obteniendo dispositivos de Tailscale: {e}")
    return devices

def get_tailscale_path():
    """Obtiene la ruta completa del ejecutable de Tailscale"""
    if os.name == 'nt':  # Windows
        # Posibles ubicaciones de Tailscale en Windows
        possible_paths = [
            r'C:\Program Files\Tailscale\tailscale.exe',
            r'C:\Program Files (x86)\Tailscale\tailscale.exe',
            os.path.expanduser(r'~\AppData\Local\Tailscale\tailscale.exe'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        # Si no se encuentra, intentar con 'tailscale' del PATH
        return 'tailscale'
    else:  # Linux/Mac
        return 'tailscale'

def tailscale_installed():
    """Verifica si Tailscale está instalado"""
    if os.name == 'nt':  # Windows
        # En Windows, verificar múltiples formas:
        # 1. Intentar el comando tailscale directamente
        try:
            result = subprocess.run('tailscale --version', shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return True
        except:
            pass
        
        # 2. Verificar si existe el ejecutable en rutas comunes
        common_paths = [
            r'C:\Program Files\Tailscale\tailscale.exe',
            r'C:\Program Files (x86)\Tailscale\tailscale.exe',
            os.path.expanduser(r'~\AppData\Local\Tailscale\tailscale.exe'),
        ]
        for path in common_paths:
            if os.path.exists(path):
                return True
        
        # 3. Verificar si el servicio de Tailscale está instalado
        try:
            result = subprocess.run('sc query Tailscale', shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'SERVICE_NAME: Tailscale' in result.stdout:
                return True
        except:
            pass
        
        return False
    else:  # Linux
        cmd = 'tailscale version'
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False

def install_tailscale():
    """Instala Tailscale en modo silencioso"""
    if os.name == 'nt':  # Windows
        installer_path = TAILSCALE_INSTALLER_WIN
        if not os.path.exists(installer_path):
            def show_not_found():
                root = Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                messagebox.showerror(t('error'), t('tailscale_installer_not_found'))
                root.destroy()
            threading.Thread(target=show_not_found, daemon=True).start()
            return False
        
        def install_thread():
            try:
                # Instalación silenciosa en Windows: /S para silent, /quiet también funciona
                # Algunos instaladores usan /SILENT o /VERYSILENT
                cmd = f'"{installer_path}" /S'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    # Esperar un poco para que la instalación se complete y el PATH se actualice
                    time.sleep(3)
                    
                    # Verificar si ahora está instalado
                    if tailscale_installed():
                        def show_success():
                            root = Tk()
                            root.withdraw()
                            root.attributes("-topmost", True)
                            messagebox.showinfo(t('tailscale_install_success'), t('tailscale_install_success'))
                            root.destroy()
                        threading.Thread(target=show_success, daemon=True).start()
                    else:
                        # Instalación exitosa pero aún no detectado (puede necesitar reiniciar)
                        def show_success_restart():
                            root = Tk()
                            root.withdraw()
                            root.attributes("-topmost", True)
                            messagebox.showinfo(t('tailscale_install_success'), 
                                              t('tailscale_install_success') + '\n\n' + 
                                              'Si no se detecta, reinicia la aplicación.')
                            root.destroy()
                        threading.Thread(target=show_success_restart, daemon=True).start()
                else:
                    raise Exception(result.stderr)
            except Exception as e:
                print(f"Error instalando Tailscale: {e}")
                def show_error():
                    root = Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    messagebox.showerror(t('error'), t('tailscale_install_error'))
                    root.destroy()
                threading.Thread(target=show_error, daemon=True).start()
        
        threading.Thread(target=install_thread, daemon=True).start()
        return True
    else:  # Linux
        installer_path = TAILSCALE_INSTALLER_LINUX
        if not os.path.exists(installer_path):
            def show_not_found():
                root = Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                messagebox.showerror(t('error'), t('tailscale_installer_not_found'))
                root.destroy()
            threading.Thread(target=show_not_found, daemon=True).start()
            return False
        
        def install_thread():
            try:
                # Hacer el script ejecutable
                os.chmod(installer_path, 0o755)
                # Ejecutar el instalador (puede requerir sudo)
                cmd = f'sudo bash "{installer_path}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    def show_success():
                        root = Tk()
                        root.withdraw()
                        root.attributes("-topmost", True)
                        messagebox.showinfo(t('tailscale_install_success'), t('tailscale_install_success'))
                        root.destroy()
                    threading.Thread(target=show_success, daemon=True).start()
                else:
                    raise Exception(result.stderr)
            except Exception as e:
                print(f"Error instalando Tailscale: {e}")
                def show_error():
                    root = Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    messagebox.showerror(t('error'), t('tailscale_install_error'))
                    root.destroy()
                threading.Thread(target=show_error, daemon=True).start()
        
        threading.Thread(target=install_thread, daemon=True).start()
        return True

def toggle_tailscale():
    """Activa o desactiva Tailscale"""
    global tailscale_running
    print("[TAILSCALE] toggle_tailscale() llamado")
    
    if not tailscale_installed():
        print("[TAILSCALE] Tailscale no está instalado")
        def show_not_installed():
            root = Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showerror(t('error'), t('tailscale_not_installed'))
            root.destroy()
        threading.Thread(target=show_not_installed, daemon=True).start()
        return
    
    def connect_tailscale():
        global tailscale_running
        print("[TAILSCALE] connect_tailscale() iniciado")
        is_windows = os.name == 'nt'
        
        # Obtener ruta completa de Tailscale
        tailscale_cmd = get_tailscale_path()
        print(f"[TAILSCALE] Usando comando: {tailscale_cmd}")
        
        try:
            # Verificar estado actual de Tailscale
            print("[TAILSCALE] Verificando estado actual...")
            status_cmd = f'"{tailscale_cmd}" status' if is_windows else f'{tailscale_cmd} status'
            status_result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True, timeout=5)
            print(f"[TAILSCALE] tailscale status returncode: {status_result.returncode}")
            print(f"[TAILSCALE] tailscale status stdout: {status_result.stdout[:200]}")
            
            # Extraer URL del status si está disponible
            auth_url = None
            import re
            if status_result.stdout:
                url_match = re.search(r'https://login\.tailscale\.com/a/[^\s\n]+', status_result.stdout)
                if url_match:
                    auth_url = url_match.group(0).strip()
                    print(f"[TAILSCALE] URL encontrada en tailscale status: {auth_url}")
            
            # Si ya está conectado, no hacer nada
            if status_result.returncode == 0:
                if 'Logged in' in status_result.stdout or '100.' in status_result.stdout:
                    print("[TAILSCALE] Ya está conectado")
                    tailscale_running = True
                    return
            
            # En Windows, ejecutar tailscale up y verificar status periódicamente para encontrar URL
            # En Linux, usar la URL del status si está disponible
            if is_windows and not auth_url:
                print("[TAILSCALE] Windows: Ejecutando tailscale up en background...")
                try:
                    # Ejecutar tailscale up en background (mostrará URL y esperará autenticación)
                    up_cmd = f'"{tailscale_cmd}" up'
                    subprocess.Popen(up_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                                     stdin=subprocess.DEVNULL)
                    print("[TAILSCALE] tailscale up iniciado en background")
                    
                    # Esperar un momento para que tailscale procese y luego verificar status para URL
                    print("[TAILSCALE] Esperando unos segundos y verificando status para URL...")
                    time.sleep(2)  # Dar tiempo para que tailscale procese
                    
                    # Verificar status nuevamente para encontrar la URL
                    status_check_cmd = f'"{tailscale_cmd}" status'
                    for _ in range(3):  # Intentar 3 veces con delay
                        status_check = subprocess.run(status_check_cmd, shell=True, capture_output=True, 
                                                      text=True, timeout=3)
                        if status_check.stdout:
                            url_match = re.search(r'https://login\.tailscale\.com/a/[^\s\n]+', status_check.stdout)
                            if url_match:
                                auth_url = url_match.group(0).strip()
                                print(f"[TAILSCALE] URL encontrada en status después de tailscale up: {auth_url}")
                                break
                        time.sleep(1)
                    
                except Exception as e:
                    print(f"[TAILSCALE] Error ejecutando tailscale up: {e}")
            
            # Si hay URL, abrir navegador
            if auth_url:
                try:
                    webbrowser.open(auth_url)
                    print(f"[TAILSCALE] Navegador abierto con URL: {auth_url}")
                except Exception as e:
                    print(f"[TAILSCALE] Error abriendo navegador: {e}")
            
            # En Linux, ejecutar tailscale up en background después de abrir navegador
            # En Windows, ya se ejecutó arriba para capturar la URL
            if not is_windows:
                print("[TAILSCALE] Linux: Ejecutando tailscale up en background...")
                try:
                    # Ejecutar tailscale up en background sin bloquear
                    up_cmd = f'{tailscale_cmd} up'
                    subprocess.Popen(up_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                    print(f"[TAILSCALE] Proceso tailscale up iniciado en background")
                except Exception as e:
                    print(f"[TAILSCALE] Error ejecutando tailscale up: {e}")
            
            # Iniciar verificación periódica
            def check_connection_periodic():
                global tailscale_running
                print(f"[TAILSCALE] Iniciando verificación periódica de conexión...")
                max_attempts = 30 if auth_url else 10  # Más tiempo si necesita autenticación
                status_cmd_check = f'"{tailscale_cmd}" status' if is_windows else f'{tailscale_cmd} status'
                for i in range(max_attempts):
                    time.sleep(1)
                    try:
                        status_result = subprocess.run(status_cmd_check, shell=True, capture_output=True, text=True, timeout=5)
                        if status_result.returncode == 0:
                            if 'Logged in' in status_result.stdout or '100.' in status_result.stdout:
                                tailscale_running = True
                                print(f"[TAILSCALE] Conectado exitosamente (intento {i+1})")
                                return
                        elif i % 5 == 0:  # Log cada 5 segundos
                            print(f"[TAILSCALE] Esperando conexión... (intento {i+1}/{max_attempts})")
                    except Exception as e:
                        print(f"[TAILSCALE] Error verificando conexión: {e}")
                print(f"[TAILSCALE] No se detectó conexión después de {max_attempts} intentos")
            
            threading.Thread(target=check_connection_periodic, daemon=True).start()
            
        except Exception as e:
            print(f"Error conectando Tailscale: {e}")
            # No mostrar error desde aquí para evitar problemas con Tkinter en hilos
    
    def disconnect_tailscale():
        global tailscale_running
        try:
            tailscale_cmd = get_tailscale_path()
            is_windows = os.name == 'nt'
            cmd = f'"{tailscale_cmd}" down' if is_windows else f'{tailscale_cmd} down'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                tailscale_running = False
                # No mostrar mensaje desde hilo secundario - el estado se refleja en la UI
            else:
                raise Exception(result.stderr)
        except Exception as e:
            print(f"Error desconectando Tailscale: {e}")
            # No mostrar error desde hilo para evitar problemas
    
    if tailscale_running:
        threading.Thread(target=disconnect_tailscale, daemon=True).start()
    else:
        threading.Thread(target=connect_tailscale, daemon=True).start()

# --- SISTEMA DE TRADUCCIONES ---
# Diccionario de traducciones
TRANSLATIONS = {
    'es': {  # Spanish (default)
        'yolo_on': 'YOLO: {0} det.',
        'yolo_off': 'YOLO: OFF',
        'tinysa_on': 'TinySA: ON',
        'tinysa_off': 'TinySA: OFF',
        'det_audio_on': 'DET AUDIO: ON',
        'det_audio_off': 'DET AUDIO: OFF',
        'ip_label': 'IP: {0}',
        'fps_label': 'FPS: {0:.1f}',
        'language_app': 'IDIOMA Y CONFIG. APP',
        'no_streaming': 'Streaming no detectado',
        'no_streaming_yolo': 'No se puede iniciar YOLO porque no hay streaming de video disponible.',
        'activate_audio_first': 'Activa primero la transmisión de sonido pulsando el icono de volumen.',
        'activate_audio_title': 'Activar transmisión de audio',
        'no_signal': 'SIN CONEXION',
        'reconnecting': 'Intentando reconectar...',
        'audio_drone_detected': 'AUDIO DRON DETECTED: {0}%',
        'no_audio_dron': 'NO AUDIO DRON: {0}%',
        'rf_drone_detected': 'DRON DETECTADO POR RF: {0:.3f} GHz ({1}%)',
        'rf_drone_detected_no_freq': 'DRON DETECTADO POR RF ({0}%)',
        'tinysa_connected': 'TinySA conectado',
        'tinysa_connected_android': 'TinySA conectado a Android',
        'adb_connected': 'ADB conectado',
        'tinysa_not_configured': 'TinySA no configurado',
        'configure_tinysa_first': 'Configura TinySA primero usando el botón de engranaje.',
        'tinysa_not_detected': 'TinySA no detectado localmente ni en el servidor Android.\nConéctalo vía USB al PC o al Android e intenta de nuevo.',
        'change_camera_ip': 'Cambiar Cámara IP',
        'enter_new_ip': 'Introduce la nueva IP y puerto:\n(actual: {0})',
        'yolo_options_title': 'Opciones de YOLO',
        'available_models': 'Modelos disponibles',
        'model': 'Modelo {0}',
        'description': 'Descripción:',
        'browse': 'Examinar',
        'select_yolo_model': 'Seleccionar modelo YOLO',
        'yolo_models': 'Modelos YOLO (*.pt)',
        'all_files': 'Todos los archivos',
        'page': 'Página {0}/{1}',
        'load_model': 'Cargar modelo',
        'load_and_save_default': 'Cargar y guardar por defecto',
        'load_default_config': 'Cargar configuración por defecto',
        'cancel': 'Cancelar',
        'model_updated': 'Modelo actualizado correctamente.',
        'error': 'Error',
        'model_empty': 'El modelo del slot {0} está vacío.',
        'file_not_found': 'No se encontró el archivo:\n{0}',
        'tinysa_mode_selection': 'TinySA - Selección de modo',
        'select_mode': 'Selecciona un modo:',
        'fpv_normal': 'FPV-Normal (2.442 GHz)',
        'fpv_alt': 'FPV-Alt (5.8 GHz)',
        'fpv_mix': 'FPV Mix (2.4 y 5.8 GHz secuencial)',
        'custom_range': 'Rango personalizado',
        'start_mhz': 'Inicio (MHz):',
        'stop_mhz': 'Fin (MHz):',
        'advanced_range': 'Rango personalizado - Intervalo avanzado',
        'ok': 'OK',
        'advanced_interval_title': 'Rango personalizado - Intervalo avanzado',
        'up_to_5_intervals': 'Hasta 5 intervalos (MHz)',
        'interval': 'Intervalo {0}:',
        'start': 'Inicio',
        'stop': 'Fin',
        'sweeps': '# barridos',
        'complete_start_end': 'Completa inicio y fin para el intervalo {0}.',
        'invalid_values': 'Valores inválidos en intervalo {0}.',
        'end_must_be_greater': 'El fin debe ser mayor que el inicio en el intervalo {0}.',
        'sweeps_must_be_positive': 'El número de barridos debe ser mayor a cero (intervalo {0}).',
        'enter_valid_interval': 'Introduce al menos un intervalo válido.',
        'enter_numeric_values': 'Introduce valores numéricos válidos para inicio y fin.',
        'end_greater_than_start': 'El fin debe ser mayor que el inicio.',
        'language_selection_title': 'Seleccionar Idioma',
        'select_language': 'Selecciona un idioma:',
        'could_not_save_language': 'No se pudo guardar el idioma.',
        'tailscale_on': 'TAILSCALE: ON',
        'tailscale_off': 'TAILSCALE: OFF',
        'tailscale_config_title': 'Configuración Tailscale',
        # 'tailscale_username' y 'tailscale_password' eliminados - no se usan (Tailscale maneja auth)
        'tailscale_not_configured': 'Tailscale no configurado',
        'configure_tailscale_first': 'Configura Tailscale primero usando el botón de engranaje.',
        'tailscale_connecting': 'Conectando a Tailscale...',
        'tailscale_disconnecting': 'Desconectando de Tailscale...',
        'tailscale_connected': 'Tailscale conectado correctamente.',
        'tailscale_disconnected': 'Tailscale desconectado correctamente.',
        'tailscale_error': 'Error al conectar/desconectar Tailscale.',
        'tailscale_not_installed': 'Tailscale no está instalado. Instálalo desde https://tailscale.com/download',
        'install_tailscale': 'Instalar servicio',
        'installing_tailscale': 'Instalando Tailscale...',
        'tailscale_install_success': 'Tailscale instalado correctamente. Reinicia la aplicación.',
        'tailscale_install_error': 'Error al instalar Tailscale.',
        'tailscale_installer_not_found': 'Instalador de Tailscale no encontrado en la ruta del proyecto.',
        'tailscale_oauth_info': 'Tailscale usa autenticación OAuth (Google, Microsoft, etc.).\nAl activar Tailscale, se abrirá tu navegador para autenticarte.',
        'tailscale_installed_info': 'Tailscale está instalado. Usa el botón TAILSCALE: ON/OFF para conectarte.',
        'tailscale_logged_in_as': 'Logueado como:',
        'tailscale_ip_device': 'IP Tailscale de este dispositivo:',
        'tailscale_other_devices': 'Otros dispositivos conectados:',
        'tailscale_create_account': 'Crear nueva cuenta Tailscale',
        'tailscale_sudo_needed': 'Tailscale requiere permisos de administrador.\n\nEjecuta una vez en la terminal:\nsudo tailscale set --operator=$USER\n\nLuego podrás usar Tailscale sin sudo.',
        'nvidia_cuda_info': 'Se recomienda si se dispone de tarjeta gráfica NVIDIA, tener los drivers actualizados. Además, en Linux se requiere NVIDIA Cuda Toolkit para correcto funcionamiento de las librerías (sudo apt install nvidia-cuda-toolkit).',
        'audio_sensitivity_label': 'Sensibilidad raw de detección de audio por Tensorflow (avanzado):',
        'audio_sensitivity_percent': '%',
        'audio_sensitivity_save_error': 'No se pudo guardar la sensibilidad de audio.',
        'audio_sensitivity_range_error': 'El valor de sensibilidad debe estar entre 1% y 100%.',
        'audio_sensitivity_number_error': 'El valor de sensibilidad debe ser un número.',
    },
    'en': {  # English
        'yolo_on': 'YOLO: {0} det.',
        'yolo_off': 'YOLO: OFF',
        'tinysa_on': 'TinySA: ON',
        'tinysa_off': 'TinySA: OFF',
        'det_audio_on': 'AUDIO DET: ON',
        'det_audio_off': 'AUDIO DET: OFF',
        'ip_label': 'IP: {0}',
        'fps_label': 'FPS: {0:.1f}',
        'language_app': 'LANGUAGE & APP CONFIG.',
        'no_streaming': 'Streaming not detected',
        'no_streaming_yolo': 'Cannot start YOLO because there is no video streaming available.',
        'activate_audio_first': 'Activate audio transmission first by clicking the volume icon.',
        'activate_audio_title': 'Activate audio transmission',
        'no_signal': 'NO SIGNAL',
        'reconnecting': 'Trying to reconnect...',
        'audio_drone_detected': 'AUDIO DRONE DETECTED: {0}%',
        'no_audio_dron': 'NO AUDIO DRONE: {0}%',
        'rf_drone_detected': 'DRONE DETECTED BY RF: {0:.3f} GHz ({1}%)',
        'rf_drone_detected_no_freq': 'DRONE DETECTED BY RF ({0}%)',
        'tinysa_connected': 'TinySA connected',
        'tinysa_connected_android': 'TinySA connected to Android',
        'adb_connected': 'ADB connected',
        'tinysa_not_configured': 'TinySA not configured',
        'configure_tinysa_first': 'Configure TinySA first using the gear button.',
        'tinysa_not_detected': 'TinySA not detected locally or on Android server.\nConnect it via USB to PC or Android and try again.',
        'change_camera_ip': 'Change Camera IP',
        'enter_new_ip': 'Enter new IP and port:\n(current: {0})',
        'yolo_options_title': 'YOLO Options',
        'available_models': 'Available Models',
        'model': 'Model {0}',
        'description': 'Description:',
        'browse': 'Browse',
        'select_yolo_model': 'Select YOLO Model',
        'yolo_models': 'YOLO Models (*.pt)',
        'all_files': 'All Files',
        'page': 'Page {0}/{1}',
        'load_model': 'Load Model',
        'load_and_save_default': 'Load and Save as Default',
        'load_default_config': 'Load Default Configuration',
        'cancel': 'Cancel',
        'model_updated': 'Model updated successfully.',
        'error': 'Error',
        'model_empty': 'Model in slot {0} is empty.',
        'file_not_found': 'File not found:\n{0}',
        'tinysa_mode_selection': 'TinySA - Mode Selection',
        'select_mode': 'Select a mode:',
        'fpv_normal': 'FPV-Normal (2.442 GHz)',
        'fpv_alt': 'FPV-Alt (5.8 GHz)',
        'fpv_mix': 'FPV Mix (2.4 and 5.8 GHz sequential)',
        'custom_range': 'Custom Range',
        'start_mhz': 'Start (MHz):',
        'stop_mhz': 'Stop (MHz):',
        'advanced_range': 'Custom Range - Advanced Interval',
        'ok': 'OK',
        'advanced_interval_title': 'Custom Range - Advanced Interval',
        'up_to_5_intervals': 'Up to 5 intervals (MHz)',
        'interval': 'Interval {0}:',
        'start': 'Start',
        'stop': 'Stop',
        'sweeps': '# sweeps',
        'complete_start_end': 'Complete start and end for interval {0}.',
        'invalid_values': 'Invalid values in interval {0}.',
        'end_must_be_greater': 'End must be greater than start in interval {0}.',
        'sweeps_must_be_positive': 'Number of sweeps must be greater than zero (interval {0}).',
        'enter_valid_interval': 'Enter at least one valid interval.',
        'enter_numeric_values': 'Enter valid numeric values for start and end.',
        'end_greater_than_start': 'End must be greater than start.',
        'language_selection_title': 'Select Language',
        'select_language': 'Select a language:',
        'could_not_save_language': 'Could not save language.',
        'tailscale_on': 'TAILSCALE: ON',
        'tailscale_off': 'TAILSCALE: OFF',
        'tailscale_config_title': 'Tailscale Configuration',
        # 'tailscale_username' y 'tailscale_password' eliminados - no se usan
        'tailscale_not_configured': 'Tailscale not configured',
        'configure_tailscale_first': 'Configure Tailscale first using the gear button.',
        'tailscale_connecting': 'Connecting to Tailscale...',
        'tailscale_disconnecting': 'Disconnecting from Tailscale...',
        'tailscale_connected': 'Tailscale connected successfully.',
        'tailscale_disconnected': 'Tailscale disconnected successfully.',
        'tailscale_error': 'Error connecting/disconnecting Tailscale.',
        'tailscale_not_installed': 'Tailscale is not installed. Install it from https://tailscale.com/download',
        'install_tailscale': 'Install Service',
        'installing_tailscale': 'Installing Tailscale...',
        'tailscale_install_success': 'Tailscale installed successfully. Restart the application.',
        'tailscale_install_error': 'Error installing Tailscale.',
        'tailscale_installer_not_found': 'Tailscale installer not found in project path.',
        'tailscale_oauth_info': 'Tailscale uses OAuth authentication (Google, Microsoft, etc.).\nWhen you activate Tailscale, your browser will open for authentication.',
        'tailscale_installed_info': 'Tailscale is installed. Use the TAILSCALE: ON/OFF button to connect.',
        'tailscale_logged_in_as': 'Logged in as:',
        'tailscale_ip_device': 'Tailscale IP of this device:',
        'tailscale_other_devices': 'Other connected devices:',
        'tailscale_create_account': 'Create new Tailscale account',
        'tailscale_sudo_needed': 'Tailscale requires administrator permissions.\n\nRun once in terminal:\nsudo tailscale set --operator=$USER\n\nThen you can use Tailscale without sudo.',
        'nvidia_cuda_info': 'It is recommended that if you have an NVIDIA graphics card, keep the drivers updated. Also, on Linux, NVIDIA CUDA Toolkit is required for proper functioning of the libraries (sudo apt install nvidia-cuda-toolkit).',
        'audio_sensitivity_label': 'Raw audio detection sensitivity by Tensorflow (advanced):',
        'audio_sensitivity_percent': '%',
        'audio_sensitivity_save_error': 'Could not save audio sensitivity.',
        'audio_sensitivity_range_error': 'Sensitivity value must be between 1% and 100%.',
        'audio_sensitivity_number_error': 'Sensitivity value must be a number.',
    },
    'fr': {  # French
        'yolo_on': 'YOLO: {0} dét.',
        'yolo_off': 'YOLO: OFF',
        'tinysa_on': 'TinySA: ON',
        'tinysa_off': 'TinySA: OFF',
        'det_audio_on': 'DET AUDIO: ON',
        'det_audio_off': 'DET AUDIO: OFF',
        'ip_label': 'IP: {0}',
        'fps_label': 'FPS: {0:.1f}',
        'language_app': 'LANGUE ET CONFIG. APP',
        'no_streaming': 'Streaming non détecté',
        'no_streaming_yolo': 'Impossible de démarrer YOLO car aucun streaming vidéo n\'est disponible.',
        'activate_audio_first': 'Activez d\'abord la transmission audio en cliquant sur l\'icône de volume.',
        'activate_audio_title': 'Activer la transmission audio',
        'no_signal': 'PAS DE SIGNAL',
        'reconnecting': 'Tentative de reconnexion...',
        'audio_drone_detected': 'DRONE AUDIO DÉTECTÉ: {0}%',
        'no_audio_dron': 'PAS DE DRONE AUDIO: {0}%',
        'rf_drone_detected': 'DRONE DÉTECTÉ PAR RF: {0:.3f} GHz ({1}%)',
        'rf_drone_detected_no_freq': 'DRONE DÉTECTÉ PAR RF ({0}%)',
        'tinysa_connected': 'TinySA connecté',
        'tinysa_connected_android': 'TinySA connecté à Android',
        'adb_connected': 'ADB connecté',
        'tinysa_not_configured': 'TinySA non configuré',
        'configure_tinysa_first': 'Configurez d\'abord TinySA en utilisant le bouton d\'engrenage.',
        'tinysa_not_detected': 'TinySA non détecté localement ni sur le serveur Android.\nConnectez-le via USB au PC ou à Android et réessayez.',
        'change_camera_ip': 'Changer IP Caméra',
        'enter_new_ip': 'Entrez le nouvel IP et port:\n(actuel: {0})',
        'yolo_options_title': 'Options YOLO',
        'available_models': 'Modèles disponibles',
        'model': 'Modèle {0}',
        'description': 'Description:',
        'browse': 'Parcourir',
        'select_yolo_model': 'Sélectionner modèle YOLO',
        'yolo_models': 'Modèles YOLO (*.pt)',
        'all_files': 'Tous les fichiers',
        'page': 'Page {0}/{1}',
        'load_model': 'Charger modèle',
        'load_and_save_default': 'Charger et enregistrer par défaut',
        'load_default_config': 'Charger configuration par défaut',
        'cancel': 'Annuler',
        'model_updated': 'Modèle mis à jour avec succès.',
        'error': 'Erreur',
        'model_empty': 'Le modèle dans l\'emplacement {0} est vide.',
        'file_not_found': 'Fichier non trouvé:\n{0}',
        'tinysa_mode_selection': 'TinySA - Sélection de mode',
        'select_mode': 'Sélectionnez un mode:',
        'fpv_normal': 'FPV-Normal (2.442 GHz)',
        'fpv_alt': 'FPV-Alt (5.8 GHz)',
        'fpv_mix': 'FPV Mix (2.4 et 5.8 GHz séquentiel)',
        'custom_range': 'Plage personnalisée',
        'start_mhz': 'Début (MHz):',
        'stop_mhz': 'Fin (MHz):',
        'advanced_range': 'Plage personnalisée - Intervalle avancé',
        'ok': 'OK',
        'advanced_interval_title': 'Plage personnalisée - Intervalle avancé',
        'up_to_5_intervals': 'Jusqu\'à 5 intervalles (MHz)',
        'interval': 'Intervalle {0}:',
        'start': 'Début',
        'stop': 'Fin',
        'sweeps': '# balayages',
        'complete_start_end': 'Complétez début et fin pour l\'intervalle {0}.',
        'invalid_values': 'Valeurs invalides dans l\'intervalle {0}.',
        'end_must_be_greater': 'La fin doit être supérieure au début dans l\'intervalle {0}.',
        'sweeps_must_be_positive': 'Le nombre de balayages doit être supérieur à zéro (intervalle {0}).',
        'enter_valid_interval': 'Entrez au moins un intervalle valide.',
        'enter_numeric_values': 'Entrez des valeurs numériques valides pour début et fin.',
        'end_greater_than_start': 'La fin doit être supérieure au début.',
        'language_selection_title': 'Sélectionner Langue',
        'select_language': 'Sélectionnez une langue:',
        'could_not_save_language': 'Impossible d\'enregistrer la langue.',
        'tailscale_on': 'TAILSCALE: ON',
        'tailscale_off': 'TAILSCALE: OFF',
        'tailscale_config_title': 'Configuration Tailscale',
        # 'tailscale_username' y 'tailscale_password' eliminados - no se usan
        'tailscale_not_configured': 'Tailscale non configuré',
        'configure_tailscale_first': 'Configurez Tailscale d\'abord en utilisant le bouton d\'engrenage.',
        'tailscale_connecting': 'Connexion à Tailscale...',
        'tailscale_disconnecting': 'Déconnexion de Tailscale...',
        'tailscale_connected': 'Tailscale connecté avec succès.',
        'tailscale_disconnected': 'Tailscale déconnecté avec succès.',
        'tailscale_error': 'Erreur lors de la connexion/déconnexion de Tailscale.',
        'tailscale_not_installed': 'Tailscale n\'est pas installé. Installez-le depuis https://tailscale.com/download',
        'install_tailscale': 'Installer le service',
        'installing_tailscale': 'Installation de Tailscale...',
        'tailscale_install_success': 'Tailscale installé avec succès. Redémarrez l\'application.',
        'tailscale_install_error': 'Erreur lors de l\'installation de Tailscale.',
        'tailscale_installer_not_found': 'Installateur Tailscale introuvable dans le chemin du projet.',
        'tailscale_oauth_info': 'Tailscale utilise l\'authentification OAuth (Google, Microsoft, etc.).\nLorsque vous activez Tailscale, votre navigateur s\'ouvrira pour l\'authentification.',
        'tailscale_installed_info': 'Tailscale est installé. Utilisez le bouton TAILSCALE: ON/OFF pour vous connecter.',
        'tailscale_logged_in_as': 'Connecté en tant que:',
        'tailscale_ip_device': 'IP Tailscale de cet appareil:',
        'tailscale_other_devices': 'Autres appareils connectés:',
        'tailscale_create_account': 'Créer un nouveau compte Tailscale',
        'tailscale_sudo_needed': 'Tailscale nécessite des permissions d\'administrateur.\n\nExécutez une fois dans le terminal:\nsudo tailscale set --operator=$USER\n\nEnsuite, vous pourrez utiliser Tailscale sans sudo.',
        'nvidia_cuda_info': 'Il est recommandé, si vous disposez d\'une carte graphique NVIDIA, d\'avoir les pilotes à jour. De plus, sur Linux, NVIDIA CUDA Toolkit est requis pour le bon fonctionnement des bibliothèques (sudo apt install nvidia-cuda-toolkit).',
        'audio_sensitivity_label': 'Sensibilité brute de détection audio par Tensorflow (avancé):',
        'audio_sensitivity_percent': '%',
        'audio_sensitivity_save_error': 'Impossible d\'enregistrer la sensibilité audio.',
        'audio_sensitivity_range_error': 'La valeur de sensibilité doit être comprise entre 1% et 100%.',
        'audio_sensitivity_number_error': 'La valeur de sensibilité doit être un nombre.',
    },
    'it': {  # Italian
        'yolo_on': 'YOLO: {0} rilev.',
        'yolo_off': 'YOLO: OFF',
        'tinysa_on': 'TinySA: ON',
        'tinysa_off': 'TinySA: OFF',
        'det_audio_on': 'RIL AUDIO: ON',
        'det_audio_off': 'RIL AUDIO: OFF',
        'ip_label': 'IP: {0}',
        'fps_label': 'FPS: {0:.1f}',
        'language_app': 'LINGUA E CONFIG. APP',
        'no_streaming': 'Streaming non rilevato',
        'no_streaming_yolo': 'Impossibile avviare YOLO perché non è disponibile alcuno streaming video.',
        'activate_audio_first': 'Attiva prima la trasmissione audio cliccando sull\'icona del volume.',
        'activate_audio_title': 'Attiva trasmissione audio',
        'no_signal': 'NESSUN SEGNALE',
        'reconnecting': 'Tentativo di riconnessione...',
        'audio_drone_detected': 'DRONE AUDIO RILEVATO: {0}%',
        'no_audio_dron': 'NESSUN DRONE AUDIO: {0}%',
        'rf_drone_detected': 'DRONE RILEVATO DA RF: {0:.3f} GHz ({1}%)',
        'rf_drone_detected_no_freq': 'DRONE RILEVATO DA RF ({0}%)',
        'tinysa_connected': 'TinySA connesso',
        'tinysa_connected_android': 'TinySA connesso ad Android',
        'adb_connected': 'ADB connesso',
        'tinysa_not_configured': 'TinySA non configurato',
        'configure_tinysa_first': 'Configura prima TinySA usando il pulsante ingranaggio.',
        'tinysa_not_detected': 'TinySA non rilevato localmente né sul server Android.\nCollegalo via USB al PC o ad Android e riprova.',
        'change_camera_ip': 'Cambia IP Camera',
        'enter_new_ip': 'Inserisci nuovo IP e porta:\n(attuale: {0})',
        'yolo_options_title': 'Opzioni YOLO',
        'available_models': 'Modelli disponibili',
        'model': 'Modello {0}',
        'description': 'Descrizione:',
        'browse': 'Sfoglia',
        'select_yolo_model': 'Seleziona modello YOLO',
        'yolo_models': 'Modelli YOLO (*.pt)',
        'all_files': 'Tutti i file',
        'page': 'Pagina {0}/{1}',
        'load_model': 'Carica modello',
        'load_and_save_default': 'Carica e salva come predefinito',
        'load_default_config': 'Carica configurazione predefinita',
        'cancel': 'Annulla',
        'model_updated': 'Modello aggiornato correttamente.',
        'error': 'Errore',
        'model_empty': 'Il modello nello slot {0} è vuoto.',
        'file_not_found': 'File non trovato:\n{0}',
        'tinysa_mode_selection': 'TinySA - Selezione modalità',
        'select_mode': 'Seleziona una modalità:',
        'fpv_normal': 'FPV-Normale (2.442 GHz)',
        'fpv_alt': 'FPV-Alt (5.8 GHz)',
        'fpv_mix': 'FPV Mix (2.4 e 5.8 GHz sequenziale)',
        'custom_range': 'Intervallo personalizzato',
        'start_mhz': 'Inizio (MHz):',
        'stop_mhz': 'Fine (MHz):',
        'advanced_range': 'Intervallo personalizzato - Intervallo avanzato',
        'ok': 'OK',
        'advanced_interval_title': 'Intervallo personalizzato - Intervallo avanzato',
        'up_to_5_intervals': 'Fino a 5 intervalli (MHz)',
        'interval': 'Intervallo {0}:',
        'start': 'Inizio',
        'stop': 'Fine',
        'sweeps': '# scansioni',
        'complete_start_end': 'Completa inizio e fine per l\'intervallo {0}.',
        'invalid_values': 'Valori non validi nell\'intervallo {0}.',
        'end_must_be_greater': 'La fine deve essere maggiore dell\'inizio nell\'intervallo {0}.',
        'sweeps_must_be_positive': 'Il numero di scansioni deve essere maggiore di zero (intervallo {0}).',
        'enter_valid_interval': 'Inserisci almeno un intervallo valido.',
        'enter_numeric_values': 'Inserisci valori numerici validi per inizio e fine.',
        'end_greater_than_start': 'La fine deve essere maggiore dell\'inizio.',
        'language_selection_title': 'Seleziona Lingua',
        'select_language': 'Seleziona una lingua:',
        'could_not_save_language': 'Impossibile salvare la lingua.',
        'tailscale_on': 'TAILSCALE: ON',
        'tailscale_off': 'TAILSCALE: OFF',
        'tailscale_config_title': 'Configurazione Tailscale',
        # 'tailscale_username' y 'tailscale_password' eliminados - no se usan
        'tailscale_not_configured': 'Tailscale non configurato',
        'configure_tailscale_first': 'Configura prima Tailscale usando il pulsante ingranaggio.',
        'tailscale_connecting': 'Connessione a Tailscale...',
        'tailscale_disconnecting': 'Disconnessione da Tailscale...',
        'tailscale_connected': 'Tailscale connesso con successo.',
        'tailscale_disconnected': 'Tailscale disconnesso con successo.',
        'tailscale_error': 'Errore durante la connessione/disconnessione di Tailscale.',
        'tailscale_not_installed': 'Tailscale non è installato. Installalo da https://tailscale.com/download',
        'install_tailscale': 'Installa servizio',
        'installing_tailscale': 'Installazione di Tailscale...',
        'tailscale_install_success': 'Tailscale installato con successo. Riavvia l\'applicazione.',
        'tailscale_install_error': 'Errore durante l\'installazione di Tailscale.',
        'tailscale_installer_not_found': 'Installatore Tailscale non trovato nel percorso del progetto.',
        'tailscale_oauth_info': 'Tailscale utilizza l\'autenticazione OAuth (Google, Microsoft, ecc.).\nQuando attivi Tailscale, il tuo browser si aprirà per l\'autenticazione.',
        'tailscale_installed_info': 'Tailscale è installato. Usa il pulsante TAILSCALE: ON/OFF per connetterti.',
        'tailscale_logged_in_as': 'Accesso effettuato come:',
        'tailscale_ip_device': 'IP Tailscale di questo dispositivo:',
        'tailscale_other_devices': 'Altri dispositivi connessi:',
        'tailscale_create_account': 'Crea nuovo account Tailscale',
        'tailscale_sudo_needed': 'Tailscale richiede permessi di amministratore.\n\nEsegui una volta nel terminale:\nsudo tailscale set --operator=$USER\n\nQuindi potrai usare Tailscale senza sudo.',
        'nvidia_cuda_info': 'Si raccomanda, se si dispone di una scheda grafica NVIDIA, di avere i driver aggiornati. Inoltre, su Linux è richiesto NVIDIA CUDA Toolkit per il corretto funzionamento delle librerie (sudo apt install nvidia-cuda-toolkit).',
        'audio_sensitivity_label': 'Sensibilità raw di rilevamento audio tramite Tensorflow (avanzato):',
        'audio_sensitivity_percent': '%',
        'audio_sensitivity_save_error': 'Impossibile salvare la sensibilità audio.',
        'audio_sensitivity_range_error': 'Il valore di sensibilità deve essere compreso tra 1% e 100%.',
        'audio_sensitivity_number_error': 'Il valore di sensibilità deve essere un numero.',
    },
    'pt': {  # Portuguese
        'yolo_on': 'YOLO: {0} det.',
        'yolo_off': 'YOLO: OFF',
        'tinysa_on': 'TinySA: ON',
        'tinysa_off': 'TinySA: OFF',
        'det_audio_on': 'DET ÁUDIO: ON',
        'det_audio_off': 'DET ÁUDIO: OFF',
        'ip_label': 'IP: {0}',
        'fps_label': 'FPS: {0:.1f}',
        'language_app': 'IDIOMA E CONFIG. APP',
        'no_streaming': 'Streaming não detectado',
        'no_streaming_yolo': 'Não é possível iniciar YOLO porque não há streaming de vídeo disponível.',
        'activate_audio_first': 'Ative primeiro a transmissão de áudio clicando no ícone de volume.',
        'activate_audio_title': 'Ativar transmissão de áudio',
        'no_signal': 'SEM SINAL',
        'reconnecting': 'Tentando reconectar...',
        'audio_drone_detected': 'DRONE DE ÁUDIO DETECTADO: {0}%',
        'no_audio_dron': 'NENHUM DRONE DE ÁUDIO: {0}%',
        'rf_drone_detected': 'DRONE DETECTADO POR RF: {0:.3f} GHz ({1}%)',
        'rf_drone_detected_no_freq': 'DRONE DETECTADO POR RF ({0}%)',
        'tinysa_connected': 'TinySA conectado',
        'tinysa_connected_android': 'TinySA conectado ao Android',
        'adb_connected': 'ADB conectado',
        'tinysa_not_configured': 'TinySA não configurado',
        'configure_tinysa_first': 'Configure TinySA primeiro usando o botão de engrenagem.',
        'tinysa_not_detected': 'TinySA não detectado localmente nem no servidor Android.\nConecte-o via USB ao PC ou ao Android e tente novamente.',
        'change_camera_ip': 'Alterar IP da Câmera',
        'enter_new_ip': 'Digite o novo IP e porta:\n(atual: {0})',
        'yolo_options_title': 'Opções YOLO',
        'available_models': 'Modelos Disponíveis',
        'model': 'Modelo {0}',
        'description': 'Descrição:',
        'browse': 'Procurar',
        'select_yolo_model': 'Selecionar Modelo YOLO',
        'yolo_models': 'Modelos YOLO (*.pt)',
        'all_files': 'Todos os Arquivos',
        'page': 'Página {0}/{1}',
        'load_model': 'Carregar Modelo',
        'load_and_save_default': 'Carregar e Salvar como Padrão',
        'load_default_config': 'Carregar Configuração Padrão',
        'cancel': 'Cancelar',
        'model_updated': 'Modelo atualizado com sucesso.',
        'error': 'Erro',
        'model_empty': 'O modelo no slot {0} está vazio.',
        'file_not_found': 'Arquivo não encontrado:\n{0}',
        'tinysa_mode_selection': 'TinySA - Seleção de Modo',
        'select_mode': 'Selecione um modo:',
        'fpv_normal': 'FPV-Normal (2.442 GHz)',
        'fpv_alt': 'FPV-Alt (5.8 GHz)',
        'fpv_mix': 'FPV Mix (2.4 e 5.8 GHz sequencial)',
        'custom_range': 'Intervalo Personalizado',
        'start_mhz': 'Início (MHz):',
        'stop_mhz': 'Fim (MHz):',
        'advanced_range': 'Intervalo Personalizado - Intervalo Avançado',
        'ok': 'OK',
        'advanced_interval_title': 'Intervalo Personalizado - Intervalo Avançado',
        'up_to_5_intervals': 'Até 5 intervalos (MHz)',
        'interval': 'Intervalo {0}:',
        'start': 'Início',
        'stop': 'Fim',
        'sweeps': '# varreduras',
        'complete_start_end': 'Complete início e fim para o intervalo {0}.',
        'invalid_values': 'Valores inválidos no intervalo {0}.',
        'end_must_be_greater': 'O fim deve ser maior que o início no intervalo {0}.',
        'sweeps_must_be_positive': 'O número de varreduras deve ser maior que zero (intervalo {0}).',
        'enter_valid_interval': 'Digite pelo menos um intervalo válido.',
        'enter_numeric_values': 'Digite valores numéricos válidos para início e fim.',
        'end_greater_than_start': 'O fim deve ser maior que o início.',
        'language_selection_title': 'Selecionar Idioma',
        'select_language': 'Selecione um idioma:',
        'could_not_save_language': 'Não foi possível salvar o idioma.',
        'tailscale_on': 'TAILSCALE: ON',
        'tailscale_off': 'TAILSCALE: OFF',
        'tailscale_config_title': 'Configuração Tailscale',
        # 'tailscale_username' y 'tailscale_password' eliminados - no se usan
        'tailscale_not_configured': 'Tailscale não configurado',
        'configure_tailscale_first': 'Configure Tailscale primeiro usando o botão de engrenagem.',
        'tailscale_connecting': 'Conectando ao Tailscale...',
        'tailscale_disconnecting': 'Desconectando do Tailscale...',
        'tailscale_connected': 'Tailscale conectado com sucesso.',
        'tailscale_disconnected': 'Tailscale desconectado com sucesso.',
        'tailscale_error': 'Erro ao conectar/desconectar Tailscale.',
        'tailscale_not_installed': 'Tailscale não está instalado. Instale-o em https://tailscale.com/download',
        'install_tailscale': 'Instalar Serviço',
        'installing_tailscale': 'Instalando Tailscale...',
        'tailscale_install_success': 'Tailscale instalado com sucesso. Reinicie a aplicação.',
        'tailscale_install_error': 'Erro ao instalar Tailscale.',
        'tailscale_installer_not_found': 'Instalador Tailscale não encontrado no caminho do projeto.',
        'tailscale_oauth_info': 'Tailscale usa autenticação OAuth (Google, Microsoft, etc.).\nAo ativar Tailscale, seu navegador será aberto para autenticação.',
        'tailscale_installed_info': 'Tailscale está instalado. Use o botão TAILSCALE: ON/OFF para conectar.',
        'tailscale_logged_in_as': 'Conectado como:',
        'tailscale_ip_device': 'IP Tailscale deste dispositivo:',
        'tailscale_other_devices': 'Outros dispositivos conectados:',
        'tailscale_create_account': 'Criar nova conta Tailscale',
        'tailscale_sudo_needed': 'Tailscale requer permissões de administrador.\n\nExecute uma vez no terminal:\nsudo tailscale set --operator=$USER\n\nDepois poderá usar Tailscale sem sudo.',
        'nvidia_cuda_info': 'Recomenda-se, se você tiver uma placa gráfica NVIDIA, manter os drivers atualizados. Além disso, no Linux, o NVIDIA CUDA Toolkit é necessário para o funcionamento correto das bibliotecas (sudo apt install nvidia-cuda-toolkit).',
        'audio_sensitivity_label': 'Sensibilidade raw de detecção de áudio por Tensorflow (avançado):',
        'audio_sensitivity_percent': '%',
        'audio_sensitivity_save_error': 'Não foi possível salvar a sensibilidade de áudio.',
        'audio_sensitivity_range_error': 'O valor de sensibilidade deve estar entre 1% e 100%.',
        'audio_sensitivity_number_error': 'O valor de sensibilidade deve ser um número.',
    }
}

# Idioma actual (por defecto: español)
current_language = 'es'

def cargar_idioma():
    """Carga el idioma guardado o retorna el por defecto (español)"""
    global current_language
    if os.path.exists(LANGUAGE_CONFIG_FILE):
        try:
            with open(LANGUAGE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                lang = config.get('language', 'es')
                if lang in TRANSLATIONS:
                    current_language = lang
                    return lang
        except Exception as e:
            print(f"Error al cargar idioma: {e}")
    current_language = 'es'
    return 'es'

def guardar_idioma(lang):
    """Guarda el idioma seleccionado"""
    global current_language
    try:
        # Cargar configuración existente para preservar otros valores
        config = {}
        if os.path.exists(LANGUAGE_CONFIG_FILE):
            try:
                with open(LANGUAGE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except:
                pass
        config['language'] = lang
        with open(LANGUAGE_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        current_language = lang
        return True
    except Exception as e:
        print(f"Error al guardar idioma: {e}")
        return False

def cargar_audio_threshold():
    """Carga el umbral de confianza de audio desde la configuración"""
    global AUDIO_CONFIDENCE_THRESHOLD
    if os.path.exists(LANGUAGE_CONFIG_FILE):
        try:
            with open(LANGUAGE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                threshold = config.get('audio_confidence_threshold', 0.15)
                # Validar que esté en rango válido (0.01 a 1.0)
                if 0.01 <= threshold <= 1.0:
                    AUDIO_CONFIDENCE_THRESHOLD = threshold
                else:
                    AUDIO_CONFIDENCE_THRESHOLD = 0.15
        except Exception as e:
            print(f"Error al cargar umbral de audio: {e}")
            AUDIO_CONFIDENCE_THRESHOLD = 0.15
    else:
        AUDIO_CONFIDENCE_THRESHOLD = 0.15

def guardar_audio_threshold(threshold):
    """Guarda el umbral de confianza de audio"""
    global AUDIO_CONFIDENCE_THRESHOLD
    try:
        # Cargar configuración existente
        config = {}
        if os.path.exists(LANGUAGE_CONFIG_FILE):
            try:
                with open(LANGUAGE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except:
                pass
        config['audio_confidence_threshold'] = threshold
        with open(LANGUAGE_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        AUDIO_CONFIDENCE_THRESHOLD = threshold
        return True
    except Exception as e:
        print(f"Error al guardar umbral de audio: {e}")
        return False

def t(key, *args):
    """Obtiene la traducción de una clave. Soporta formato con argumentos."""
    global current_language
    translation = TRANSLATIONS.get(current_language, TRANSLATIONS['es']).get(key, key)
    if args:
        try:
            return translation.format(*args)
        except:
            return translation
    return translation

def show_language_selection_dialog():
    """Muestra el diálogo para seleccionar idioma."""
    root = tk.Tk()
    root.title(t('language_selection_title'))
    root.attributes("-topmost", True)
    root.resizable(False, False)
    
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)
    
    ttk.Label(main_frame, text=t('select_language'), 
              font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 15))
    
    languages = [
        ('es', 'Español'),
        ('en', 'English'),
        ('fr', 'Français'),
        ('it', 'Italiano'),
        ('pt', 'Português')
    ]
    
    selected_lang = tk.StringVar(value=current_language)
    
    # Crear los labels primero (para que las funciones puedan acceder a ellos)
    nvidia_label = ttk.Label(main_frame, text=t('nvidia_cuda_info'), 
                             font=("Arial", 9), 
                             foreground="gray",
                             wraplength=400,
                             justify="left")
    
    # Crear el frame de sensibilidad primero (para que la función pueda acceder al label)
    sensitivity_frame = ttk.Frame(main_frame)
    sensitivity_label = ttk.Label(sensitivity_frame, text=t('audio_sensitivity_label'), 
                                  font=("Arial", 9))
    
    # Función para actualizar el mensaje cuando cambie el idioma
    def update_nvidia_message():
        # Guardar temporalmente el idioma seleccionado
        temp_lang = selected_lang.get()
        # Obtener la traducción del mensaje para ese idioma
        nvidia_text = TRANSLATIONS.get(temp_lang, TRANSLATIONS['es']).get('nvidia_cuda_info', '')
        nvidia_label.config(text=nvidia_text)
        # Actualizar también la etiqueta de sensibilidad
        sensitivity_label.config(text=t('audio_sensitivity_label'))
    
    # Crear los radiobuttons
    for lang_code, lang_name in languages:
        ttk.Radiobutton(main_frame, text=lang_name, variable=selected_lang, 
                       value=lang_code, command=update_nvidia_message).pack(anchor="w", pady=5)
    
    # Separador (se muestra después de los radiobuttons)
    ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=(15, 15))
    
    # Control de sensibilidad de audio (se muestra después del separador)
    sensitivity_frame.pack(fill="x", pady=(0, 15))
    sensitivity_label.pack(anchor="w", pady=(0, 5))
    
    sensitivity_control_frame = ttk.Frame(sensitivity_frame)
    sensitivity_control_frame.pack(fill="x")
    
    # Convertir el umbral actual (0.15) a porcentaje (15)
    current_threshold_percent = int(AUDIO_CONFIDENCE_THRESHOLD * 100)
    sensitivity_var = tk.StringVar(value=str(current_threshold_percent))
    
    sensitivity_spinbox = ttk.Spinbox(sensitivity_control_frame, 
                                       from_=1, 
                                       to=100, 
                                       textvariable=sensitivity_var,
                                       width=10)
    sensitivity_spinbox.pack(side="left", padx=(0, 5))
    
    ttk.Label(sensitivity_control_frame, text=t('audio_sensitivity_percent')).pack(side="left")
    
    # Mensaje sobre NVIDIA CUDA
    ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=(15, 15))
    nvidia_label.pack(anchor="w", pady=(0, 15))
    
    result = {"selected": None}
    
    def on_ok():
        result["selected"] = selected_lang.get()
        # Guardar idioma
        if not guardar_idioma(result["selected"]):
            messagebox.showerror(t('error'), t('could_not_save_language'))
            return
        
        # Guardar umbral de audio
        try:
            threshold_percent = int(sensitivity_var.get())
            if 1 <= threshold_percent <= 100:
                threshold_value = threshold_percent / 100.0
                if not guardar_audio_threshold(threshold_value):
                    messagebox.showerror(t('error'), t('audio_sensitivity_save_error'))
            else:
                messagebox.showerror(t('error'), t('audio_sensitivity_range_error'))
                return
        except ValueError:
            messagebox.showerror(t('error'), t('audio_sensitivity_number_error'))
            return
        
        root.destroy()
    
    def on_cancel():
        result["selected"] = None
        root.destroy()
    
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(15, 0))
    
    ttk.Button(btn_frame, text=t('ok'), command=on_ok, width=12).pack(side="left", padx=5)
    ttk.Button(btn_frame, text=t('cancel'), command=on_cancel, width=12).pack(side="left", padx=5)
    
    # Separador antes del footer
    ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=(15, 15))
    
    # Footer con logo de GitHub y copyright (al final del diálogo)
    footer_frame = ttk.Frame(main_frame)
    footer_frame.pack(fill="x", pady=(0, 0))
    
    # Logo de GitHub
    github_logo_path = os.path.join(BASE_DIR, "ghlogo.png")
    github_logo = None
    github_button = None
    
    if os.path.exists(github_logo_path):
        try:
            from PIL import Image, ImageTk
            img = Image.open(github_logo_path)
            # Redimensionar a un tamaño razonable (16x16 píxeles para icono pequeño)
            img = img.resize((16, 16), Image.Resampling.LANCZOS)
            github_logo = ImageTk.PhotoImage(img)
            
            github_button = tk.Button(footer_frame, 
                                     image=github_logo,
                                     command=lambda: webbrowser.open('https://github.com/zarkentroska/ADAS3-Server'),
                                     cursor="hand2",
                                     relief="flat",
                                     borderwidth=0)
            github_button.pack(side="left", padx=(0, 10))
            # Mantener referencia a la imagen para evitar que se elimine
            github_button.image = github_logo
        except ImportError:
            # Si PIL no está disponible, intentar con PhotoImage de tkinter
            try:
                github_logo = tk.PhotoImage(file=github_logo_path)
                # Redimensionar si es necesario (PhotoImage no tiene resize fácil, usar subimage)
                github_button = tk.Button(footer_frame,
                                        image=github_logo,
                                        command=lambda: webbrowser.open('https://github.com/zarkentroska/ADAS3-Server'),
                                        cursor="hand2",
                                        relief="flat",
                                        borderwidth=0)
                github_button.pack(side="left", padx=(0, 10))
                github_button.image = github_logo
            except Exception as e:
                print(f"No se pudo cargar el logo de GitHub: {e}")
        except Exception as e:
            print(f"Error al cargar el logo de GitHub: {e}")
    
    # Texto de copyright
    copyright_label = ttk.Label(footer_frame, 
                               text="ADAS3 Server v0.5 |  Copyright (C) 2026 GNU GPL 3.0",
                               font=("Arial", 8),
                               foreground="gray")
    copyright_label.pack(side="left")
    
    root.mainloop()
    return result["selected"]

def draw_tailscale_indicator(frame, mouse_pos, click_pos):
    """Dibuja el indicador de Tailscale."""
    x = frame.shape[1] - 40
    y = 110  # Debajo de DET AUDIO (y=80) + 30 píxeles
    
    if tailscale_running:
        color = (0, 255, 0)
        text = t('tailscale_on')
    else:
        color = (0, 0, 255)
        text = t('tailscale_off')
    
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)

def draw_tailscale_settings_icon(frame, mouse_pos, click_pos):
    """Dibuja el icono PNG de ajustes para Tailscale."""
    icon = get_yolo_settings_icon()  # Reutilizamos el mismo icono
    if icon is None:
        return frame, False

    h, w = icon.shape[:2]
    padding = 10
    x2 = frame.shape[1] - 10
    x1 = x2 - w
    y1 = 110 - h // 2 - 5  # Posición al lado de Tailscale (y=110), subido 3 píxeles
    y2 = y1 + h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    roi = frame[y1:y2, x1:x2]
    icon_resized = icon[: y2 - y1, : x2 - x1]

    if icon_resized.shape[2] == 4:
        alpha = icon_resized[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * icon_resized[:, :, c]
    else:
        roi[:] = icon_resized

    mx, my = mouse_pos
    is_hover = x1 <= mx <= x2 and y1 <= my <= y2
    is_clicked = False
    if click_pos:
        cx_click, cy_click = click_pos
        if x1 <= cx_click <= x2 and y1 <= cy_click <= y2:
            is_clicked = True

    if is_hover:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    return frame, is_clicked

def show_tailscale_config_dialog():
    """Muestra el diálogo de configuración de Tailscale."""
    root = tk.Tk()
    root.title(t('tailscale_config_title'))
    root.attributes("-topmost", True)
    root.resizable(False, False)
    
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)
    
    # Información sobre autenticación OAuth
    info_text = t('tailscale_oauth_info')
    info_label = ttk.Label(main_frame, text=info_text, font=("Arial", 9), foreground="gray", wraplength=300, justify="left")
    info_label.pack(anchor="w", pady=(0, 15))
    
    # Botón de instalación (si Tailscale no está instalado y el instalador existe)
    install_btn_frame = None
    if not tailscale_installed():
        installer_exists = False
        if os.name == 'nt':  # Windows
            installer_exists = os.path.exists(TAILSCALE_INSTALLER_WIN)
        else:  # Linux
            installer_exists = os.path.exists(TAILSCALE_INSTALLER_LINUX)
        
        if installer_exists:
            install_btn_frame = ttk.Frame(main_frame)
            install_btn_frame.pack(fill="x", pady=(0, 15))
            
            def on_install():
                if messagebox.askyesno(t('install_tailscale'), t('installing_tailscale')):
                    install_tailscale()
            
            ttk.Button(install_btn_frame, text=t('install_tailscale'), command=on_install, width=25).pack()
        else:
            # Si no está instalado y no hay instalador, mostrar mensaje
            no_installer_label = ttk.Label(main_frame, text=t('tailscale_not_installed'), font=("Arial", 9), foreground="orange", wraplength=300, justify="left")
            no_installer_label.pack(anchor="w", pady=(0, 15))
    else:
        # Si está instalado, mostrar estado
        status_text = t('tailscale_installed_info')
        status_label = ttk.Label(main_frame, text=status_text, font=("Arial", 9), foreground="green", wraplength=300, justify="left")
        status_label.pack(anchor="w", pady=(0, 15))
        
        # Obtener información de Tailscale si está conectado
        username = get_tailscale_username()
        tailscale_ip = get_tailscale_ip()
        
        if username or tailscale_ip:
            # Mostrar información adicional si está conectado
            info_frame = ttk.Frame(main_frame)
            info_frame.pack(anchor="w", pady=(0, 15))
            
            if username:
                logged_in_label = ttk.Label(info_frame, text=f"{t('tailscale_logged_in_as')} {username}", 
                                           font=("Arial", 9), foreground="gray", wraplength=300, justify="left")
                logged_in_label.pack(anchor="w", pady=(0, 5))
            
            if tailscale_ip:
                ip_label = ttk.Label(info_frame, text=f"{t('tailscale_ip_device')} {tailscale_ip}", 
                                    font=("Arial", 9), foreground="darkblue", wraplength=300, justify="left")
                ip_label.pack(anchor="w", pady=(0, 5))
            
            # Obtener y mostrar otros dispositivos conectados (excluyendo la IP actual)
            connected_devices = get_tailscale_connected_devices()
            # Filtrar para excluir la IP del dispositivo actual
            if tailscale_ip and connected_devices:
                connected_devices = [d for d in connected_devices if d['ip'] != tailscale_ip]
            
            if connected_devices:
                other_devices_label = ttk.Label(info_frame, text=t('tailscale_other_devices'), 
                                               font=("Arial", 9), foreground="darkblue", wraplength=300, justify="left")
                other_devices_label.pack(anchor="w", pady=(0, 5))
                
                # Mostrar cada dispositivo
                for device in connected_devices:
                    device_text = f"{device['ip']} ({device['name']})"
                    device_label = ttk.Label(info_frame, text=device_text, 
                                            font=("Arial", 9), foreground="red", wraplength=300, justify="left")
                    device_label.pack(anchor="w", pady=(0, 2))
    
    def on_close():
        root.destroy()
    
    def on_create_account():
        webbrowser.open('https://login.tailscale.com/start')
    
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(15, 0))
    
    ttk.Button(btn_frame, text=t('ok'), command=on_close, width=12).pack(side="left", padx=5)
    ttk.Button(btn_frame, text=t('tailscale_create_account'), command=on_create_account, width=25).pack(side="left", padx=5)
    
    root.mainloop()
    return True

def open_tailscale_options_dialog():
    """Abre la ventana de opciones de Tailscale en un hilo aparte."""
    global tailscale_options_thread
    if tailscale_options_thread and tailscale_options_thread.is_alive():
        return

    def runner():
        global tailscale_options_thread
        try:
            show_tailscale_config_dialog()
        finally:
            tailscale_options_thread = None

    tailscale_options_thread = threading.Thread(target=runner, daemon=True)
    tailscale_options_thread.start()

def draw_language_indicator(frame, mouse_pos, click_pos):
    """Dibuja el indicador de idioma."""
    x = frame.shape[1] - 40
    y = 140  # Debajo de Tailscale (y=110) + 30 píxeles
    
    text = t('language_app')
    color = (255, 255, 255)  # Blanco
    
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)

def solicitar_nueva_ip(ip_actual):
    """Muestra diálogo para cambiar la IP"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    nueva_ip = simpledialog.askstring(
        t('change_camera_ip'),
        t('enter_new_ip', ip_actual),
        initialvalue=ip_actual
    )
    
    root.destroy()
    
    if nueva_ip and nueva_ip.strip():
        return nueva_ip.strip()
    return None

# Cargar IP guardada al iniciar
ip_y_puerto = cargar_ip()
base_url = f"http://{ip_y_puerto}"
video_url = base_url + "/video"
audio_url = base_url + "/audio"
window_name = 'ADAS3 Server'

# Cargar idioma al inicio
cargar_idioma()
# Cargar umbral de audio
cargar_audio_threshold()

# NOTA: No cargamos configuración de Tailscale - Tailscale maneja la autenticación persistentemente
# El usuario se autentica mediante OAuth cuando ejecuta 'tailscale up'
# Verificar estado real de Tailscale al iniciar
verificar_estado_tailscale()

print(f"Iniciando con IP guardada: {base_url}")

# Estados auxiliares Windows / conexión video
video_connection_attempts = []
windows_cursor_fixed = False
windows_cursor_warning = False

# --- CONFIGURACIÓN DE AUDIO ---
CHUNK = 1024
p = pyaudio.PyAudio()
audio_stream = None
stop_audio_thread = False
audio_enabled = False
audio_playback_muted = False  # Mute playback sin afectar detección
audio_thread = None

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': '*/*',
    'Connection': 'keep-alive'
}

# --- CONFIGURACIÓN DETECCIÓN DE AUDIO ---
audio_model = None
audio_mean = None
audio_std = None
audio_detection_enabled = False
audio_detection_thread = None
audio_buffer = queue.Queue(maxsize=20)
audio_detection_result = {"is_drone": False, "confidence": 0.0}
audio_detection_lock = threading.Lock()
# Sistema de alerta persistente
audio_detection_alert_time = None  # Timestamp de la última detección que superó el umbral
audio_detection_max_confidence = 0.0  # Máximo porcentaje alcanzado durante la alerta actual
AUDIO_ALERT_DURATION = 30  # Duración de la alerta en segundos

# Variables para espectrograma de audio (DATOS RAW)
audio_spectrogram_data = None
audio_spectrogram_freqs = None
audio_spectrogram_lock = threading.Lock()

# Variables para el RENDERIZADO del espectrograma
spectrogram_image_ready = None
spectrogram_render_thread = None
spectrogram_render_active = False
spectrogram_image_lock = threading.Lock()

AUDIO_SAMPLE_RATE = 22050
AUDIO_DURATION = 2  # Segundos (entero para evitar errores de slice)
# AUDIO_CONFIDENCE_THRESHOLD se carga desde la configuración, por defecto 0.15
AUDIO_CONFIDENCE_THRESHOLD = 0.15  # Umbral muy bajo para detectar señales débiles  # 70% de confianza para detectar dron
AUDIO_VISUAL_MULTIPLIER = 3  # Multiplicador visual para mostrar porcentajes más altos (hasta 100% máximo)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# --- UTILIDADES AUDIO ---
def prepare_chunk_for_detection(raw_chunk, sample_rate, channels):
    """Convierte un chunk PCM a mono 44.1 kHz para la IA."""
    try:
        audio_array = np.frombuffer(raw_chunk, dtype=np.int16)
        if len(audio_array) == 0:
            return b''
        
        if channels > 1:
            remainder = len(audio_array) % channels
            if remainder:
                audio_array = audio_array[:-remainder]
            if len(audio_array) == 0:
                return b''
            audio_array = audio_array.reshape(-1, channels).mean(axis=1)
        
        if sample_rate != 44100:
            audio_array = audio_array.astype(np.float32) / 32768.0
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=44100)
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = (audio_array * 32767.0).astype(np.int16)
        else:
            audio_array = audio_array.astype(np.int16)
        
        return audio_array.tobytes()
    except Exception as e:
        print(f"[AUDIO] Error preparando chunk: {e}")
        return raw_chunk
# --- CONFIGURACIÓN TINYSA ULTRA+ ---
# Soporta dos modos: serial directo (PC) o HTTP (Android)
tinysa_serial = None
tinysa_running = False
tinysa_thread = None
tinysa_render_thread = None
tinysa_menu_thread = None
tinysa_http_response = None  # Stream HTTP para recibir datos
tinysa_use_http = False  # Indica si usar modo HTTP o serial

tinysa_data_lock = threading.Lock()
tinysa_render_lock = threading.Lock()

# Datos compartidos
tinysa_data_ready = None         # (freqs, levels) actual
tinysa_image_ready = None        # último frame RGBA renderizado

# Detección de drones por RF
rf_drone_detection_result = {"is_drone": False, "confidence": 0.0, "frequency": None}
rf_drone_detection_lock = threading.Lock()
rf_drone_detection_enabled = True
rf_drone_detection_history = []  # Historial de detecciones para persistencia

# Parámetros ajustables de detección RF (con sliders)
rf_peak_threshold = -80.0  # dBm - umbral mínimo para considerar un pico significativo
rf_min_peak_height_db = 15.0  # dB - altura mínima del pico sobre el ruido
rf_min_peak_width_mhz = 10.0  # MHz - ancho mínimo del pico
rf_max_peak_width_mhz = 50.0  # MHz - ancho máximo del pico
rf_sliders_visible = False  # Control de visibilidad de sliders RF
rf_detection_params_lock = threading.Lock()  # Lock para parámetros RF
tinysa_overlay_cache = None

# Configuración actual
current_tinysa_config = None
tinysa_sequence = []
tinysa_sequence_index = 0
TIN_YSA_SWEEPS_PER_RANGE = 5
tinysa_current_label = ""
# ADVANCED_INTERVALS_FILE ya está definido arriba usando CONFIG_DIR (persistente)
last_advanced_intervals = []
tinysa_detected = False
tinysa_last_check = 0.0
TIN_YSA_CHECK_INTERVAL = 5.0
tinysa_last_sequence_payload = None  # Copia del último payload enviado en modo HTTP
TINYSA_HTTP_CONNECT_TIMEOUT = 5.0
TINYSA_HTTP_READ_TIMEOUT = 120.0
TINYSA_STREAM_CHUNK_SIZE = 8192  # 8KB para JSON con 200 puntos (~5KB)
TINYSA_NO_DATA_TIMEOUT = 12.0
TINYSA_POINTS = 200  # Puntos por barrido

TINYSA_PRESETS = {
    "Normal": {"center": 2442000000, "span": 100000000, "points": TINYSA_POINTS},
    "Alt":    {"start": 5725000000, "stop": 5850000000, "points": TINYSA_POINTS}
}


def _preset_to_range(config, label):
    """Convierte un preset en un rango start/stop en Hz."""
    if "center" in config and "span" in config:
        start = int(config["center"] - config["span"] / 2)
        stop = int(config["center"] + config["span"] / 2)
    else:
        start = int(config["start"])
        stop = int(config["stop"])
    return {
        "start": start,
        "stop": stop,
        "points": int(config.get("points", TINYSA_POINTS)),
        "sweeps": TIN_YSA_SWEEPS_PER_RANGE,
        "label": label.replace("–", "-"),
    }


def build_tinysa_sequence(selection, custom_data=None, advanced_ranges=None):
    """Genera la secuencia de barridos según la selección del usuario."""
    sequence = []

    if selection == "preset1":
        sequence.append(_preset_to_range(TINYSA_PRESETS["Normal"], "FPV-Normal 2.442 GHz"))
    elif selection == "preset2":
        sequence.append(_preset_to_range(TINYSA_PRESETS["Alt"], "FPV-Alt 5.8 GHz"))
    elif selection == "mix":
        sequence.append(_preset_to_range(TINYSA_PRESETS["Normal"], "FPV Mix - 2.442 GHz"))
        sequence.append(_preset_to_range(TINYSA_PRESETS["Alt"], "FPV Mix - 5.8 GHz"))
    elif selection == "custom" and custom_data:
        start_mhz, stop_mhz = custom_data
        start_hz = int(start_mhz * 1e6)
        stop_hz = int(stop_mhz * 1e6)
        sequence.append({
            "start": start_hz,
            "stop": stop_hz,
            "points": TINYSA_POINTS,
            "sweeps": TIN_YSA_SWEEPS_PER_RANGE,
            "label": f"Custom {start_mhz:.3f}-{stop_mhz:.3f} MHz",
        })
    elif selection == "advanced" and advanced_ranges:
        # Guardar la última configuración para reutilizarla
        last_advanced_intervals.clear()
        for idx, (start_mhz, stop_mhz, sweeps_val) in enumerate(advanced_ranges, start=1):
            last_advanced_intervals.append(
                {"start_mhz": start_mhz, "stop_mhz": stop_mhz, "sweeps": sweeps_val}
            )
            sequence.append({
                "start": int(start_mhz * 1e6),
                "stop": int(stop_mhz * 1e6),
                "points": TINYSA_POINTS,
                "sweeps": max(1, int(sweeps_val)),
                "label": f"Avanzado #{idx}: {start_mhz:.3f}-{stop_mhz:.3f} MHz",
            })

    return sequence


def show_advanced_interval_dialog(parent):
    """Diálogo para configurar hasta 5 intervalos personalizados."""
    dialog = tk.Toplevel(parent)
    dialog.title(t('advanced_interval_title'))
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.lift()
    dialog.focus_force()

    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text=t('up_to_5_intervals')).grid(row=0, column=0, columnspan=5, pady=(0, 10))

    entries = []
    for i in range(5):
        start_var = tk.StringVar()
        stop_var = tk.StringVar()
        sweeps_var = tk.StringVar(value=str(TIN_YSA_SWEEPS_PER_RANGE))
        ttk.Label(frame, text=t('interval', i + 1)).grid(row=i + 1, column=0, sticky="w", padx=(0, 10))
        ttk.Label(frame, text=t('start')).grid(row=i + 1, column=1, sticky="e")
        start_entry = ttk.Entry(frame, textvariable=start_var, width=10)
        start_entry.grid(row=i + 1, column=2, padx=5, pady=2)
        ttk.Label(frame, text=t('stop')).grid(row=i + 1, column=3, sticky="e")
        stop_entry = ttk.Entry(frame, textvariable=stop_var, width=10)
        stop_entry.grid(row=i + 1, column=4, padx=5, pady=2)
        ttk.Label(frame, text=t('sweeps')).grid(row=i + 1, column=5, sticky="e")
        sweeps_entry = ttk.Entry(frame, textvariable=sweeps_var, width=6)
        sweeps_entry.grid(row=i + 1, column=6, padx=5, pady=2)
        entries.append((start_var, stop_var, sweeps_var))

    # Prefill con la última configuración guardada
    for (start_var, stop_var, sweeps_var), saved in zip(entries, last_advanced_intervals):
        start_var.set(str(saved["start_mhz"]))
        stop_var.set(str(saved["stop_mhz"]))
        sweeps_var.set(str(saved["sweeps"]))

    result = {"ranges": None}

    def on_ok():
        ranges = []
        for idx, (start_var, stop_var, sweeps_var) in enumerate(entries, start=1):
            start_text = start_var.get().strip()
            stop_text = stop_var.get().strip()
            if not start_text and not stop_text:
                continue
            if not start_text or not stop_text:
                messagebox.showerror(t('error'), t('complete_start_end', idx))
                return
            try:
                start_val = float(start_text)
                stop_val = float(stop_text)
                sweeps_val = int(sweeps_var.get().strip())
            except ValueError:
                messagebox.showerror(t('error'), t('invalid_values', idx))
                return
            if stop_val <= start_val:
                messagebox.showerror(t('error'), t('end_must_be_greater', idx))
                return
            if sweeps_val <= 0:
                messagebox.showerror(t('error'), t('sweeps_must_be_positive', idx))
                return
            ranges.append((start_val, stop_val, sweeps_val))

        if not ranges:
            messagebox.showerror(t('error'), t('enter_valid_interval'))
            return

        result["ranges"] = ranges
        try:
            with open(ADVANCED_INTERVALS_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"start_mhz": r[0], "stop_mhz": r[1], "sweeps": r[2]}
                        for r in ranges
                    ],
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"[TinySA] No se pudo guardar configuración avanzada: {e}")
        dialog.destroy()

    def on_cancel():
        result["ranges"] = None
        dialog.destroy()

    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=7, column=0, columnspan=5, pady=(15, 0))
    ttk.Button(btn_frame, text=t('ok'), command=on_ok, width=12).pack(side="left", padx=5)
    ttk.Button(btn_frame, text=t('cancel'), command=on_cancel, width=12).pack(side="left", padx=5)

    dialog.wait_window()
    return result["ranges"]


def show_tinysa_menu():
    """Muestra el selector gráfico para TinySA."""
    root = tk.Tk()
    root.title(t('tinysa_mode_selection'))
    root.attributes("-topmost", True)
    root.resizable(False, False)

    # Precargar intervalos avanzados guardados
    global last_advanced_intervals
    if os.path.exists(ADVANCED_INTERVALS_FILE):
        try:
            with open(ADVANCED_INTERVALS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    last_advanced_intervals = data
        except Exception as e:
            print(f"[TinySA] No se pudo leer configuración avanzada previa: {e}")

    main_frame = ttk.Frame(root, padding=15)
    main_frame.pack(fill="both", expand=True)

    selection_var = tk.StringVar(value="preset1")
    custom_start = tk.StringVar()
    custom_stop = tk.StringVar()

    ttk.Label(main_frame, text=t('select_mode'), font=("Arial", 11, "bold")).pack(anchor="w")

    options_frame = ttk.Frame(main_frame)
    options_frame.pack(fill="x", pady=10)

    ttk.Radiobutton(
        options_frame, text=t('fpv_normal'), variable=selection_var, value="preset1"
    ).pack(anchor="w", pady=2)
    ttk.Radiobutton(
        options_frame, text=t('fpv_alt'), variable=selection_var, value="preset2"
    ).pack(anchor="w", pady=2)
    ttk.Radiobutton(
        options_frame,
        text=t('fpv_mix'),
        variable=selection_var,
        value="mix",
    ).pack(anchor="w", pady=2)

    custom_radio = ttk.Radiobutton(
        options_frame,
        text=t('custom_range'),
        variable=selection_var,
        value="custom",
    )
    custom_radio.pack(anchor="w", pady=2)

    custom_frame = ttk.Frame(options_frame)
    custom_frame.pack(anchor="w", padx=20, pady=(0, 5))
    ttk.Label(custom_frame, text=t('start_mhz')).grid(row=0, column=0, sticky="w")
    custom_start_entry = ttk.Entry(custom_frame, textvariable=custom_start, width=10, state="disabled")
    custom_start_entry.grid(row=0, column=1, padx=5)
    ttk.Label(custom_frame, text=t('stop_mhz')).grid(row=0, column=2, sticky="w")
    custom_stop_entry = ttk.Entry(custom_frame, textvariable=custom_stop, width=10, state="disabled")
    custom_stop_entry.grid(row=0, column=3, padx=5)

    ttk.Radiobutton(
        options_frame,
        text=t('advanced_range'),
        variable=selection_var,
        value="advanced",
    ).pack(anchor="w", pady=2)

    result = {"selection": None, "custom": None, "advanced": None}

    def update_custom_state(*_):
        state = "normal" if selection_var.get() == "custom" else "disabled"
        custom_start_entry.configure(state=state)
        custom_stop_entry.configure(state=state)

    selection_var.trace_add("write", update_custom_state)

    def finish_and_close():
        root.quit()

    def on_ok():
        sel = selection_var.get()
        if sel == "custom":
            try:
                start_val = float(custom_start.get())
                stop_val = float(custom_stop.get())
            except ValueError:
                messagebox.showerror(t('error'), t('enter_numeric_values'))
                return
            if stop_val <= start_val:
                messagebox.showerror(t('error'), t('end_greater_than_start'))
                return
            result["selection"] = sel
            result["custom"] = (start_val, stop_val)
            finish_and_close()
        elif sel == "advanced":
            try:
                root.attributes("-disabled", True)
            except Exception:
                pass
            ranges = show_advanced_interval_dialog(root)
            try:
                root.attributes("-disabled", False)
            except Exception:
                pass
            if ranges is None:
                root.focus_set()
                return
            result["selection"] = sel
            result["advanced"] = ranges
            finish_and_close()
        else:
            result["selection"] = sel
            finish_and_close()

    def on_cancel():
        result["selection"] = None
        finish_and_close()

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(10, 0))
    ttk.Button(btn_frame, text=t('ok'), command=on_ok, width=12).pack(side="left", padx=5)
    ttk.Button(btn_frame, text=t('cancel'), command=on_cancel, width=12).pack(side="left", padx=5)

    def on_close():
        result["selection"] = None
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    if root.winfo_exists():
        root.destroy()
    return result

# --- FUNCIONES TINYSA HARDWARE ---

def find_tinysa_port():
    """Busca el puerto COM del TinySA Ultra"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.vid == 0x0483 and port.pid == 0x5740:
            return port.device
    return None


def read_exactly(ser, n):
    """
    Lee exactamente n bytes del puerto serie, reintentando si es necesario.
    Devuelve menos bytes sólo si se agota el timeout.
    """
    data = b''
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if not chunk:
            break  # Timeout
        data += chunk
    return data


def send_tinysa_command(command_json):
    """
    Envía un comando JSON al servidor Android para controlar TinySA.
    """
    try:
        command_url = base_url + "/tinysa/command"
        response = requests.post(
            command_url,
            json=command_json,
            headers={'Content-Type': 'application/json'},
            timeout=10  # Aumentado timeout
        )
        if response.status_code == 200:
            print(f"[TINYSA] Comando enviado: {command_json.get('action', 'unknown')}")
            return True
        else:
            print(f"[TINYSA] Error enviando comando: HTTP {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"[TINYSA] Timeout enviando comando (puede que el servidor no esté respondiendo)")
        return False
    except Exception as e:
        print(f"[TINYSA] Error enviando comando: {e}")
        return False

def tinysa_hardware_worker_serial():
    """
    Hilo de hardware para TinySA Ultra en modo scanraw con secuencias de barrido (serial directo).
    Recorre los rangos seleccionados ejecutando varios barridos en cada uno.
    """
    global tinysa_data_ready, tinysa_running, tinysa_serial
    global tinysa_sequence_index, tinysa_current_label

    print("[TINYSA] Hardware Worker Serial iniciado")

    if not tinysa_sequence:
        print("[TINYSA] Sin secuencia activa, saliendo worker.")
        return

    try:
        if tinysa_serial is None or not tinysa_serial.is_open:
            print("[TINYSA] Puerto serie no disponible en hardware worker.")
            return

        tinysa_serial.reset_input_buffer()
        tinysa_serial.write(b"abort\r")
        try:
            tinysa_serial.read_until(b"ch> ")
        except Exception:
            pass
        time.sleep(0.05)

        while tinysa_running and tinysa_sequence:
            config = tinysa_sequence[tinysa_sequence_index]
            start = int(config["start"])
            stop = int(config["stop"])
            points = int(config.get("points", TINYSA_POINTS))
            sweeps_target = max(1, int(config.get("sweeps", TIN_YSA_SWEEPS_PER_RANGE)))
            tinysa_current_label = config.get("label", "")

            cmd = f"scanraw {start} {stop} {points}\r".encode()
            sweeps_done = 0

            while tinysa_running and sweeps_done < sweeps_target:
                tinysa_serial.write(cmd)

                try:
                    raw_block = tinysa_serial.read_until(b"}")
                except Exception as e:
                    print(f"[TINYSA] Error leyendo bloque scanraw: {e}")
                    time.sleep(0.05)
                    continue

                if not raw_block:
                    time.sleep(0.02)
                    continue

                start_idx = raw_block.find(b"{")
                end_idx = raw_block.rfind(b"}")

                if start_idx == -1 or end_idx <= start_idx + 1:
                    time.sleep(0.02)
                    continue
                data_bytes = raw_block[start_idx + 1 : end_idx]

                if len(data_bytes) < 30:
                    time.sleep(0.02)
                    continue

                n_points = len(data_bytes) // 3
                if len(data_bytes) % 3 != 0:
                    data_bytes = data_bytes[: n_points * 3]

                if n_points != points:
                    print(
                        f"[TINYSA] Aviso: dispositivo devolvió {n_points} puntos "
                        f"en lugar de {points}."
                    )

                try:
                    values = [v[0] for v in struct.iter_unpack("<xH", data_bytes)]
                    if len(values) != n_points:
                        time.sleep(0.02)
                        continue

                    levels = (np.asarray(values, dtype=np.float32) / 32.0) - 174.0
                    freqs_dynamic = np.linspace(start, stop, n_points, dtype=np.float32)

                    with tinysa_data_lock:
                        tinysa_data_ready = (freqs_dynamic, levels)

                except Exception as e:
                    print(f"[TINYSA] Error parseando datos scanraw: {e}")
                    time.sleep(0.02)
                    continue

                try:
                    tinysa_serial.read_until(b"ch> ")
                except Exception:
                    pass

                sweeps_done += 1

            tinysa_sequence_index = (tinysa_sequence_index + 1) % len(tinysa_sequence)

    except Exception as e:
        print(f"[TINYSA] Error crítico en hardware worker: {e}")
    finally:
        tinysa_current_label = ""

    print("[TINYSA] Hardware Worker Serial finalizado")

def tinysa_hardware_worker():
    """
    Hilo de hardware para TinySA Ultra usando HTTP stream desde Android.
    Lee datos JSON del endpoint /tinysa/data en tiempo real.
    """
    global tinysa_data_ready, tinysa_running, tinysa_http_response
    global tinysa_current_label
    global tinysa_last_sequence_payload, tinysa_use_http

    print("[TINYSA] Hardware Worker HTTP iniciado")
    print(f"[TINYSA] HTTP timeouts -> connect: {TINYSA_HTTP_CONNECT_TIMEOUT}s, read: {TINYSA_HTTP_READ_TIMEOUT}s")

    def restart_remote_scanning(reason):
        """
        Reenvía la secuencia y el comando start al servidor Android si se corta el stream.
        """
        if not tinysa_running or not tinysa_use_http:
            return
        if not tinysa_last_sequence_payload:
            return
        print(f"[TINYSA] Reiniciando barrido remoto ({reason})")
        # No detener si ya está detenido; los comandos fallidos solo mostrarán el log
        send_tinysa_command({"action": "stop"})
        send_tinysa_command({"action": "set_sequence", "sequence": tinysa_last_sequence_payload})
        send_tinysa_command({"action": "start"})
        return time.time()

    buffers_since_data = 0
    BUFFER_LIMIT_BEFORE_TIMEOUT = 24  # 24 * 0.5s = 12s aprox
    
    try:
        data_url = base_url + "/tinysa/data"
        print(f"[TINYSA] Conectando a {data_url}...")
        
        # Conectar al stream de datos
        tinysa_http_response = requests.get(
            data_url,
            stream=True,
            headers=headers,
            timeout=(TINYSA_HTTP_CONNECT_TIMEOUT, TINYSA_HTTP_READ_TIMEOUT)
        )
        
        if tinysa_http_response.status_code != 200:
            print(f"[TINYSA] Error conectando: HTTP {tinysa_http_response.status_code}")
            tinysa_running = False
            tinysa_use_http = False
            return
        
        print("[TINYSA] Conectado al stream de datos")
        
        tinysa_http_response.raw.decode_content = True
        
        # Buffer para acumular datos JSON (pueden llegar fragmentados)
        buffer = ""
        last_data_time = time.time()
        
        while tinysa_running:
            try:
                chunk = tinysa_http_response.raw.read(TINYSA_STREAM_CHUNK_SIZE)
                
                if not chunk:
                    if tinysa_running:
                        print("[TINYSA] Stream cerrado, reintentando...")
                        time.sleep(1)
                        # Reintentar conexión
                        try:
                            tinysa_http_response.close()
                        except:
                            pass
                        tinysa_http_response = requests.get(
                            data_url,
                            stream=True,
                            headers=headers,
                            timeout=(TINYSA_HTTP_CONNECT_TIMEOUT, TINYSA_HTTP_READ_TIMEOUT)
                        )
                        if tinysa_http_response.status_code != 200:
                            break
                        buffer = ""
                        ts = restart_remote_scanning("reconexión tras stream cerrado")
                        if ts:
                            last_data_time = ts
                        else:
                            last_data_time = time.time()
                        continue  # Volver a intentar leer
                    else:
                        break
                
                # Decodificar y agregar al buffer
                buffer += chunk.decode('utf-8', errors='ignore')
                
                # Procesar líneas JSON completas
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Solo procesar si parece un JSON completo
                        if not line.startswith('{') or not line.endswith('}'):
                            # Línea incompleta, ignorar
                            continue
                        
                        # Parsear JSON
                        data = json.loads(line)
                        freqs_array = data.get('freqs', [])
                        levels_array = data.get('levels', [])
                        
                        if len(freqs_array) > 0 and len(levels_array) > 0:
                            freqs = np.array(freqs_array, dtype=np.float32)
                            levels = np.array(levels_array, dtype=np.float32)
                            
                            print(f"[HTTP {time.time():.2f}] Datos: {len(freqs)} pts")
                            
                            with tinysa_data_lock:
                                tinysa_data_ready = (freqs, levels)
                            last_data_time = time.time()
                    except json.JSONDecodeError as e:
                        # JSON incompleto o corrupto, ignorar silenciosamente
                        continue
                    except Exception as e:
                        print(f"[TINYSA] Error procesando datos: {e}")
                        continue
                        
            except requests.exceptions.RequestException as e:
                if tinysa_running:
                    print(f"[TINYSA] Error en stream: {e}, reintentando...")
                    time.sleep(1)
                    try:
                        tinysa_http_response.close()
                    except:
                        pass
                    try:
                        tinysa_http_response = requests.get(
                            data_url,
                            stream=True,
                            headers=headers,
                            timeout=(TINYSA_HTTP_CONNECT_TIMEOUT, TINYSA_HTTP_READ_TIMEOUT)
                        )
                        if tinysa_http_response.status_code != 200:
                            break
                        buffer = ""
                        ts = restart_remote_scanning("reconexión tras error de red")
                        if ts:
                            last_data_time = ts
                        else:
                            last_data_time = time.time()
                    except:
                        break
                else:
                    break
            except Exception as e:
                print(f"[TINYSA] Error inesperado: {e}")
                time.sleep(0.1)

    except Exception as e:
        print(f"[TINYSA] Error crítico en hardware worker HTTP: {e}")
        tinysa_running = False
        tinysa_use_http = False
    finally:
        tinysa_current_label = ""
        try:
            if tinysa_http_response:
                tinysa_http_response.close()
        except:
            pass
        tinysa_http_response = None
    
    print("[TINYSA] Hardware Worker HTTP finalizado")

    print("[TINYSA] Hardware Worker HTTP finalizado")

def detect_drone_rf(freqs, levels):
    """
    Detecta señales de drones en el espectro RF basándose en patrones característicos.
    
    Características de señales de drones FPV:
    - Picos significativos en bandas 2.4 GHz o 5.8 GHz
    - Ancho de banda típico: 20-40 MHz
    - Potencia por encima del ruido de fondo (> -80 dBm típicamente)
    - Forma de pico característica (montañitas)
    
    Args:
        freqs: Array de frecuencias en Hz
        levels: Array de niveles en dBm
        
    Returns:
        dict con is_drone (bool), confidence (float 0-1), frequency (Hz o None)
    """
    global rf_drone_detection_history
    
    if len(freqs) == 0 or len(levels) == 0 or len(freqs) != len(levels):
        return {"is_drone": False, "confidence": 0.0, "frequency": None}
    
    # Convertir a numpy arrays si no lo son
    freqs = np.array(freqs)
    levels = np.array(levels)
    
    # Bandas típicas de drones (en Hz)
    DRONE_BANDS = [
        (2400000000, 2500000000),  # 2.4 GHz (WiFi/FPV)
        (5725000000, 5875000000),  # 5.8 GHz (FPV)
    ]
    
    # Parámetros de detección (usar variables globales ajustables)
    global rf_peak_threshold, rf_min_peak_height_db, rf_min_peak_width_mhz, rf_max_peak_width_mhz
    
    with rf_detection_params_lock:
        PEAK_THRESHOLD = rf_peak_threshold
        MIN_PEAK_HEIGHT_DB = rf_min_peak_height_db
        MIN_PEAK_WIDTH_MHZ = rf_min_peak_width_mhz
        MAX_PEAK_WIDTH_MHZ = rf_max_peak_width_mhz
    
    NOISE_FLOOR = -100  # dBm - nivel de ruido de fondo típico
    
    # Filtrar datos válidos
    valid_mask = np.isfinite(levels) & (levels > -150) & (levels < 0)
    if np.sum(valid_mask) < 10:  # Necesitamos al menos 10 puntos válidos
        return {"is_drone": False, "confidence": 0.0, "frequency": None}
    
    freqs_valid = freqs[valid_mask]
    levels_valid = levels[valid_mask]
    
    # Calcular ruido de fondo (percentil 10 para evitar picos)
    noise_level = np.percentile(levels_valid, 10)
    
    # Buscar picos significativos usando detección de máximos locales (sin scipy)
    # Buscar picos que estén al menos MIN_PEAK_HEIGHT_DB por encima del ruido
    peak_threshold_relative = noise_level + MIN_PEAK_HEIGHT_DB
    
    # Detección simple de picos: un punto es un pico si es mayor que sus vecinos
    # y está por encima del umbral
    min_distance = max(1, len(levels_valid) // 50)  # Distancia mínima entre picos
    peaks = []
    
    for i in range(min_distance, len(levels_valid) - min_distance):
        if levels_valid[i] < peak_threshold_relative:
            continue
        
        # Verificar que sea un máximo local
        is_peak = True
        for j in range(max(0, i - min_distance), min(len(levels_valid), i + min_distance + 1)):
            if j != i and levels_valid[j] >= levels_valid[i]:
                is_peak = False
                break
        
        if is_peak:
            peaks.append(i)
    
    if len(peaks) == 0:
        return {"is_drone": False, "confidence": 0.0, "frequency": None}
    
    peaks = np.array(peaks)
    
    # Analizar cada pico
    best_peak = None
    best_confidence = 0.0
    best_frequency = None
    
    for peak_idx in peaks:
        peak_freq = freqs_valid[peak_idx]
        peak_level = levels_valid[peak_idx]
        
        # Verificar que esté en una banda de drones
        in_drone_band = False
        for band_start, band_stop in DRONE_BANDS:
            if band_start <= peak_freq <= band_stop:
                in_drone_band = True
                break
        
        if not in_drone_band:
            continue
        
        # Calcular ancho de banda del pico (FWHM - Full Width at Half Maximum)
        half_max = peak_level - (peak_level - noise_level) / 2
        
        # Encontrar puntos donde la señal cruza half_max
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and levels_valid[left_idx] > half_max:
            left_idx -= 1
        while right_idx < len(levels_valid) - 1 and levels_valid[right_idx] > half_max:
            right_idx += 1
        
        # Calcular ancho de banda en MHz
        if left_idx < right_idx:
            bandwidth_hz = freqs_valid[right_idx] - freqs_valid[left_idx]
            bandwidth_mhz = bandwidth_hz / 1e6
        else:
            bandwidth_mhz = 0
        
        # Verificar criterios de detección
        if (peak_level > PEAK_THRESHOLD and 
            MIN_PEAK_WIDTH_MHZ <= bandwidth_mhz <= MAX_PEAK_WIDTH_MHZ):
            
            # Calcular confianza basada en:
            # 1. Altura del pico sobre el ruido
            # 2. Ancho de banda (óptimo alrededor de 20-30 MHz)
            # 3. Potencia absoluta
            
            height_above_noise = peak_level - noise_level
            height_confidence = min(1.0, height_above_noise / 40.0)  # Normalizar a 40 dB
            
            # Ancho de banda óptimo alrededor de 20-25 MHz
            optimal_bw = 22.5
            bw_diff = abs(bandwidth_mhz - optimal_bw)
            bw_confidence = max(0.0, 1.0 - (bw_diff / 20.0))
            
            # Potencia absoluta (picos más fuertes = más confianza)
            power_confidence = min(1.0, (peak_level - PEAK_THRESHOLD) / 30.0)
            
            # Confianza combinada
            confidence = (height_confidence * 0.4 + bw_confidence * 0.3 + power_confidence * 0.3)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_peak = peak_idx
                best_frequency = peak_freq
    
    # Persistencia temporal: requerir detecciones consistentes
    current_time = time.time()
    rf_drone_detection_history = [
        (t, freq, conf) for t, freq, conf in rf_drone_detection_history
        if current_time - t < 2.0  # Mantener últimos 2 segundos
    ]
    
    if best_confidence > 0.5:  # Umbral de confianza
        rf_drone_detection_history.append((current_time, best_frequency, best_confidence))
        
        # Requerir al menos 2 detecciones en los últimos 2 segundos
        if len(rf_drone_detection_history) >= 2:
            avg_confidence = np.mean([conf for _, _, conf in rf_drone_detection_history])
            avg_frequency = np.mean([freq for _, freq, _ in rf_drone_detection_history])
            
            return {
                "is_drone": True,
                "confidence": min(1.0, avg_confidence),
                "frequency": avg_frequency
            }
    
    return {"is_drone": False, "confidence": 0.0, "frequency": None}

def tinysa_render_worker():
    """
    Hilo que dibuja el gráfico TinySA con Matplotlib (Agg) y produce un frame RGBA
    listo para superponer en OpenCV.

    - Figura y ejes con fondo negro opaco.
    - Se crea UNA sola figura y UNA sola línea.
    - Sólo se actualiza la Y de la línea y se redibuja el canvas.
    - Se mantiene siempre el último frame válido.
    """
    global tinysa_image_ready
    print("[TINYSA] Render Worker iniciado")

    if current_tinysa_config is None:
        print("[TINYSA] Sin configuración activa, saliendo render worker.")
        return

    # --- Crear figura estática ---
    fig = Figure(figsize=(5, 2.5), facecolor="black")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # Fondo y estética
    ax.set_facecolor("black")
    ax.grid(True, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("dBm", color="white", fontsize=8)
    ax.set_xlabel("MHz", color="white", fontsize=8)
    ax.tick_params(axis="x", colors="white", labelsize=7)
    ax.tick_params(axis="y", colors="white", labelsize=7)

    # --- Ejes iniciales a partir de la config ---
    if "center" in current_tinysa_config:
        start = int(current_tinysa_config["center"] - current_tinysa_config["span"] / 2)
        stop = int(current_tinysa_config["center"] + current_tinysa_config["span"] / 2)
    else:
        start = int(current_tinysa_config["start"])
        stop = int(current_tinysa_config["stop"])

    points = int(current_tinysa_config["points"])
    freqs_init = np.linspace(start, stop, points, dtype=np.float32)

    ax.set_xlim(freqs_init[0] / 1e6, freqs_init[-1] / 1e6)
    ax.set_ylim(-125, -10)

    modo = "2.4 GHz" if freqs_init[0] < 3e9 else "5.8 GHz"
    ax.set_title(
        f"TinySA Ultra - {modo}", color="#00FF00", fontsize=9, fontweight="bold"
    )

    # Línea inicial (todo a -110 dBm)
    line, = ax.plot(
        freqs_init / 1e6,
        np.full(points, -110.0, dtype=np.float32),
        color="#FFFF00",
        linewidth=1.5,
    )

    fig.tight_layout()

    last_levels_hash = None
    
    while tinysa_running:
        # Obtener datos
        with tinysa_data_lock:
            data = tinysa_data_ready

        if data is None:
            time.sleep(0.01)
            continue

        freqs, levels = data
        actual_points = len(freqs)

        if actual_points == 0 or len(levels) != actual_points:
            time.sleep(0.005)
            continue

        # Detectar cambios por contenido, no por referencia
        current_hash = hash(levels.tobytes())
        if current_hash == last_levels_hash:
            time.sleep(0.002)  # Polling más rápido
            continue
        last_levels_hash = current_hash
        
        # Detectar drones por RF si está habilitado
        if rf_drone_detection_enabled:
            try:
                detection = detect_drone_rf(freqs, levels)
                with rf_drone_detection_lock:
                    rf_drone_detection_result.update(detection)
                if detection["is_drone"]:
                    freq_mhz = detection["frequency"] / 1e6 if detection["frequency"] else 0
                    print(f"[RF DRONE] DETECTADO: {freq_mhz:.3f} MHz, confianza: {detection['confidence']:.2f}")
            except Exception as e:
                print(f"[RF DRONE] Error en detección: {e}")

        render_start = time.time()
        
        try:
            # Actualizar X e Y de la línea
            line.set_xdata(freqs / 1e9)
            line.set_ydata(levels)
            
            # Ajustar límites del eje X dinámicamente
            if len(freqs) > 0:
                ax.set_xlim(freqs[0] / 1e9, freqs[-1] / 1e9)

            # Renderizar a buffer RGBA
            canvas.draw()
            buf = canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8)
            img = img.reshape(canvas.get_width_height()[::-1] + (4,))

            # Publicar imagen para overlay
            with tinysa_render_lock:
                tinysa_image_ready = img
            
            render_time = (time.time() - render_start) * 1000
            print(f"[RENDER {time.time():.2f}] {actual_points} pts en {render_time:.0f}ms")

        except Exception as e:
            print(f"[TINYSA] Error en render: {e}")
            time.sleep(0.05)

    print("[TINYSA] Render Worker finalizado")

def start_tinysa_with_sequence(sequence):
    """Inicia TinySA con una secuencia ya configurada."""
    global tinysa_running, tinysa_serial, current_tinysa_config
    global tinysa_thread, tinysa_render_thread, tinysa_data_ready, tinysa_image_ready
    global tinysa_sequence, tinysa_sequence_index, tinysa_current_label
    global tinysa_detected, tinysa_http_response, tinysa_use_http
    global tinysa_last_sequence_payload
    
    if not sequence:
        print("No hay secuencia configurada para TinySA.")
        return False
    
    tinysa_sequence = sequence
    tinysa_sequence_index = 0
    current_tinysa_config = tinysa_sequence[0]
    tinysa_current_label = current_tinysa_config.get("label", "")
    
    # Decidir modo: primero intentar serial directo, luego HTTP
    port = find_tinysa_port()
    tinysa_detected = port is not None
    
    try:
        if port:
            # Modo serial directo (TinySA conectado al PC)
            try:
                print(f"Conectando a TinySA en {port} (modo serial directo)...")
                tinysa_serial = serial.Serial(port, 921600, timeout=8.0)

                tinysa_serial.flushInput()
                tinysa_serial.write(b"abort\r")
                tinysa_serial.read_until(b"ch> ")

                tinysa_running = True
                tinysa_use_http = False

                with tinysa_data_lock:
                    tinysa_data_ready = None

                tinysa_thread = threading.Thread(
                    target=tinysa_hardware_worker_serial, daemon=True
                )
                tinysa_thread.start()

                tinysa_render_thread = threading.Thread(
                    target=tinysa_render_worker, daemon=True
                )
                tinysa_render_thread.start()

                print("TinySA Activado (modo serial directo)")
                tinysa_detected = True
                return True

            except Exception as e:
                print(f"Error al conectar TinySA por serial: {e}")
                if tinysa_serial:
                    try:
                        tinysa_serial.close()
                    except:
                        pass
                    tinysa_serial = None
                tinysa_running = False
                return False
        else:
            # Modo HTTP (TinySA conectado al Android)
            print(f"[TINYSA] TinySA no detectado localmente, intentando modo HTTP...")
            tinysa_use_http = True
            
            try:
                # Convertir secuencia al formato JSON esperado por el servidor
                sequence_json = []
                for config in sequence:
                    sequence_json.append({
                        "start": int(config["start"]),
                        "stop": int(config["stop"]),
                        "points": int(config.get("points", TINYSA_POINTS)),
                        "sweeps": int(config.get("sweeps", TIN_YSA_SWEEPS_PER_RANGE)),
                        "label": config.get("label", "")
                    })
                
                # Guardar copia profunda para poder rearmar la secuencia si el stream se corta
                try:
                    tinysa_last_sequence_payload = json.loads(json.dumps(sequence_json))
                except Exception:
                    tinysa_last_sequence_payload = sequence_json[:]
                
                # Enviar comando set_sequence
                command = {
                    "action": "set_sequence",
                    "sequence": sequence_json
                }
                
                if not send_tinysa_command(command):
                    print("[TINYSA] Error configurando secuencia en servidor")
                    def show_warning():
                        root = Tk()
                        root.withdraw()
                        root.attributes("-topmost", True)
                        messagebox.showwarning(
                            t('tinysa_not_configured'),
                            t('tinysa_not_detected')
                        )
                        root.destroy()
                    threading.Thread(target=show_warning, daemon=True).start()
                    return False
                
                # Iniciar scanning
                if not send_tinysa_command({"action": "start"}):
                    print("[TINYSA] Error iniciando scanning en servidor")
                    return False

                # Verificar que TinySA esté realmente conectado en el servidor Android
                try:
                    status_url = base_url + "/tinysa/status"
                    response = requests.get(status_url, timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        if not data.get("connected", False):
                            print("[TINYSA] TinySA no está conectado en el servidor Android")
                            tinysa_running = False
                            tinysa_use_http = False
                            def show_warning():
                                root = Tk()
                                root.withdraw()
                                root.attributes("-topmost", True)
                                messagebox.showwarning(
                                    t('tinysa_not_configured'),
                                    t('tinysa_not_detected')
                                )
                                root.destroy()
                            threading.Thread(target=show_warning, daemon=True).start()
                            return False
                    else:
                        print("[TINYSA] Error verificando estado en servidor Android")
                        tinysa_running = False
                        tinysa_use_http = False
                        return False
                except Exception as e:
                    print(f"[TINYSA] Error verificando estado: {e}")
                    tinysa_running = False
                    tinysa_use_http = False
                    return False

                # Solo establecer tinysa_running = True después de verificar que está conectado
                tinysa_running = True

                with tinysa_data_lock:
                    tinysa_data_ready = None

                # Iniciar thread para recibir datos HTTP
                tinysa_thread = threading.Thread(
                    target=tinysa_hardware_worker, daemon=True
                )
                tinysa_thread.start()

                # Iniciar thread de renderizado
                tinysa_render_thread = threading.Thread(
                    target=tinysa_render_worker, daemon=True
                )
                tinysa_render_thread.start()

                print("TinySA Activado (modo HTTP)")
                tinysa_detected = True
                tinysa_use_http = True
                return True

            except Exception as e:
                print(f"Error al conectar TinySA por HTTP: {e}")
                tinysa_running = False
                tinysa_use_http = False
                return False
    except Exception as e:
        print(f"Error general al iniciar TinySA: {e}")
        return False

def toggle_tinysa():
    """
    Activa/Desactiva el TinySA usando la configuración seleccionada.
    Si no hay configuración, muestra un mensaje.
    """
    global tinysa_running, tinysa_serial, tinysa_sequence
    global tinysa_thread, tinysa_render_thread, tinysa_data_ready, tinysa_image_ready
    global tinysa_sequence_index, tinysa_current_label
    global tinysa_detected, tinysa_http_response, tinysa_use_http

    if tinysa_running:
        # Apagar
        tinysa_running = False
        
        if tinysa_use_http:
            # Enviar comando stop al servidor Android
            send_tinysa_command({"action": "stop"})
            # Cerrar conexión HTTP
            try:
                if tinysa_http_response:
                    tinysa_http_response.close()
            except:
                pass
            tinysa_http_response = None
        else:
            # Modo serial directo
            if tinysa_serial:
                try:
                    tinysa_serial.write(b"abort\r")
                    tinysa_serial.close()
                except Exception:
                    pass
                tinysa_serial = None

        # Limpiar buffers compartidos
        with tinysa_data_lock:
            tinysa_data_ready = None
        with tinysa_render_lock:
            tinysa_image_ready = None

        tinysa_sequence_index = 0
        tinysa_current_label = ""
        tinysa_use_http = False
        print("TinySA Desactivado")
        return

    # Intentar activar con la secuencia actual si existe
    if tinysa_sequence and len(tinysa_sequence) > 0:
        start_tinysa_with_sequence(tinysa_sequence)
    else:
        # No hay configuración, mostrar mensaje
        def show_message():
            root = Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showinfo(
                t('tinysa_not_configured'),
                t('configure_tinysa_first')
            )
            root.destroy()
        threading.Thread(target=show_message, daemon=True).start()

def open_tinysa_options_dialog():
    """Abre la ventana de opciones TinySA en un hilo aparte."""
    global tinysa_menu_thread
    if tinysa_menu_thread and tinysa_menu_thread.is_alive():
        return

    def runner():
        global tinysa_menu_thread, tinysa_sequence, tinysa_sequence_index
        global current_tinysa_config, tinysa_current_label
        
        try:
            selection_data = show_tinysa_menu()
            selection = selection_data.get("selection")
            if not selection:
                tinysa_menu_thread = None
                return

            sequence = build_tinysa_sequence(
                selection,
                custom_data=selection_data.get("custom"),
                advanced_ranges=selection_data.get("advanced"),
            )

            if not sequence:
                print("Selección TinySA inválida.")
                tinysa_menu_thread = None
                return

            # Guardar la secuencia para uso futuro
            tinysa_sequence = sequence
            tinysa_sequence_index = 0
            current_tinysa_config = tinysa_sequence[0]
            tinysa_current_label = current_tinysa_config.get("label", "")
            
            # Si TinySA está corriendo, reiniciarlo con la nueva configuración
            if tinysa_running:
                # Apagar primero
                old_running = True
                toggle_tinysa()
                # Activar con nueva configuración
                if old_running:
                    start_tinysa_with_sequence(tinysa_sequence)
        finally:
            tinysa_menu_thread = None

    tinysa_menu_thread = threading.Thread(target=runner, daemon=True)
    tinysa_menu_thread.start()

def overlay_tinysa_graph(frame):
    """
    Dibuja el gráfico del TinySA directamente con OpenCV, transparente sobre el vídeo,
    incluyendo cuadrícula y etiquetas de ejes.
    """

    if not tinysa_running:
        return frame

    # 1. Obtener datos actuales del TinySA
    with tinysa_data_lock:
        data = tinysa_data_ready

    if data is None:
        return frame

    freqs, levels = data
    if freqs is None or levels is None or len(freqs) == 0 or len(levels) == 0:
        return frame

    global tinysa_current_label

    try:
        h, w = frame.shape[:2]

        # Tamaño del panel RF reducido
        panel_w = int(w * 0.27)
        panel_h = int(h * 0.18)

        if panel_w <= 10 or panel_h <= 10:
            return frame

        # Esquina inferior derecha
        x0 = w - panel_w - 10
        y0 = h - panel_h - 10
        x1 = x0 + panel_w
        y1 = y0 + panel_h

        if x0 < 0 or y0 < 0:
            return frame

        # ROI del vídeo donde se superpone el gráfico
        roi = frame[y0:y1, x0:x1]

        # Imagen negra donde dibujamos sólo el gráfico y la cuadrícula
        graph = np.zeros_like(roi)

        # 2. Parámetros de escala
        points = len(levels)

        # Rango de dBm del eje Y (ajústalo a tu gusto)
        db_min = -125.0
        db_max = -10.0

        # Clampear niveles
        lv = np.clip(levels, db_min, db_max)

        # Normalizar e invertir eje Y (dBm altos arriba)
        norm = (lv - db_min) / (db_max - db_min)  # 0..1
        ys = (1.0 - norm) * (panel_h - 1)

        xs = np.linspace(0, panel_w - 1, points)

        pts = np.vstack([xs, ys]).T.astype(np.int32)

        # 3. Dibujar fondo gris oscuro
        graph[:] = (40, 40, 40)

        # 4. Dibujar cuadrícula (ejemplo: 5 divisiones Y, 6 X)
        grid_color = (40, 40, 40)
        n_y = 5
        n_x = 6

        for i in range(1, n_y):
            gy = int(round(i * panel_h / n_y))
            cv2.line(graph, (0, gy), (panel_w - 1, gy), grid_color, 1)

        for i in range(1, n_x):
            gx = int(round(i * panel_w / n_x))
            cv2.line(graph, (gx, 0), (gx, panel_h - 1), grid_color, 1)

        # 5. Ejes y etiquetas de dBm (eje Y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_color = (200, 200, 200)
        thickness = 1

        for i in range(n_y + 1):
            frac = i / n_y
            gy = int(round(frac * (panel_h - 1)))
            db_val = db_max - frac * (db_max - db_min)
            text = f"{int(db_val)}"

            # Línea de referencia gruesa en el borde izquierdo
            cv2.line(graph, (0, gy), (5, gy), (80, 80, 80), 1)

            # Texto a la izquierda (dentro del panel)
            cv2.putText(
                graph,
                text,
                (8, max(10, gy - 2)),
                font,
                font_scale,
                font_color,
                thickness,
                cv2.LINE_AA,
            )

        # 6. Etiquetas de frecuencia aproximadas (eje X)
        f_start_mhz = freqs[0] / 1e6
        f_stop_mhz = freqs[-1] / 1e6

        font_scale_x = 0.4
        for i in range(n_x + 1):
            frac = i / n_x
            gx = int(round(frac * (panel_w - 1)))
            f_val_mhz = f_start_mhz + frac * (f_stop_mhz - f_start_mhz)
            if f_val_mhz >= 1000.0:
                text = f"{f_val_mhz / 1000:.2f}"
                text_offset = 13
            else:
                text = f"{int(round(f_val_mhz))}"
                text_offset = 9

            cv2.line(graph, (gx, panel_h - 8), (gx, panel_h - 1), (80, 80, 80), 1)
            cv2.putText(
                graph,
                text,
                (gx - text_offset, panel_h - 4),
                font,
                font_scale_x,
                font_color,
                thickness,
                cv2.LINE_AA,
            )

        # 7. Título
        ghz_start = freqs[0] / 1e9
        ghz_stop = freqs[-1] / 1e9
        title = f"TinySA Ultra - {ghz_start:.2f}-{ghz_stop:.2f} GHz"
        dynamic_label = tinysa_current_label or f"{freqs[0]/1e6:.2f}-{freqs[-1]/1e6:.2f} MHz"
        cv2.putText(
            graph,
            title,
            (8, 14),
            font,
            0.32,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            graph,
            f"Rango actual: {dynamic_label}",
            (8, 28),
            font,
            0.3,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        # 8. Dibujar la traza en amarillo
        cv2.polylines(graph, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

        # 9. Mezclar ROI original con el gráfico usando alpha fijo (transparente)
        alpha = 0.45  # 75% gráfico, 25% vídeo
        cv2.addWeighted(graph, alpha, roi, 1.0 - alpha, 0.0, roi)

    except Exception:
        # Si algo falla, no tocamos el frame
        pass

    return frame

# --- FUNCIONES MODELO AUDIO (EXISTENTES) ---
def cargar_modelo_audio():
    """Carga el modelo de detección de audio y estadísticas de normalización"""
    global audio_model, audio_mean, audio_std
    try:
        if not os.path.exists(AUDIO_MODEL_PATH):
            print(f"Error: No se encuentra el modelo '{AUDIO_MODEL_PATH}'")
            return False
        
        if os.path.exists(AUDIO_MEAN_PATH) and os.path.exists(AUDIO_STD_PATH):
            audio_mean = np.load(AUDIO_MEAN_PATH)
            audio_std = np.load(AUDIO_STD_PATH)
            print(f"Estadísticas cargadas - Mean: {audio_mean:.4f}, Std: {audio_std:.4f}")
        else:
            print("ERROR: No se encontraron archivos de normalización")
            return False
        
        print("Cargando modelo de detección de audio...")
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        print("Modelo de audio cargado correctamente")
        return True
    except Exception as e:
        print(f"Error al cargar modelo de audio: {e}")
        return False

def extract_features_realtime(audio_chunk):
    """Extrae features de un chunk de audio en tiempo real"""
    global audio_mean, audio_std, audio_spectrogram_data, audio_spectrogram_freqs
    
    try:
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        
        # Log temporal: ver valores raw del audio
        raw_min, raw_max = np.min(audio_data), np.max(audio_data)
        raw_mean = np.mean(np.abs(audio_data))
        
        audio_data = audio_data / 32768.0
        
        # Log temporal: ver valores normalizados
        norm_min, norm_max = np.min(audio_data), np.max(audio_data)
        norm_mean = np.mean(np.abs(audio_data))
        
        # Aplicar ganancia adaptativa para aumentar la señal (solo para detección, no afecta playback)
        # Si el audio es muy bajo, aplicar más ganancia
        mean_abs_level = np.mean(np.abs(audio_data))
        
        if mean_abs_level < 0.005:  # Audio muy bajo (< 0.5% del rango)
            AUDIO_GAIN = 40.0  # Ganancia muy alta para señales muy débiles
        elif mean_abs_level < 0.01:  # Audio bajo (< 1% del rango)
            AUDIO_GAIN = 30.0
        elif mean_abs_level < 0.02:  # Audio moderado (< 2% del rango)
            AUDIO_GAIN = 20.0
        else:  # Audio normal
            AUDIO_GAIN = 10.0
        
        audio_data = audio_data * AUDIO_GAIN
        
        # Limitar a [-1, 1] para evitar clipping
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        if len(audio_data) > 0:
            audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=AUDIO_SAMPLE_RATE)
        
        required_length = AUDIO_SAMPLE_RATE * AUDIO_DURATION
        if len(audio_data) < required_length:
            audio_data = np.pad(audio_data, (0, required_length - len(audio_data)))
        else:
            audio_data = audio_data[:required_length]
        
        # Calcular espectrograma mel
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=AUDIO_SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Log temporal: ver valores del mel spectrograma antes de normalizar
        mel_min, mel_max = np.min(mel_spec_db), np.max(mel_spec_db)
        mel_mean = np.mean(mel_spec_db)
        
        # Guardar datos para el espectrograma visual
        with audio_spectrogram_lock:
            freqs_mel = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=AUDIO_SAMPLE_RATE/2)
            audio_spectrogram_freqs = freqs_mel
            audio_spectrogram_data = mel_spec_db
        
        # Normalizar para el modelo (EXACTAMENTE como en el código antiguo)
        if audio_mean is not None and audio_std is not None:
            mel_spec_db = (mel_spec_db - audio_mean) / (audio_std + 1e-8)
            
            # Log temporal: ver valores después de normalizar
            norm_mel_min, norm_mel_max = np.min(mel_spec_db), np.max(mel_spec_db)
            norm_mel_mean = np.mean(mel_spec_db)
            
            # Log cada 10 llamadas para no saturar
            if not hasattr(extract_features_realtime, '_call_count'):
                extract_features_realtime._call_count = 0
            extract_features_realtime._call_count += 1
            
            if extract_features_realtime._call_count % 10 == 0:
                # Calcular valores después del resample
                gain_min, gain_max = np.min(audio_data), np.max(audio_data)
                gain_mean = np.mean(np.abs(audio_data))
                print(f"[DEBUG AUDIO] Raw: min={raw_min:.0f}, max={raw_max:.0f}, mean_abs={raw_mean:.0f} | "
                      f"Norm: min={norm_min:.4f}, max={norm_max:.4f}, mean_abs={norm_mean:.4f} (level={mean_abs_level:.5f}) | "
                      f"Gain {AUDIO_GAIN:.1f}x: min={gain_min:.4f}, max={gain_max:.4f}, mean_abs={gain_mean:.4f} | "
                      f"Mel dB: min={mel_min:.2f}, max={mel_max:.2f}, mean={mel_mean:.2f} | "
                      f"Mel norm: min={norm_mel_min:.2f}, max={norm_mel_max:.2f}, mean={norm_mel_mean:.2f}")
        else:
            return None
        
        if mel_spec_db.shape[1] < 87:
            pad_width = 87 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)))
        else:
            mel_spec_db = mel_spec_db[:, :87]
        
        # Retornar solo features (sin nivel, ya no lo necesitamos)
        return mel_spec_db
        
    except Exception as e:
        print(f"[FEATURES] Error: {e}")
        return None

def audio_detection_worker():
    """Worker thread para detección de audio"""
    global audio_detection_result, audio_detection_alert_time, audio_detection_max_confidence
    
    accumulated_audio = b''
    required_bytes = int(44100 * AUDIO_DURATION * 2)
    
    print(f"[AUDIO] Worker iniciado")
    
    while audio_detection_enabled:
        try:
            chunk = audio_buffer.get(timeout=1)
            accumulated_audio += chunk
            
            if len(accumulated_audio) >= required_bytes:
                try:
                    features = extract_features_realtime(accumulated_audio[:required_bytes])
                    
                    if features is not None and audio_model is not None:
                        features = features[..., np.newaxis]
                        features = np.expand_dims(features, axis=0)
                        
                        prediction = audio_model.predict(features, verbose=0)[0][0]
                        
                        # Aplicar multiplicador visual (limitado a 100%)
                        visual_confidence = min(1.0, prediction * AUDIO_VISUAL_MULTIPLIER)
                        
                        # Debug: siempre mostrar la primera predicción para verificar que funciona
                        if not hasattr(audio_detection_worker, '_first_prediction_shown'):
                            print(f"[AUDIO] Primera predicción: {visual_confidence*100:.1f}% (raw: {prediction*100:.1f}%)")
                            audio_detection_worker._first_prediction_shown = True
                        
                        # Si supera el umbral, activar alerta y guardar timestamp
                        if prediction >= AUDIO_CONFIDENCE_THRESHOLD:
                            current_time = time.time()
                            if audio_detection_alert_time is None:
                                # Nueva detección - guardar timestamp y máximo
                                audio_detection_alert_time = current_time
                                audio_detection_max_confidence = prediction
                                alert_time_str = time.strftime("%H:%M:%S", time.localtime(current_time))
                                visual_pct = min(100, int(visual_confidence * 100))
                                print(f"[AUDIO] ⚠ DRON DETECTADO A LAS {alert_time_str} - {visual_pct}% (raw: {prediction*100:.1f}%)")
                            else:
                                # Ya hay una alerta activa - actualizar máximo si es mayor
                                if prediction > audio_detection_max_confidence:
                                    audio_detection_max_confidence = prediction
                                # Actualizar timestamp si es una nueva detección muy fuerte
                                if prediction > 0.5:  # Si es muy alta (>50%), actualizar timestamp
                                    audio_detection_alert_time = current_time
                                    alert_time_str = time.strftime("%H:%M:%S", time.localtime(current_time))
                                    visual_pct = min(100, int(visual_confidence * 100))
                                    print(f"[AUDIO] ⚠ NUEVA DETECCIÓN A LAS {alert_time_str} - {visual_pct}% (raw: {prediction*100:.1f}%)")
                        
                        # Verificar si la alerta sigue activa (dentro de los 30 segundos)
                        is_drone = False
                        if audio_detection_alert_time is not None:
                            elapsed = time.time() - audio_detection_alert_time
                            if elapsed < AUDIO_ALERT_DURATION:
                                is_drone = True
                            else:
                                # Alerta expirada - resetear máximo
                                audio_detection_alert_time = None
                                audio_detection_max_confidence = 0.0
                        
                        # Guardar confianza visual para mostrar
                        with audio_detection_lock:
                            audio_detection_result = {
                                "is_drone": is_drone,
                                "confidence": float(visual_confidence)  # Confianza visual (multiplicada)
                            }
                        
                        # Mostrar predicción siempre
                        status = "⚠ ALERTA ACTIVA" if is_drone else ""
                        visual_pct = min(100, int(visual_confidence * 100))
                        print(f"[AUDIO] Predicción: {visual_pct}% (raw: {prediction*100:.1f}%) | Drone: {is_drone} {status}")
                        
                except Exception as e:
                    print(f"[AUDIO] Error: {e}")
                
                overlap_bytes = int(44100 * 0.5 * 2)
                accumulated_audio = accumulated_audio[-overlap_bytes:]
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[AUDIO] Error crítico: {e}")
    
    print("[AUDIO] Worker finalizado")

# --- NUEVO: WORKER PARA RENDERIZAR ESPECTROGRAMA ---
def spectrogram_render_worker():
    """
    Thread dedicado a generar la imagen del espectrograma con Matplotlib.
    Esto evita que el bucle principal de video se congele.
    """
    global spectrogram_image_ready
    
    print("[RENDER] Worker iniciado")
    
    while spectrogram_render_active:
        try:
            # 1. Obtener datos RAW
            with audio_spectrogram_lock:
                if audio_spectrogram_data is None or audio_spectrogram_freqs is None:
                    time.sleep(0.1)
                    continue
                data = audio_spectrogram_data.copy()
                freqs = audio_spectrogram_freqs.copy()
            
            # 2. Obtener estado drone
            with audio_detection_lock:
                is_drone = audio_detection_result["is_drone"]
            
            # 3. Configurar colores
            if is_drone:
                blink = int(time.time() * 3) % 2 == 0
                line_color = '#FF0000' if blink else '#AA0000'
                cmap_name = 'hot'
            else:
                line_color = '#00FFFF'
                cmap_name = 'viridis'
            
            # 4. Crear figura Matplotlib (Heavy lifting)
            # Usamos Figure directamente para evitar problemas de thread-safety con plt.
            fig = Figure(figsize=(5, 1.8), facecolor='none')
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            
            ax.imshow(data, aspect='auto', origin='lower', 
                      cmap=cmap_name,
                      extent=[0, data.shape[1], freqs[0]/1000, freqs[-1]/1000],
                      alpha=0.8)
            
            ax.set_facecolor('none')
            ax.set_xlabel('Tiempo', color='white', fontsize=7)
            ax.set_ylabel('Freq (kHz)', color='white', fontsize=7)
            ax.tick_params(colors='white', labelsize=6)
            ax.set_title('Audio Spectrogram', color=line_color, fontsize=8, fontweight='bold')
            
            fig.tight_layout()
            
            # 5. Renderizar a buffer
            canvas.draw()
            buf = canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8)
            img = img.reshape(canvas.get_width_height()[::-1] + (4,))
            
            # 6. Guardar resultado para el main thread
            with spectrogram_image_lock:
                spectrogram_image_ready = img
            
            # Limitar FPS del renderizado (ahorra mucha CPU)
            # 15 FPS para el gráfico es más que suficiente para el ojo humano
            time.sleep(0.06) 
            
        except Exception as e:
            print(f"[RENDER] Error: {e}")
            time.sleep(0.5)
            
    print("[RENDER] Worker finalizado")

def toggle_audio_detection():
    """Activa/desactiva la detección de audio y el renderizado"""
    global audio_detection_enabled, audio_detection_thread
    global spectrogram_render_active, spectrogram_render_thread
    
    if not audio_detection_enabled:
        if audio_model is None:
            if not cargar_modelo_audio():
                return
        
        # Si el audio no está activo, iniciarlo automáticamente (con playback muteado)
        if not audio_enabled:
            global audio_playback_muted
            audio_playback_muted = True  # Mutear playback automáticamente
            start_audio()
            print("[AUDIO] Stream iniciado automáticamente (playback muteado)")
        
        # Iniciar thread de detección (IA)
        audio_detection_enabled = True
        audio_detection_thread = threading.Thread(target=audio_detection_worker, daemon=True)
        audio_detection_thread.start()
        
        # Iniciar thread de renderizado (Gráfico) - NUEVO
        spectrogram_render_active = True
        spectrogram_render_thread = threading.Thread(target=spectrogram_render_worker, daemon=True)
        spectrogram_render_thread.start()
        
        print("Detección y Renderizado activados")
    else:
        # Apagar detección
        audio_detection_enabled = False
        
        # Apagar renderizado
        spectrogram_render_active = False
        
        if audio_detection_thread:
            audio_detection_thread.join(timeout=2)
        
        if spectrogram_render_thread:
            spectrogram_render_thread.join(timeout=2)
        
        # Limpiar datos
        with audio_spectrogram_lock:
            global audio_spectrogram_data, audio_spectrogram_freqs
            audio_spectrogram_data = None
            audio_spectrogram_freqs = None
            
        with spectrogram_image_lock:
            global spectrogram_image_ready
            spectrogram_image_ready = None
        
        print("Detección desactivada")

# --- CONFIGURACIÓN YOLO CON THREADING ---
yolo_model = None
yolo_enabled = False
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
YOLO_SCALE = 0.5  # Procesar al 50% de resolución

# Threading YOLO
yolo_frame_queue = queue.Queue(maxsize=2)
yolo_result_queue = queue.Queue(maxsize=2)
yolo_worker_thread = None
yolo_worker_running = False
yolo_result_lock = threading.Lock()
ultimo_resultado_yolo = {"frame": None, "detecciones": 0, "boxes_data": []}
yolo_conf_threshold = CONFIDENCE_THRESHOLD
yolo_iou_threshold = IOU_THRESHOLD
yolo_threshold_lock = threading.Lock()
yolo_reload_requested = False
yolo_settings_icon = None
mute_icon = None
vol_icon = None
yolo_slider_active = None
rf_slider_active = None


def apply_yolo_model(new_path, save_default=False, selected_slot=None):
    """Configura el modelo YOLO a usar y marca recarga si estaba activo."""
    global yolo_model_path, yolo_default_slot, yolo_model, yolo_model_slots, yolo_reload_requested

    if not new_path:
        print(f"[YOLO] Ruta de modelo inválida: {new_path}")
        return False
    
    # Normalizar la ruta del modelo
    normalized_path = normalize_model_path(new_path)
    
    if not normalized_path or not os.path.exists(normalized_path):
        print(f"[YOLO] Ruta de modelo inválida o archivo no encontrado: {new_path}")
        return False

    yolo_model_path = normalized_path

    if save_default and selected_slot is not None:
        yolo_default_slot = selected_slot

    save_yolo_models_config()

    if yolo_enabled:
        yolo_reload_requested = True
    else:
        yolo_model = None

    print(f"[YOLO] Modelo activo: {yolo_model_path}")
    return True
def cargar_modelo_yolo():
    """Carga el modelo YOLO"""
    global yolo_model
    try:
        if not os.path.exists(yolo_model_path):
            print(f"Error: No se encuentra el modelo '{yolo_model_path}'")
            return False
        
        print("Cargando modelo YOLO...")
        yolo_model = YOLO(yolo_model_path)
        print(f"Modelo YOLO cargado - Dispositivo: {yolo_model.device}")
        return True
    except Exception as e:
        print(f"Error al cargar modelo YOLO: {e}")
        return False

def yolo_inference_worker():
    """Thread worker dedicado para inferencia YOLO"""
    global ultimo_resultado_yolo
    
    print("[YOLO] Worker thread iniciado")
    
    while yolo_worker_running:
        try:
            # Obtener frame de la cola (con timeout para poder salir limpiamente)
            frame_original, original_shape = yolo_frame_queue.get(timeout=0.1)
            
            if yolo_model is None:
                continue
            
            # Redimensionar para procesamiento más rápido
            small_frame = cv2.resize(frame_original, 
                                    (int(original_shape[1] * YOLO_SCALE), 
                                     int(original_shape[0] * YOLO_SCALE)))
            
            with yolo_threshold_lock:
                conf_thr = yolo_conf_threshold
                iou_thr = yolo_iou_threshold

            # Inferencia YOLO
            results = yolo_model(
                small_frame,
                verbose=False,
                conf=conf_thr,
                iou=iou_thr
            )
            
            boxes_data = []
            detecciones = 0
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Coordenadas escaladas al tamaño original
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1 = int(x1 / YOLO_SCALE)
                    y1 = int(y1 / YOLO_SCALE)
                    x2 = int(x2 / YOLO_SCALE)
                    y2 = int(y2 / YOLO_SCALE)
                    
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = yolo_model.names[cls] if cls < len(yolo_model.names) else f"Class {cls}"
                    boxes_data.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'conf': conf, 'class_name': class_name
                    })
                    detecciones += 1
            
            # Guardar resultado
            with yolo_result_lock:
                ultimo_resultado_yolo = {
                    "frame": frame_original,
                    "detecciones": detecciones,
                    "boxes_data": boxes_data
                }
            
            # Marcar tarea completada
            yolo_frame_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[YOLO] Error en worker: {e}")
    
    print("[YOLO] Worker thread finalizado")

def start_yolo_worker():
    """Inicia el thread worker de YOLO"""
    global yolo_worker_thread, yolo_worker_running
    
    if yolo_worker_thread is not None and yolo_worker_thread.is_alive():
        return
    
    yolo_worker_running = True
    yolo_worker_thread = threading.Thread(target=yolo_inference_worker, daemon=True)
    yolo_worker_thread.start()
    print("[YOLO] Thread worker iniciado")

def stop_yolo_worker():
    """Detiene el thread worker de YOLO"""
    global yolo_worker_running, yolo_worker_thread
    
    if yolo_worker_thread is None:
        return
    
    yolo_worker_running = False
    
    if yolo_worker_thread.is_alive():
        yolo_worker_thread.join(timeout=2)
    
    # Limpiar colas
    while not yolo_frame_queue.empty():
        try:
            yolo_frame_queue.get_nowait()
        except queue.Empty:
            break
    
    print("[YOLO] Thread worker detenido")

def toggle_yolo():
    """Activa o desactiva YOLO"""
    global yolo_enabled, yolo_model
    
    print(f"[DEBUG] toggle_yolo llamado. Estado actual: {yolo_enabled}")

    if not yolo_enabled:
        if yolo_model is None:
            print("[DEBUG] Cargando modelo YOLO...")
            if not cargar_modelo_yolo():
                print("[ERROR] Fallo al cargar modelo YOLO")
                return
        
        print("[DEBUG] Iniciando worker YOLO...")
        start_yolo_worker()
        yolo_enabled = True
        print("YOLO activado")
    else:
        yolo_enabled = False
        stop_yolo_worker()
        
        # Limpiar último resultado
        with yolo_result_lock:
            global ultimo_resultado_yolo
            ultimo_resultado_yolo = {"frame": None, "detecciones": 0, "boxes_data": []}
        
        print("YOLO desactivado")

def enviar_frame_a_yolo(frame):
    """Envía frame a YOLO solo si no está ocupado"""
    if not yolo_enabled:
        return
    
    try:
        # Enviar sin bloquear - si la cola está llena, se salta el frame
        yolo_frame_queue.put_nowait((frame.copy(), frame.shape))
    except queue.Full:
        pass  # YOLO ocupado, saltar este frame

def obtener_resultado_yolo():
    """Obtiene el último resultado de YOLO disponible"""
    with yolo_result_lock:
        return ultimo_resultado_yolo.copy()

def dibujar_detecciones_yolo(frame, boxes_data):
    """Dibuja las detecciones YOLO en el frame"""
    for box in boxes_data:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        conf = box['conf']
        class_name = box['class_name']
        
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{class_name}: {conf:.2f}"
        
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    
    return frame

# --- CLASE DE CAPTURA DE VIDEO ROBUSTA ---
class ThreadedVideoCapture:
    """
    Captura de video optimizada con reconexión segura.
    """
    
    def __init__(self, src):
        self.src = src
        self.successful_init = False
        self.stopped = False
        self.frame = None
        self.ret = False
        self.frame_id = 0
        self.init_time = time.time() # Hora de inicio del objeto
        self.last_frame_time = time.time()
        self.lock = threading.Lock()
        
        # Inicialización del recurso de video
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(src)
            
        # Configuración básica
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except:
            pass
            
        # VERIFICACIÓN CRÍTICA INICIAL
        if not self.cap.isOpened():
            print("[VIDEO] Error: No se pudo abrir el stream.")
            return
            
        self.successful_init = True
        
        # Iniciar thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        print("[VIDEO] Captura Low-Latency iniciada")
    
    def is_valid(self):
        """Retorna True si la captura se inicializó correctamente"""
        return self.successful_init
    
    def _update(self):
        """Loop de lectura"""
        while not self.stopped:
            if self.cap.isOpened():
                grabbed = self.cap.grab()
                if grabbed:
                    self.ret, frame = self.cap.retrieve()
                    if self.ret and frame is not None:
                        with self.lock:
                            self.frame = frame
                            self.frame_id += 1
                            self.last_frame_time = time.time()
                else:
                    time.sleep(0.005)
            else:
                time.sleep(0.1)
    
    def read(self):
        """Devuelve frame con lógica inteligente de espera"""
        if not self.successful_init:
            return False, None, -1

        with self.lock:
            # SI TENEMOS FRAME (Caso normal)
            if self.frame is not None:
                # Watchdog estricto: Si hace más de 3s que no hay nada nuevo -> Muerte
                if time.time() - self.last_frame_time > 3.0:
                      return False, None, -1
                return True, self.frame.copy(), self.frame_id

            # SI NO TENEMOS FRAME (Caso inicial / Cargando)
            # Si estamos en los primeros 5 segundos de vida, decimos "Todo OK, espera"
            if time.time() - self.init_time < 5.0:
                return True, None, -1 # Retorna True (vivo) pero None (vacío)
            
            # Si pasaron 5 segundos y sigue sin haber frame -> Muerte
            return False, None, -1
    
    def release(self):
        self.stopped = True
        if self.successful_init and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.cap.isOpened():
            self.cap.release()
        print("[VIDEO] Captura liberada")


def schedule_video_connection(target_url, force=False):
    """
    Lanza un intento de conexión en background para no bloquear la UI.
    """
    global video_connection_attempts

    if not force:
        for attempt in video_connection_attempts:
            if attempt.get("url") == target_url and attempt.get("thread") and attempt["thread"].is_alive():
                return

    result_queue = queue.Queue(maxsize=1)

    def worker(url, result_queue):
        new_cap = None
        try:
            print(f"[VIDEO] Intentando conectar a {url} (async)...")
            new_cap = ThreadedVideoCapture(url)
            if not new_cap.is_valid():
                new_cap.release()
                new_cap = None
        except Exception as e:
            print(f"[VIDEO] Error al iniciar conexión: {e}")
            if new_cap:
                new_cap.release()
                new_cap = None
        finally:
            try:
                result_queue.put((url, new_cap), timeout=1)
            except queue.Full:
                if new_cap:
                    new_cap.release()

    thread = threading.Thread(target=worker, args=(target_url, result_queue), daemon=True)
    video_connection_attempts.append({"thread": thread, "queue": result_queue, "url": target_url})
    thread.start()


def process_pending_video_connections(current_cap, current_url):
    """
    Revisa conexiones finalizadas y retorna (cap, se_asignó_nuevo_cap).
    """
    global video_connection_attempts

    if not video_connection_attempts:
        return current_cap, False

    new_cap_assigned = False
    remaining_attempts = []

    for attempt in video_connection_attempts:
        q = attempt["queue"]
        try:
            result_url, candidate_cap = q.get_nowait()
        except queue.Empty:
            remaining_attempts.append(attempt)
            continue

        if candidate_cap and candidate_cap.is_valid() and current_cap is None and result_url == current_url and not new_cap_assigned:
            current_cap = candidate_cap
            new_cap_assigned = True
            print(f"[VIDEO] Conexión establecida con {result_url}")
        else:
            if candidate_cap:
                candidate_cap.release()

    video_connection_attempts = remaining_attempts
    return current_cap, new_cap_assigned


def ensure_windows_cursor(window_title):
    """
    Reemplaza el cursor en ventanas OpenCV por el puntero clásico.
    """
    global windows_cursor_fixed, windows_cursor_warning

    if windows_cursor_fixed or os.name != "nt":
        return

    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return

        IDC_ARROW = 32512
        cursor = user32.LoadCursorW(None, IDC_ARROW)
        GCL_HCURSOR = -12

        if ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_long):
            user32.SetClassLongW(hwnd, GCL_HCURSOR, cursor)
        else:
            user32.SetClassLongPtrW(hwnd, GCL_HCURSOR, cursor)
        user32.SetCursor(cursor)
        windows_cursor_fixed = True
        print("[WINDOWS] Cursor estándar aplicado.")
    except Exception as e:
        if not windows_cursor_warning:
            print(f"[WINDOWS] No se pudo ajustar el cursor: {e}")
            windows_cursor_warning = True

# --- VISUALIZACIÓN OPTIMIZADA ---

def overlay_audio_spectrogram(frame):
    """
    Superpone el espectrograma.
    """
    if not spectrogram_render_active:
        return frame
    
    # Intentar obtener la imagen más reciente del worker
    spectrogram_img = None
    with spectrogram_image_lock:
        if spectrogram_image_ready is not None:
            spectrogram_img = spectrogram_image_ready.copy()
            
    if spectrogram_img is None:
        return frame
    
    try:
        h, w = frame.shape[:2]
        oh, ow = spectrogram_img.shape[:2]
        
        # Redimensionar si es necesario (el worker ya lo hace aprox, pero ajustamos al frame actual)
        target_h = int(h * 0.22)
        target_w = int(w * 0.35)
        
        if oh != target_h or ow != target_w:
             spectrogram_img = cv2.resize(spectrogram_img, (target_w, target_h))
             oh, ow = target_h, target_w

        # Bajar el espectrograma para evitar solapamiento con sliders de YOLO
        # Los sliders de YOLO están en y=105-161 (2 sliders de 50px + spacing)
        y_offset = 250
        x_offset = 10
        
        # Alpha blending
        alpha = (spectrogram_img[:, :, 3] / 255.0) * 0.5
        
        for c in range(3):
            frame[y_offset:y_offset+oh, x_offset:x_offset+ow, c] = \
                frame[y_offset:y_offset+oh, x_offset:x_offset+ow, c] * (1 - alpha) + \
                spectrogram_img[:, :, c] * alpha
        
    except Exception as e:
        pass # Ignorar errores puntuales de renderizado para no congelar video
    
    return frame

# --- AUDIO STREAMING ---
def stream_audio():
    global audio_stream, stop_audio_thread
    
    max_retries = 5
    retry_delay = 3  # Aumentar el tiempo de espera para que el servidor limpie conexiones anteriores
    
    for attempt in range(max_retries):
        if stop_audio_thread:
            return
            
        try:
            # Usar timeout más largo: (connect_timeout, read_timeout)
            # connect_timeout: tiempo para establecer conexión
            # read_timeout: tiempo entre chunks de datos
            with requests.get(audio_url, stream=True, timeout=(15, 30), headers=headers) as r:
                if r.status_code == 503:
                    # Servicio no disponible - probablemente hay clientes anteriores que no se han limpiado
                    # Esperar más tiempo para que el servidor detecte las desconexiones
                    if attempt < max_retries - 1:
                        wait_time = retry_delay + (attempt * 1)  # Aumentar el tiempo de espera progresivamente
                        print(f"Error audio: Servicio no disponible (HTTP 503). Esperando {wait_time} segundos para que el servidor limpie conexiones anteriores... ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("Error audio: Servicio no disponible (HTTP 503) después de varios intentos.")
                        print("Sugerencia: Detén y vuelve a iniciar el streaming en la app Android, o espera unos segundos y vuelve a intentar.")
                        return
                elif r.status_code != 200:
                    print(f"Error audio: HTTP {r.status_code}")
                    return
                
                # Obtener metadatos del stream para ajustar sample rate / canales
                content_type = r.headers.get('Content-Type', '')
                parsed_sample_rate = 44100
                parsed_channels = 1
                if content_type:
                    for part in content_type.split(';'):
                        part = part.strip().lower()
                        if part.startswith('rate='):
                            try:
                                parsed_sample_rate = int(part.split('=')[1])
                            except (ValueError, IndexError):
                                parsed_sample_rate = 44100
                        elif part.startswith('channels='):
                            try:
                                parsed_channels = int(part.split('=')[1])
                            except (ValueError, IndexError):
                                parsed_channels = 1
                parsed_sample_rate = max(8000, min(parsed_sample_rate, 96000))
                parsed_channels = max(1, min(parsed_channels, 2))

                print(f"[AUDIO] Stream configurado: {parsed_channels} canal(es) @ {parsed_sample_rate} Hz")

                # El servidor Android envía PCM crudo directamente, sin header WAV
                # Inicializar PyAudio antes de leer datos
                audio_stream = p.open(format=pyaudio.paInt16,
                                      channels=parsed_channels,
                                      rate=parsed_sample_rate,
                                      output=True,
                                      frames_per_buffer=CHUNK)
                
                # Leer chunks de PCM directamente (el timeout ya está configurado en la petición)
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if stop_audio_thread:
                        break
                    if chunk and audio_stream:
                        try:
                            # Solo reproducir si no está muteado
                            if not audio_playback_muted:
                                audio_stream.write(chunk)
                            
                            if audio_detection_enabled:
                                try:
                                    audio_buffer.put_nowait(chunk)
                                except queue.Full:
                                    pass  # Buffer lleno, descartar chunk
                                    
                        except Exception as e:
                            print(f"Error audio escribiendo chunk: {e}")
                            break
                
                # Si llegamos aquí, la conexión se estableció correctamente
                break
                        
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                print(f"Error audio: Timeout. Reintentando en {retry_delay} segundos... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                print(f"Error audio: Timeout después de {max_retries} intentos - {e}")
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                print(f"Error audio: No se pudo conectar. Reintentando en {retry_delay} segundos... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                print(f"Error audio: No se pudo conectar al servidor después de {max_retries} intentos - {e}")
        except Exception as e:
            print(f"Error audio: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            break
    
    # Limpiar recursos
    if audio_stream:
        try:
            audio_stream.stop_stream()
            audio_stream.close()
            audio_stream = None
        except:
            pass

def start_audio():
    global audio_thread, stop_audio_thread, audio_enabled
    
    # Si hay un hilo activo, detenerlo primero y esperar a que termine
    if audio_thread is not None and audio_thread.is_alive():
        stop_audio_thread = True
        audio_thread.join(timeout=3)
    
    # Esperar un poco para que el servidor Android detecte la desconexión anterior
    time.sleep(1)
    
    stop_audio_thread = False
    audio_enabled = True
    audio_thread = threading.Thread(target=stream_audio, daemon=True)
    audio_thread.start()
    print("Audio iniciado")

def stop_audio():
    global stop_audio_thread, audio_enabled, audio_thread, audio_detection_enabled
    
    if audio_thread is None or not audio_thread.is_alive():
        return
    
    if audio_detection_enabled:
        toggle_audio_detection()
    
    stop_audio_thread = True
    audio_enabled = False
    
    if audio_thread:
        audio_thread.join(timeout=2)
    
    print("Audio detenido")

def toggle_audio_mute():
    """Mute/Unmute el playback de audio sin afectar la detección"""
    global audio_playback_muted
    audio_playback_muted = not audio_playback_muted
    status = "MUTE" if audio_playback_muted else "UNMUTE"
    print(f"[AUDIO] Playback {status}")

def cambiar_ip_camara(cap_actual, nueva_ip=None):
    if audio_enabled:
        stop_audio()
    
    if cap_actual is not None:
        cap_actual.release()
    
    if nueva_ip is None:
        nueva_ip = solicitar_nueva_ip(ip_y_puerto)
    
    if nueva_ip:
        update_stream_endpoints(nueva_ip, record_wifi=True)
        schedule_video_connection(video_url, force=True)

    return None

# --- INDICADORES ---
def draw_interactive_button(frame, text, x_start, y_center, w, h, text_color, mouse_pos, click_pos, align_right=False):
    """
    Dibuja un botón redondeado transparente con efecto hover y detección de clic.
    Retorna: (frame_modificado, fue_cliqueado)
    """
    # Coordenadas de la caja
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    padding_x = 20
    padding_y = 12

    width = text_size[0] + padding_x
    height = text_size[1] + padding_y
    if w > 0:
        width = max(width, w)
    if h > 0:
        height = max(height, h)

    y1 = int(y_center - text_size[1] - padding_y / 2)
    y2 = y1 + int(height)

    if align_right:
        x2 = int(x_start)
        x1 = x2 - int(width)
        text_x = x1 + padding_x // 2
    else:
        x1 = int(x_start)
        x2 = x1 + int(width)
        text_x = x1 + padding_x // 2
    
    # Detectar Hover
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2) and (y1 <= my <= y2)
    
    # Detectar Clic
    is_clicked = False
    if click_pos:
        cx, cy = click_pos
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            is_clicked = True

    # Configuración visual
    overlay = frame.copy()
    bg_color = (0, 0, 0)
    alpha = 0.6 if is_hover else 0.4 # Más opaco si pasas el ratón
    radius = 10
    
    # Dibujar rectángulo redondeado
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), bg_color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), bg_color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, bg_color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, bg_color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, bg_color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, bg_color, -1)
    
    # Aplicar transparenciaopen_
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Texto
    text_y = y2 - padding_y // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
    return frame, is_clicked


def draw_yolo_indicator(frame, mouse_pos, click_pos, detecciones=0):
    x = frame.shape[1] - 40
    y = 20
    
    if yolo_enabled:
        color = (0, 255, 0)
        text = t('yolo_on', detecciones)
    else:
        color = (0, 0, 255)
        text = t('yolo_off')
    
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)

def draw_yolo_settings_icon(frame, mouse_pos, click_pos):
    """Dibuja el icono PNG de ajustes para YOLO."""
    icon = get_yolo_settings_icon()
    if icon is None:
        return frame, False

    h, w = icon.shape[:2]
    padding = 10
    x2 = frame.shape[1] - 10
    x1 = x2 - w
    y1 = 15 - h // 2
    y2 = y1 + h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    roi = frame[y1:y2, x1:x2]
    icon_resized = icon[: y2 - y1, : x2 - x1]

    if icon_resized.shape[2] == 4:
        alpha = icon_resized[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * icon_resized[:, :, c]
    else:
        roi[:] = icon_resized

    mx, my = mouse_pos
    is_hover = x1 <= mx <= x2 and y1 <= my <= y2
    is_clicked = False
    if click_pos:
        cx_click, cy_click = click_pos
        if x1 <= cx_click <= x2 and y1 <= cy_click <= y2:
            is_clicked = True

    if is_hover:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    return frame, is_clicked

def draw_tinysa_indicator(frame, mouse_pos, click_pos):
    x = frame.shape[1] - 40
    y = 50
    
    if tinysa_running:
        color = (0, 255, 0)
        text = t('tinysa_on')
    else:
        color = (0, 0, 255)
        text = t('tinysa_off')
    
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)

def draw_tinysa_settings_icon(frame, mouse_pos, click_pos):
    """Dibuja el icono PNG de ajustes para TinySA."""
    icon = get_yolo_settings_icon()  # Reutilizamos el mismo icono
    if icon is None:
        return frame, False

    h, w = icon.shape[:2]
    padding = 10
    x2 = frame.shape[1] - 10
    x1 = x2 - w
    y1 = 45 - h // 2  # Posición al lado de TinySA (y=50)
    y2 = y1 + h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    roi = frame[y1:y2, x1:x2]
    icon_resized = icon[: y2 - y1, : x2 - x1]

    if icon_resized.shape[2] == 4:
        alpha = icon_resized[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * icon_resized[:, :, c]
    else:
        roi[:] = icon_resized

    mx, my = mouse_pos
    is_hover = x1 <= mx <= x2 and y1 <= my <= y2
    is_clicked = False
    if click_pos:
        cx_click, cy_click = click_pos
        if x1 <= cx_click <= x2 and y1 <= cy_click <= y2:
            is_clicked = True

    if is_hover:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    return frame, is_clicked

def draw_audio_volume_icon(frame, mouse_pos, click_pos):
    """Dibuja el icono de volumen (mute.png o vol.png) a la izquierda de DET AUDIO."""
    x_text = frame.shape[1] - 40
    y_text = 80
    
    # Posición del icono: más a la izquierda y ligeramente más arriba
    # Mostrar mute si audio desactivado O si playback está muteado
    icon = get_audio_volume_icon(muted=(not audio_enabled) or audio_playback_muted)
    if icon is None:
        return frame, False
    
    h, w = icon.shape[:2]
    icon_x = x_text - 175  # Un poco más a la izquierda
    icon_y = y_text - h // 2 - 6  # Muy poco más arriba (2-3 píxeles adicionales)
    
    x1 = max(0, icon_x)
    y1 = max(0, icon_y)
    x2 = min(frame.shape[1], icon_x + w)
    y2 = min(frame.shape[0], icon_y + h)
    
    if x2 <= x1 or y2 <= y1:
        return frame, False
    
    roi = frame[y1:y2, x1:x2]
    icon_resized = icon[: y2 - y1, : x2 - x1]
    
    if icon_resized.shape[2] == 4:
        alpha = icon_resized[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * icon_resized[:, :, c]
    else:
        roi[:] = icon_resized
    
    mx, my = mouse_pos
    is_hover = x1 <= mx <= x2 and y1 <= my <= y2
    is_clicked = False
    if click_pos:
        cx_click, cy_click = click_pos
        if x1 <= cx_click <= x2 and y1 <= cy_click <= y2:
            is_clicked = True
    
    if is_hover:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame, is_clicked

def draw_audio_detection_toggle(frame, mouse_pos, click_pos):
    """Dibuja el indicador de detección de audio."""
    x_text = frame.shape[1] - 40
    y = 80
    
    if audio_detection_enabled:
        color = (0, 255, 0)
        text = t('det_audio_on')
    else:
        color = (0, 0, 255)
        text = t('det_audio_off')
    
    # Dibujar el texto como botón interactivo
    return draw_interactive_button(frame, text, x_text, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def open_yolo_options_dialog():
    """Abre la ventana de opciones YOLO en un hilo aparte."""
    global yolo_options_thread
    if yolo_options_thread and yolo_options_thread.is_alive():
        return

    def runner():
        try:
            show_yolo_options_window()
        finally:
            yolo_options_thread = None

    yolo_options_thread = threading.Thread(target=runner, daemon=True)
    yolo_options_thread.start()


def draw_ip_indicator(frame, mouse_pos, click_pos):
    x = 10
    y = 20
    text = t('ip_label', ip_y_puerto)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    padding_x = 14
    padding_y = 10
    x1 = x - 6
    y1 = y - text_size[1] - padding_y // 2
    x2 = x1 + text_size[0] + padding_x
    y2 = y + padding_y // 2

    x1 = max(0, x1)
    y1 = max(0, y1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return frame, False


def draw_adb_message(frame):
    if not adb_connected:
        return frame
    text = t('adb_connected')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10, 85), font, 0.6, (0, 255, 255), 2)
    return frame


def draw_tinysa_message(frame):
    if not tinysa_detected:
        return frame
    # Mostrar mensaje diferente según el modo de conexión
    if tinysa_use_http:
        text = t('tinysa_connected_android')
    else:
        text = t('tinysa_connected')
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Posición abajo a la izquierda
    x = 10
    y = frame.shape[0] - 15  # Abajo del frame
    cv2.putText(frame, text, (x, y), font, 0.55, (0, 255, 255), 2)
    
    # Mostrar alerta de dron detectado por RF
    with rf_drone_detection_lock:
        rf_result = rf_drone_detection_result.copy()
    
    if rf_result.get("is_drone", False) and rf_drone_detection_enabled:
        confidence = rf_result.get("confidence", 0.0)
        frequency = rf_result.get("frequency")
        
        if frequency:
            freq_mhz = frequency / 1e6
            alert_text = t('rf_drone_detected', freq_mhz, int(confidence * 100))
        else:
            alert_text = t('rf_drone_detected_no_freq', int(confidence * 100))
        
        # Dibujar fondo semitransparente rojo para la alerta
        text_size, _ = cv2.getTextSize(alert_text, font, 0.7, 2)
        text_w, text_h = text_size
        alert_x = 10
        alert_y = y - text_h - 25
        
        # Rectángulo de fondo
        cv2.rectangle(
            frame,
            (alert_x - 5, alert_y - 5),
            (alert_x + text_w + 5, alert_y + text_h + 5),
            (0, 0, 255),  # Rojo
            -1
        )
        
        # Texto de alerta
        cv2.putText(
            frame,
            alert_text,
            (alert_x, alert_y + text_h),
            font,
            0.7,
            (255, 255, 255),  # Blanco
            2
        )
    
    return frame

def draw_ip_settings_icon(frame, mouse_pos, click_pos):
    icon = get_yolo_settings_icon()
    if icon is None:
        return frame, False

    h, w = icon.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(f"IP: {ip_y_puerto}", font, 0.5, 2)
    x2 = 10 + text_size[0] + 20 + w
    x1 = x2 - w
    y_center = 15
    y1 = y_center - h // 2
    y2 = y1 + h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    roi = frame[y1:y2, x1:x2]
    icon_resized = icon[: y2 - y1, : x2 - x1]

    if icon_resized.shape[2] == 4:
        alpha = icon_resized[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * icon_resized[:, :, c]
    else:
        roi[:] = icon_resized

    mx, my = mouse_pos
    is_hover = x1 <= mx <= x2 and y1 <= my <= y2
    is_clicked = False
    if click_pos:
        cx_click, cy_click = click_pos
        if x1 <= cx_click <= x2 and y1 <= cy_click <= y2:
            is_clicked = True

    if is_hover:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    return frame, is_clicked


def open_ip_change_dialog():
    global ip_dialog_thread
    if ip_dialog_thread and ip_dialog_thread.is_alive():
        return

    def runner():
        global pending_ip_change, ip_dialog_thread
        nueva_ip = solicitar_nueva_ip(ip_y_puerto)
        if nueva_ip and nueva_ip.strip():
            pending_ip_change = nueva_ip.strip()
        ip_dialog_thread = None

    ip_dialog_thread = threading.Thread(target=runner, daemon=True)
    ip_dialog_thread.start()


def apply_pending_ip_change(cap_actual):
    global pending_ip_change
    if pending_ip_change:
        nueva_ip = pending_ip_change
        pending_ip_change = None
        cap_actual = cambiar_ip_camara(cap_actual, nueva_ip=nueva_ip)
    return cap_actual


def setup_adb_forward():
    try:
        subprocess.run(["adb", "forward", "--remove", "tcp:8080"], capture_output=True, text=True, timeout=3)
    except Exception:
        pass
    try:
        subprocess.run(["adb", "forward", "tcp:8080", "tcp:8080"], check=True, capture_output=True, text=True, timeout=3)
        return True
    except Exception as e:
        print(f"[ADB] Error configurando forward: {e}")
        return False


def teardown_adb_forward():
    try:
        subprocess.run(["adb", "forward", "--remove", "tcp:8080"], capture_output=True, text=True, timeout=3)
    except Exception:
        pass


def poll_adb_connection():
    global last_adb_check, adb_connected, pending_ip_change, adb_message_timer, last_wifi_ip

    now = time.time()
    if now - last_adb_check < ADB_CHECK_INTERVAL:
        return
    last_adb_check = now

    if shutil.which("adb") is None:
        if adb_connected:
            adb_connected = False
        return

    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=3
        )
        lines = result.stdout.strip().splitlines()
        connected = any("\tdevice" in line for line in lines[1:])
    except Exception:
        connected = False

    if connected and not adb_connected:
        if last_wifi_ip is None or (not last_wifi_ip or last_wifi_ip == ADB_TARGET_IP):
            last_wifi_ip = ip_y_puerto
        if setup_adb_forward():
            adb_connected = True
            pending_ip_change = ADB_TARGET_IP
            print("[ADB] Conectado. Cambiando IP a 127.0.0.1.")
        else:
            print("[ADB] Fallo configurando el túnel. Mantengo IP actual.")
    elif not connected and adb_connected:
        adb_connected = False
        teardown_adb_forward()
        if last_wifi_ip and last_wifi_ip != ADB_TARGET_IP:
            pending_ip_change = last_wifi_ip
            print("[ADB] Desconectado. Volviendo a IP anterior.")


def poll_tinysa_presence(force=False):
    """
    Verifica si TinySA está conectado (localmente vía USB o en el servidor Android).
    """
    global tinysa_last_check, tinysa_detected, tinysa_use_http
    now = time.time()
    if not force and now - tinysa_last_check < TIN_YSA_CHECK_INTERVAL:
        return
    tinysa_last_check = now
    
    # Verificar puerto local primero
    port = find_tinysa_port()
    if port is not None:
        tinysa_detected = True
        tinysa_use_http = False
    else:
        # Si no está localmente, verificar servidor Android
        try:
            status_url = base_url + "/tinysa/status"
            response = requests.get(status_url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                is_connected = data.get("connected", False)
                tinysa_detected = is_connected
                if is_connected:
                    tinysa_use_http = True
                else:
                    tinysa_detected = False
            else:
                tinysa_detected = False
        except Exception:
            tinysa_detected = False

def show_yolo_options_window():
    """Ventana para gestionar modelos YOLO."""
    global yolo_model_slots, yolo_default_slot

    root = tk.Tk()
    root.title(t('yolo_options_title'))
    root.attributes("-topmost", True)
    root.resizable(False, False)

    main_frame = ttk.Frame(root, padding=15)
    main_frame.pack(fill="both", expand=True)

    ttk.Label(main_frame, text=t('available_models'), font=("Arial", 11, "bold")).pack(anchor="w")

    slots_frame = ttk.Frame(main_frame)
    slots_frame.pack(fill="both", expand=True, pady=(10, 15))

    total_slots = len(yolo_model_slots)
    slots_per_page = 5
    total_pages = max(1, (total_slots + slots_per_page - 1) // slots_per_page)
    current_page = tk.IntVar(value=0)

    path_vars = [tk.StringVar(value=slot.get("path", "")) for slot in yolo_model_slots]
    desc_vars = [tk.StringVar(value=slot.get("description", "")) for slot in yolo_model_slots]
    selected_var = tk.IntVar(value=yolo_default_slot)

    def browse_file(idx):
        filepath = filedialog.askopenfilename(
            title=t('select_yolo_model'),
            filetypes=[(t('yolo_models'), "*.pt"), (t('all_files'), "*.*")],
            parent=root
        )
        if filepath:
            path_vars[idx].set(filepath)

    total_slots = len(yolo_model_slots)
    slots_per_page = 5
    total_pages = max(1, (total_slots + slots_per_page - 1) // slots_per_page)
    current_page = tk.IntVar(value=0)

    rows_container = ttk.Frame(slots_frame)
    rows_container.pack(fill="both", expand=True)

    def build_page(page_idx):
        for child in rows_container.winfo_children():
            child.destroy()

        start_idx = page_idx * slots_per_page
        end_idx = min(start_idx + slots_per_page, total_slots)

        for idx in range(start_idx, end_idx):
            frame_slot = ttk.Frame(rows_container, padding=5)
            frame_slot.pack(fill="x", pady=3)

            ttk.Radiobutton(frame_slot, variable=selected_var, value=idx).grid(row=0, column=0, rowspan=2, padx=(0, 8))
            ttk.Label(frame_slot, text=t('model', idx + 1)).grid(row=0, column=1, sticky="w")
            entry_path = ttk.Entry(frame_slot, textvariable=path_vars[idx], width=45)
            entry_path.grid(row=0, column=2, padx=5, sticky="we")
            ttk.Button(frame_slot, text=t('browse'), command=lambda i=idx: browse_file(i)).grid(row=0, column=3, padx=5)
            ttk.Label(frame_slot, text=t('description')).grid(row=1, column=1, sticky="e", pady=2)
            ttk.Entry(frame_slot, textvariable=desc_vars[idx], width=45).grid(row=1, column=2, padx=5, sticky="we")
            frame_slot.columnconfigure(2, weight=1)

    nav_frame = ttk.Frame(main_frame)
    nav_frame.pack(fill="x", pady=(5, 5))

    page_label_var = tk.StringVar()

    def update_page_label():
        page_label_var.set(t('page', current_page.get() + 1, total_pages))

    def go_prev():
        if current_page.get() > 0:
            current_page.set(current_page.get() - 1)
            build_page(current_page.get())
            update_page_label()

    def go_next():
        if current_page.get() < total_pages - 1:
            current_page.set(current_page.get() + 1)
            build_page(current_page.get())
            update_page_label()

    ttk.Button(nav_frame, text="◀", width=3, command=go_prev).pack(side="left")
    ttk.Label(nav_frame, textvariable=page_label_var).pack(side="left", padx=10)
    ttk.Button(nav_frame, text="▶", width=3, command=go_next).pack(side="left")

    build_page(0)
    update_page_label()

    status_var = tk.StringVar(value="")

    def sync_slots():
        for idx in range(len(yolo_model_slots)):
            yolo_model_slots[idx]["path"] = path_vars[idx].get().strip()
            yolo_model_slots[idx]["description"] = desc_vars[idx].get().strip()

    def apply_action(save_default=False, reset_default=False):
        sync_slots()
        slot_idx = selected_var.get()

        if reset_default:
            yolo_model_slots[0]["path"] = YOLO_DEFAULT_MODEL
            yolo_model_slots[0]["description"] = "Modelo por defecto"
            path_vars[0].set(YOLO_DEFAULT_MODEL)
            desc_vars[0].set("Modelo por defecto")
            slot_idx = 0
            save_default = True

        path = yolo_model_slots[slot_idx]["path"]
        if not path:
            messagebox.showerror(t('error'), t('model_empty', slot_idx + 1))
            return
        
        # Normalizar la ruta antes de verificar existencia
        normalized_path = normalize_model_path(path)
        if not normalized_path or not os.path.exists(normalized_path):
            messagebox.showerror(t('error'), t('file_not_found', path))
            return

        if apply_yolo_model(normalized_path, save_default=save_default, selected_slot=slot_idx if save_default else None):
            status_var.set(t('model_updated'))
            root.destroy()

    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=(5, 10))

    ttk.Button(btn_frame, text=t('load_model'), command=lambda: apply_action(False)).pack(side="left", padx=5)
    ttk.Button(btn_frame, text=t('load_and_save_default'), command=lambda: apply_action(True)).pack(side="left", padx=5)
    ttk.Button(btn_frame, text=t('load_default_config'), command=lambda: apply_action(True, True)).pack(side="left", padx=5)
    ttk.Button(btn_frame, text=t('cancel'), command=root.destroy).pack(side="right", padx=5)

    ttk.Label(main_frame, textvariable=status_var, foreground="#0077cc").pack(anchor="w")

    root.mainloop()


def draw_slider_control(frame, label, value, min_val, max_val, origin, size, mouse_pos, click_pos, slider_key):
    """Dibuja un slider semi-transparente y devuelve nuevo valor si se hizo click."""
    global yolo_slider_active, rf_slider_active
    x, y = origin
    width, height = size
    overlay = frame.copy()

    # Panel
    panel_y1 = y
    panel_y2 = y + height
    cv2.rectangle(overlay, (x, panel_y1), (x + width, panel_y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Textos
    text_y = panel_y1 + 18
    cv2.putText(frame, label, (x + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)
    cv2.putText(frame, f"{value:.2f}", (x + width - 55, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)

    slider_offset = 10
    panel_y1 += slider_offset
    panel_y2 += slider_offset

    # Slider track semitransparente
    track_x1 = x + 20
    track_x2 = x + width - 20
    track_y = panel_y1 + height - 25
    track_overlay = frame.copy()
    cv2.line(track_overlay, (track_x1, track_y), (track_x2, track_y), (210, 210, 210), 6, cv2.LINE_AA)
    cv2.addWeighted(track_overlay, 0.4, frame, 0.6, 0, frame)

    # Handle
    ratio = (value - min_val) / (max_val - min_val)
    ratio = max(0.0, min(1.0, ratio))
    handle_x = int(track_x1 + ratio * (track_x2 - track_x1))
    handle_overlay = frame.copy()
    cv2.circle(handle_overlay, (handle_x, track_y), 10, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(handle_overlay, (handle_x, track_y), 10, (0, 102, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(handle_overlay, 0.7, frame, 0.3, 0, frame)

    # Clic
    new_value = None
    active_slider = yolo_slider_active if slider_key.startswith("conf") or slider_key.startswith("iou") else rf_slider_active
    
    if click_pos:
        cx, cy = click_pos
        if track_y - 18 <= cy <= track_y + 18 and track_x1 <= cx <= track_x2:
            ratio = (cx - track_x1) / (track_x2 - track_x1)
            ratio = max(0.0, min(1.0, ratio))
            new_value = min_val + ratio * (max_val - min_val)
            if slider_key.startswith("rf_"):
                rf_slider_active = slider_key
            else:
                yolo_slider_active = slider_key
    elif active_slider == slider_key and mouse_is_down:
        mx, my = mouse_pos
        ratio = (mx - track_x1) / (track_x2 - track_x1)
        ratio = max(0.0, min(1.0, ratio))
        new_value = min_val + ratio * (max_val - min_val)
    elif active_slider == slider_key and not mouse_is_down:
        if slider_key.startswith("rf_"):
            rf_slider_active = None
        else:
            yolo_slider_active = None

    return frame, new_value


def draw_yolo_sliders(frame, mouse_pos, click_pos):
    """Muestra sliders de parámetros YOLO si está activado."""
    global yolo_conf_threshold, yolo_iou_threshold
    if not yolo_enabled:
        return frame, click_pos

    slider_width = int(frame.shape[1] * 0.16)
    slider_height = 50
    x = 50
    y_start = 105
    spacing = 6

    specs = [
        ("Confidence threshold", yolo_conf_threshold, 0.05, 0.99, "conf"),
        ("IoU threshold", yolo_iou_threshold, 0.05, 0.99, "iou"),
    ]

    # Working copy of click to avoid multiple updates con el mismo clic
    remaining_click = click_pos

    for idx, (label, value, v_min, v_max, key) in enumerate(specs):
        y = y_start + idx * (slider_height + spacing)
        frame, new_val = draw_slider_control(
            frame,
            label,
            value,
            v_min,
            v_max,
            (x, y),
            (slider_width, slider_height),
            mouse_pos,
            remaining_click,
            key
        )

        if new_val is not None:
            with yolo_threshold_lock:
                if key == "conf":
                    yolo_conf_threshold = new_val
                else:
                    yolo_iou_threshold = new_val
            remaining_click = None

    return frame, remaining_click

def draw_rf_drone_sliders(frame, mouse_pos, click_pos):
    """Muestra sliders de parámetros de detección RF de drones si está activado."""
    global rf_peak_threshold, rf_min_peak_height_db, rf_min_peak_width_mhz, rf_max_peak_width_mhz
    global rf_sliders_visible
    
    if not rf_sliders_visible or not tinysa_running:
        return frame, click_pos

    slider_width = int(frame.shape[1] * 0.20)
    slider_height = 50
    x = 50
    y_start = 105
    spacing = 6

    specs = [
        ("Umbral Potencia (dBm)", rf_peak_threshold, -100.0, -50.0, "rf_peak_thresh"),
        ("Altura Min Ruido (dB)", rf_min_peak_height_db, 1.0, 40.0, "rf_min_height"),
        ("Ancho Min (MHz)", rf_min_peak_width_mhz, 1.0, 30.0, "rf_min_width"),
        ("Ancho Max (MHz)", rf_max_peak_width_mhz, 20.0, 80.0, "rf_max_width"),
    ]

    # Working copy of click to avoid multiple updates con el mismo clic
    remaining_click = click_pos

    for idx, (label, value, v_min, v_max, key) in enumerate(specs):
        y = y_start + idx * (slider_height + spacing)
        frame, new_val = draw_slider_control(
            frame,
            label,
            value,
            v_min,
            v_max,
            (x, y),
            (slider_width, slider_height),
            mouse_pos,
            remaining_click,
            key
        )

        if new_val is not None:
            with rf_detection_params_lock:
                if key == "rf_peak_thresh":
                    rf_peak_threshold = new_val
                elif key == "rf_min_height":
                    rf_min_peak_height_db = new_val
                elif key == "rf_min_width":
                    rf_min_peak_width_mhz = new_val
                elif key == "rf_max_width":
                    rf_max_peak_width_mhz = new_val
            print(f"[RF SLIDER] {label}: {new_val:.2f}")
            remaining_click = None  # Consumir el click

    return frame, remaining_click

def draw_audio_detection_indicator(frame):
    global audio_detection_alert_time, audio_detection_max_confidence, AUDIO_VISUAL_MULTIPLIER
    
    if not audio_detection_enabled:
        return frame
    
    with audio_detection_lock:
        is_drone = audio_detection_result["is_drone"]
        confidence = audio_detection_result["confidence"]  # Ya viene con multiplicador aplicado
    
    y = frame.shape[0] - 30
    
    if is_drone:
        blink = int(time.time() * 2) % 2 == 0
        color = (0, 255, 255) if blink else (0, 128, 128)
        # Mostrar timestamp de la alerta si está disponible
        if audio_detection_alert_time is not None:
            alert_time_str = time.strftime("%H:%M:%S", time.localtime(audio_detection_alert_time))
            # Mostrar el máximo alcanzado durante la alerta (con multiplicador visual)
            max_visual = min(100, int(audio_detection_max_confidence * AUDIO_VISUAL_MULTIPLIER * 100))
            text = f"AUDIO DRON DETECTADO A LAS {alert_time_str} - {max_visual}%"
        else:
            text = t('audio_drone_detected', int(confidence * 100))
    else:
        color = (0, 0, 255)
        text = t('no_audio_dron', int(confidence * 100))
    
    # Calcular tamaño del texto primero para posicionarlo correctamente
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    # Posicionar más a la izquierda para que quepa el texto completo
    # Dejar al menos 20 píxeles de margen desde el borde derecho
    x = frame.shape[1] - text_size[0] - 20
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 23), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame


def process_pending_yolo_reload():
    """Reinicia YOLO en el hilo principal si hay cambios de modelo pendientes."""
    global yolo_reload_requested
    if yolo_reload_requested and yolo_enabled:
        yolo_reload_requested = False
        print("[YOLO] Recargando modelo seleccionado...")
        toggle_yolo()
        toggle_yolo()
    elif yolo_reload_requested:
        yolo_reload_requested = False
        # YOLO apagado: solo marcamos para cargar en siguiente activación
        print("[YOLO] Modelo actualizado para próximo inicio.")

def draw_fps_indicator(frame, fps):
    x = 10
    y = 50
    text = t('fps_label', fps)
    
    overlay = frame.copy()
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(overlay, (x - 5, y - 18), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

# --- MAIN ---
print("Iniciando programa FULL THREADED + TinySA Ultra (Modo Síncrono)...")
print("Controles:")
print("  Q - Salir")
print("  M - Audio ON/OFF")
print("  A - Detección audio ON/OFF")
print("  Y - YOLO ON/OFF")
print("  R - Sliders RF ON/OFF")
print("  T - TinySA (RF) ON/OFF")
print("  I - Cambiar IP")

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
DEFAULT_WINDOW_SIZE = (1280, 720)
current_window_size = list(DEFAULT_WINDOW_SIZE)
cv2.resizeWindow(window_name, *DEFAULT_WINDOW_SIZE)
# IMPRESCINDIBLE: Activar el callback del ratón
cv2.setMouseCallback(window_name, mouse_handler)
ensure_windows_cursor(window_name)

cap = None
stop_program = False

schedule_video_connection(video_url, force=True)

detecciones_count = 0
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0.0
prev_frame_id = -1
last_reconnect_try = 0  # Timer para no saturar la reconexión

while not stop_program:
    # 1. Gestión del ratón al inicio del frame
    current_click = click_event_pos
    click_event_pos = None  # Resetear clic
    current_mouse = (mouse_x, mouse_y)
    poll_adb_connection()
    poll_tinysa_presence()
    ensure_windows_cursor(window_name)

    cap, new_cap_ready = process_pending_video_connections(cap, video_url)
    if new_cap_ready:
        fps_start_time = time.time()
        last_reconnect_try = time.time()

    if cap is None:
        # MOSTRAR PANTALLA DE ESPERA (NO SIGNAL)
        frame_negro = np.zeros((DEFAULT_WINDOW_SIZE[1], DEFAULT_WINDOW_SIZE[0], 3), dtype=np.uint8)
        texto = t('no_signal')
        texto2 = t('reconnecting')
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(texto, font, 1.5, 2)
        cv2.putText(frame_negro, texto, ((640-tw)//2, (480+th)//2 - 20), font, 1.5, (255, 255, 255), 2)
        
        (tw2, th2), _ = cv2.getTextSize(texto2, font, 0.7, 1)
        cv2.putText(frame_negro, texto2, ((640-tw2)//2, (480+th2)//2 + 30), font, 0.7, (200, 200, 200), 1)

        # Overlay TinySA incluso sin vídeo
        frame_negro = overlay_tinysa_graph(frame_negro)

        # Controles
        if yolo_enabled:
            frame_negro, current_click = draw_yolo_sliders(frame_negro, current_mouse, current_click)
        if tinysa_running:
            frame_negro, current_click = draw_rf_drone_sliders(frame_negro, current_mouse, current_click)

        frame_negro, _ = draw_ip_indicator(frame_negro, current_mouse, current_click)
        frame_negro = draw_adb_message(frame_negro)
        frame_negro, ip_settings_clicked = draw_ip_settings_icon(frame_negro, current_mouse, current_click)
        if ip_settings_clicked:
            open_ip_change_dialog()
            current_click = None
             
        frame_negro, tinysa_clicked = draw_tinysa_indicator(frame_negro, current_mouse, current_click)
        if tinysa_clicked:
             toggle_tinysa()
        frame_negro, tinysa_settings_clicked = draw_tinysa_settings_icon(frame_negro, current_mouse, current_click)
        if tinysa_settings_clicked:
            open_tinysa_options_dialog()
            current_click = None
        
        frame_negro, yolo_clicked = draw_yolo_indicator(frame_negro, current_mouse, current_click)
        if yolo_clicked:
            def show_no_streaming_yolo():
                root = Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                messagebox.showwarning(t('no_streaming'), t('no_streaming_yolo'))
                root.destroy()
            threading.Thread(target=show_no_streaming_yolo, daemon=True).start()
            current_click = None
        frame_negro, yolo_settings_clicked = draw_yolo_settings_icon(frame_negro, current_mouse, current_click)
        if yolo_settings_clicked:
            open_yolo_options_dialog()
            current_click = None
        
        # Icono de volumen de audio
        frame_negro, volume_icon_clicked = draw_audio_volume_icon(frame_negro, current_mouse, current_click)
        if volume_icon_clicked:
            if cap is None:
                def show_no_streaming():
                    root = Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    messagebox.showwarning(t('no_streaming'), t('no_streaming'))
                    root.destroy()
                threading.Thread(target=show_no_streaming, daemon=True).start()
            else:
                if audio_enabled: 
                    stop_audio()
                else: 
                    start_audio()
            current_click = None
        
        frame_negro, audio_det_clicked = draw_audio_detection_toggle(frame_negro, current_mouse, current_click)
        if audio_det_clicked:
            # toggle_audio_detection() inicia el stream automáticamente si no está activo
            toggle_audio_detection()
            current_click = None
        
        # Tailscale
        frame_negro, tailscale_clicked = draw_tailscale_indicator(frame_negro, current_mouse, current_click)
        if tailscale_clicked:
            toggle_tailscale()
            current_click = None
        frame_negro, tailscale_settings_clicked = draw_tailscale_settings_icon(frame_negro, current_mouse, current_click)
        if tailscale_settings_clicked:
            open_tailscale_options_dialog()
            current_click = None
        
        # Idioma APP
        frame_negro, language_clicked = draw_language_indicator(frame_negro, current_mouse, current_click)
        if language_clicked:
            def show_language_dialog():
                show_language_selection_dialog()
            threading.Thread(target=show_language_dialog, daemon=True).start()
            current_click = None

        process_pending_yolo_reload()
        cap = apply_pending_ip_change(cap)
        frame_negro = draw_adb_message(frame_negro)
        frame_negro = draw_tinysa_message(frame_negro)
        
        if tuple(current_window_size) != DEFAULT_WINDOW_SIZE:
            cv2.resizeWindow(window_name, *DEFAULT_WINDOW_SIZE)
            current_window_size[:] = DEFAULT_WINDOW_SIZE
        cv2.imshow(window_name, frame_negro)
        
        # GESTIÓN DE TECLAS EN MODO NO-SIGNAL
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            stop_program = True
            break
        elif key == ord('i') or key == ord('I'):
             cap = cambiar_ip_camara(None)
             continue
        elif key == ord('t') or key == ord('T'):
             toggle_tinysa()
             continue

        # INTENTO DE RECONEXIÓN CONTROLADO
        if time.time() - last_reconnect_try > 2.0:
            last_reconnect_try = time.time()
            schedule_video_connection(video_url)
        
    else:
        # MODO CONECTADO
        ret, frame, current_id = cap.read()
        
        if not ret:
            print("Señal perdida. Cerrando captura...")
            cap.release()
            cap = None
            last_reconnect_try = time.time()
            schedule_video_connection(video_url, force=True)
            continue
        
        if frame is None:
            if tuple(current_window_size) != DEFAULT_WINDOW_SIZE:
                cv2.resizeWindow(window_name, *DEFAULT_WINDOW_SIZE)
                current_window_size[:] = DEFAULT_WINDOW_SIZE
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            continue

        # ESCALAR EL FRAME A 1280x720 ANTES DE PROCESARLO
        # Esto asegura que todas las funciones de dibujo trabajen con el tamaño fijo
        if frame.shape[:2] != (DEFAULT_WINDOW_SIZE[1], DEFAULT_WINDOW_SIZE[0]):
            frame = cv2.resize(frame, DEFAULT_WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)
        
        if current_id != prev_frame_id:
            fps_frame_count += 1
            prev_frame_id = current_id
            
            if yolo_enabled:
                # Enviar el frame escalado a YOLO (YOLO redimensiona internamente para procesamiento)
                enviar_frame_a_yolo(frame)
        
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        
        resultado_yolo = obtener_resultado_yolo()
        if resultado_yolo["boxes_data"]:
            # Las detecciones ya están en el tamaño correcto (1280x720) porque YOLO procesó el frame escalado
            frame = dibujar_detecciones_yolo(frame, resultado_yolo["boxes_data"])
            detecciones_count = resultado_yolo["detecciones"]
        else:
            detecciones_count = 0
        
        # Renderizado capas
        frame = overlay_audio_spectrogram(frame)
        frame = overlay_tinysa_graph(frame) 
        
        # --- DIBUJAR INDICADORES INTERACTIVOS ---
        if yolo_enabled:
            frame, current_click = draw_yolo_sliders(frame, current_mouse, current_click)
        if tinysa_running:
            frame, current_click = draw_rf_drone_sliders(frame, current_mouse, current_click)

        # 1. Icono de volumen de audio
        frame, volume_icon_clicked = draw_audio_volume_icon(frame, current_mouse, current_click)
        if volume_icon_clicked:
            # Verificar si hay streaming
            if cap is None:
                def show_no_streaming():
                    root = Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    messagebox.showwarning(t('no_streaming'), t('no_streaming'))
                    root.destroy()
                threading.Thread(target=show_no_streaming, daemon=True).start()
            else:
                if audio_enabled: 
                    stop_audio()
                else: 
                    start_audio()
            current_click = None
            
        # 2. YOLO
        frame, yolo_clicked = draw_yolo_indicator(frame, current_mouse, current_click, detecciones_count)
        if yolo_clicked:
            if cap is None:
                def show_no_streaming_yolo():
                    root = Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    messagebox.showwarning(t('no_streaming'), t('no_streaming_yolo'))
                    root.destroy()
                threading.Thread(target=show_no_streaming_yolo, daemon=True).start()
                current_click = None
            else:
                toggle_yolo()
        frame, yolo_settings_clicked = draw_yolo_settings_icon(frame, current_mouse, current_click)
        if yolo_settings_clicked:
            open_yolo_options_dialog()
            current_click = None
            
        # 3. TinySA
        frame, tinysa_clicked = draw_tinysa_indicator(frame, current_mouse, current_click)
        if tinysa_clicked:
            toggle_tinysa()
        frame, tinysa_settings_clicked = draw_tinysa_settings_icon(frame, current_mouse, current_click)
        if tinysa_settings_clicked:
            open_tinysa_options_dialog()
            current_click = None
            
        # 4. Detección audio
        frame, audio_det_clicked = draw_audio_detection_toggle(frame, current_mouse, current_click)
        if audio_det_clicked:
            # toggle_audio_detection() inicia el stream automáticamente si no está activo
            toggle_audio_detection()
            current_click = None

        # 5. Tailscale
        frame, tailscale_clicked = draw_tailscale_indicator(frame, current_mouse, current_click)
        if tailscale_clicked:
            toggle_tailscale()
            current_click = None
        frame, tailscale_settings_clicked = draw_tailscale_settings_icon(frame, current_mouse, current_click)
        if tailscale_settings_clicked:
            open_tailscale_options_dialog()
            current_click = None

        # 6. Idioma APP
        frame, language_clicked = draw_language_indicator(frame, current_mouse, current_click)
        if language_clicked:
            def show_language_dialog():
                show_language_selection_dialog()
            threading.Thread(target=show_language_dialog, daemon=True).start()
            current_click = None

        # 6. IP
        frame, _ = draw_ip_indicator(frame, current_mouse, current_click)
        frame = draw_adb_message(frame)
        frame, ip_settings_clicked = draw_ip_settings_icon(frame, current_mouse, current_click)
        if ip_settings_clicked:
            open_ip_change_dialog()
            current_click = None

        process_pending_yolo_reload()
        cap = apply_pending_ip_change(cap)
        frame = draw_adb_message(frame)
        frame = draw_tinysa_message(frame)

        frame = draw_audio_detection_indicator(frame)
        frame = draw_fps_indicator(frame, current_fps)
        
        # Asegurar que la ventana siempre tenga el tamaño correcto
        if tuple(current_window_size) != DEFAULT_WINDOW_SIZE:
            cv2.resizeWindow(window_name, *DEFAULT_WINDOW_SIZE)
            current_window_size[:] = DEFAULT_WINDOW_SIZE
        
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            stop_program = True
            break
        elif key == ord('m') or key == ord('M'):
            if audio_enabled: stop_audio()
            else: start_audio()
        elif key == ord('u') or key == ord('U'):
            # Mute/Unmute playback sin afectar detección
            toggle_audio_mute()
        elif key == ord('a') or key == ord('A'):
            toggle_audio_detection()
        elif key == ord('y') or key == ord('Y'):
            if cap is None:
                def show_no_streaming_yolo_key():
                    root = Tk()
                    root.withdraw()
                    root.attributes("-topmost", True)
                    messagebox.showwarning(t('no_streaming'), t('no_streaming_yolo'))
                    root.destroy()
                threading.Thread(target=show_no_streaming_yolo_key, daemon=True).start()
            else:
                toggle_yolo()
        elif key == ord('t') or key == ord('T'): 
            toggle_tinysa()
        elif key == ord('r') or key == ord('R'):
            rf_sliders_visible = not rf_sliders_visible
            print(f"[RF] Sliders {'activados' if rf_sliders_visible else 'desactivados'}")
        elif key == ord('i') or key == ord('I'):
            cap = cambiar_ip_camara(cap)
        
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            stop_program = True
            break
    except cv2.error:
        stop_program = True
        break

print("Cerrando aplicación...")
if yolo_enabled:
    stop_yolo_worker()
if audio_detection_enabled:
    toggle_audio_detection()
if audio_enabled:
    stop_audio()
if tinysa_running:
    toggle_tinysa() 
if cap is not None:
    cap.release()

p.terminate()
cv2.destroyAllWindows()
print("Programa finalizado.")
