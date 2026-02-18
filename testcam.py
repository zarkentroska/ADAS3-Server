import os
import sys

# Reducir ruido de logs de TensorFlow (INFO) salvo que el usuario lo sobrescriba.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
import copy
import wave
import tempfile
import numpy as np
import requests
import pyaudio
import threading
import tkinter as tk
from tkinter import Tk, messagebox
import json
from collections import deque

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
import matplotlib
# Configurar backend no interactivo para hilos (CRÍTICO para evitar crasheos)
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import queue
import serial
import serial.tools.list_ports
from modules.audio_detection_engine import load_audio_model, run_audio_detection_worker
from modules.adb_bridge import (
    poll_adb_connection as poll_adb_connection_core,
)
from modules.audio_features import extract_features_realtime as extract_audio_features
from modules.connection_runtime import (
    apply_pending_ip_change as apply_pending_ip_change_core,
    cambiar_ip_camara as cambiar_ip_camara_core,
    cargar_ip as cargar_ip_core,
    guardar_ip as guardar_ip_core,
    open_ip_change_dialog as open_ip_change_dialog_core,
    update_stream_endpoints_state as update_stream_endpoints_state_core,
)
from modules.i18n_runtime import (
    cargar_audio_threshold,
    cargar_idioma,
    get_audio_confidence_threshold,
    get_current_language,
    guardar_audio_threshold,
    guardar_idioma,
    initialize_i18n,
    t,
    translate_for_language,
)
from modules.status_overlays import draw_adb_message, draw_fps_indicator, draw_tinysa_message
from modules.tailscale_runtime import (
    get_tailscale_connected_devices as get_tailscale_connected_devices_core,
    get_tailscale_ip as get_tailscale_ip_core,
    get_tailscale_path as get_tailscale_path_core,
    get_tailscale_username as get_tailscale_username_core,
    tailscale_installed as tailscale_installed_core,
    verificar_estado_tailscale as verificar_estado_tailscale_core,
)
from modules.tailscale_control import (
    install_tailscale as install_tailscale_core,
    toggle_tailscale as toggle_tailscale_core,
)
from modules.tinysa_hardware_engine import (
    find_tinysa_port as find_tinysa_port_engine,
    run_tinysa_hardware_worker_http,
    run_tinysa_hardware_worker_serial,
    send_tinysa_command as send_tinysa_command_engine,
)
from modules.telegram_notifier import TelegramEvent, TelegramNotifier
from modules.rf_detection import detect_drone_rf as detect_drone_rf_core
from modules.translations_data import TRANSLATIONS
from modules.ui_helpers import show_warning_async, solicitar_nueva_ip as solicitar_nueva_ip_ui
from modules.ui_assets import (
    get_audio_volume_icon as get_audio_volume_icon_core,
    get_yolo_settings_icon as get_yolo_settings_icon_core,
)
from modules.video_connection import VideoConnectionManager
from modules.yolo_engine import (
    clear_queue_safely,
    draw_yolo_detections,
    load_yolo_model,
    run_yolo_inference_worker,
)
from modules.yolo_models_config import (
    load_yolo_models_config as load_yolo_models_config_core,
    normalize_model_path as normalize_model_path_core,
    save_yolo_models_config as save_yolo_models_config_core,
)
from modules.ui_indicators import (
    draw_audio_detection_indicator as draw_audio_detection_indicator_ui,
    draw_audio_detection_toggle as draw_audio_detection_toggle_ui,
    draw_audio_volume_icon as draw_audio_volume_icon_ui,
    draw_interactive_button as draw_interactive_button_ui,
    draw_ip_indicator as draw_ip_indicator_ui,
    draw_language_indicator as draw_language_indicator_ui,
    draw_telegram_indicator as draw_telegram_indicator_ui,
    draw_ip_settings_icon as draw_ip_settings_icon_ui,
    draw_tailscale_indicator as draw_tailscale_indicator_ui,
    draw_tailscale_settings_icon as draw_tailscale_settings_icon_ui,
    draw_tinysa_settings_icon as draw_tinysa_settings_icon_ui,
    draw_tinysa_indicator as draw_tinysa_indicator_ui,
    draw_yolo_settings_icon as draw_yolo_settings_icon_ui,
    draw_yolo_indicator as draw_yolo_indicator_ui,
)
from modules.ui_language_options import show_language_selection_dialog as show_language_selection_dialog_ui
from modules.ui_telegram_options import show_telegram_options_dialog as show_telegram_options_dialog_ui
from modules.ui_tinysa_options import (
    show_tinysa_menu as show_tinysa_menu_ui,
)
from modules.ui_tailscale_options import show_tailscale_config_dialog as show_tailscale_config_dialog_ui
from modules.ui_yolo_options import show_yolo_options_window as show_yolo_options_window_ui

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
        # Modo desarrollo: centralizar archivos JSON en carpeta configs/
        config_dir = os.path.join(BASE_DIR, "configs")
        os.makedirs(config_dir, exist_ok=True)
        return config_dir

# Directorio para archivos de configuración (persistente)
CONFIG_DIR = get_config_dir()

# Rutas a archivos de configuración (se guardan en CONFIG_DIR, que es persistente)
CONFIG_FILE = os.path.join(CONFIG_DIR, "config_camara.json")
LANGUAGE_CONFIG_FILE = os.path.join(CONFIG_DIR, "language_config.json")
YOLO_MODELS_CONFIG = os.path.join(CONFIG_DIR, "yolo_models_config.json")
ADVANCED_INTERVALS_FILE = os.path.join(CONFIG_DIR, "tinysa_advanced_intervals.json")
TELEGRAM_CONFIG_FILE = os.path.join(CONFIG_DIR, "telegram_config.json")

# Inicializar runtime de idioma/sensibilidad persistidos
initialize_i18n(
    language_config_file=LANGUAGE_CONFIG_FILE,
    translations=TRANSLATIONS,
    default_language="es",
    default_audio_threshold=0.15,
)

# Rutas absolutas a los recursos (se cargan desde BASE_DIR, que está en el ejecutable)
# TAILSCALE_CONFIG_FILE eliminado - no guardamos credenciales por seguridad
def _resource_with_fallback(*relative_candidates):
    for rel in relative_candidates:
        candidate = os.path.join(BASE_DIR, rel)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(BASE_DIR, relative_candidates[0])


TAILSCALE_INSTALLER_WIN = _resource_with_fallback(
    os.path.join("installers", "tailscale-setup.exe"),
    "tailscale-setup.exe",
)
TAILSCALE_INSTALLER_LINUX = _resource_with_fallback(
    os.path.join("installers", "tailscale-installer.sh"),
    "tailscale-installer.sh",
)
AUDIO_MODEL_PATH = _resource_with_fallback(os.path.join("models", "drone_audio_model.h5"), "drone_audio_model.h5")
YOLO_DEFAULT_MODEL = _resource_with_fallback(os.path.join("models", "best.pt"), "best.pt")
AUDIO_MEAN_PATH = _resource_with_fallback(os.path.join("models", "audio_mean.npy"), "audio_mean.npy")
AUDIO_STD_PATH = _resource_with_fallback(os.path.join("models", "audio_std.npy"), "audio_std.npy")
SETTINGS_ICON_PATH = _resource_with_fallback(os.path.join("assets", "icons", "settings.png"), "settings.png")
MUTE_ICON_PATH = _resource_with_fallback(os.path.join("assets", "icons", "mute.png"), "mute.png")
VOL_ICON_PATH = _resource_with_fallback(os.path.join("assets", "icons", "vol.png"), "vol.png")

# Estado de modelos YOLO
yolo_model_path = YOLO_DEFAULT_MODEL
yolo_model_slots = []
yolo_default_slot = 0
yolo_options_thread = None
tailscale_options_thread = None
telegram_options_thread = None
language_options_thread = None

# --- VARIABLES GLOBALES UI ---
mouse_x, mouse_y = -1, -1
click_event_pos = None
mouse_is_down = False
pending_ip_change = None
ip_dialog_thread = None
adb_connected = False
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
    new_state = update_stream_endpoints_state_core(
        ip_with_port=ip_with_port,
        record_wifi=record_wifi,
        adb_target_ip=ADB_TARGET_IP,
        last_wifi_ip=last_wifi_ip,
    )
    ip_y_puerto = new_state["ip_y_puerto"]
    base_url = new_state["base_url"]
    video_url = new_state["video_url"]
    audio_url = new_state["audio_url"]
    last_wifi_ip = new_state["last_wifi_ip"]
    guardar_ip(ip_y_puerto)


def normalize_model_path(path):
    """Wrapper de normalización de rutas YOLO (yolo_models_config.py)."""
    return normalize_model_path_core(path, BASE_DIR)

def load_yolo_models_config():
    """Wrapper de carga de configuración YOLO (yolo_models_config.py)."""
    global yolo_model_slots, yolo_default_slot, yolo_model_path
    yolo_model_slots, yolo_default_slot, yolo_model_path = load_yolo_models_config_core(
        YOLO_MODELS_CONFIG,
        BASE_DIR,
    )


def save_yolo_models_config():
    """Wrapper de guardado de configuración YOLO (yolo_models_config.py)."""
    save_yolo_models_config_core(
        YOLO_MODELS_CONFIG,
        BASE_DIR,
        yolo_model_slots,
        yolo_default_slot,
    )


TELEGRAM_DEFAULT_CONFIG = {
    "enabled": False,
    "token": "",
    "chat_id": "",
    "cooldowns": {
        "yolo": 30.0,
        "rf": 30.0,
        "audio": 30.0,
    },
    "send_yolo_photo": True,
    "send_rf_image": True,
    "send_audio_clip": True,
}
telegram_config = copy.deepcopy(TELEGRAM_DEFAULT_CONFIG)


def _normalize_telegram_config(raw_config):
    merged = copy.deepcopy(TELEGRAM_DEFAULT_CONFIG)
    if not isinstance(raw_config, dict):
        return merged

    merged["enabled"] = bool(raw_config.get("enabled", merged["enabled"]))
    merged["token"] = str(raw_config.get("token", merged["token"])).strip()
    merged["chat_id"] = str(raw_config.get("chat_id", merged["chat_id"])).strip()
    merged["send_yolo_photo"] = bool(raw_config.get("send_yolo_photo", merged["send_yolo_photo"]))
    merged["send_rf_image"] = bool(raw_config.get("send_rf_image", merged["send_rf_image"]))
    merged["send_audio_clip"] = bool(raw_config.get("send_audio_clip", merged["send_audio_clip"]))

    cooldowns_raw = raw_config.get("cooldowns", {})
    if isinstance(cooldowns_raw, dict):
        for key, default_value in merged["cooldowns"].items():
            value = cooldowns_raw.get(key, default_value)
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = default_value
            if parsed < 0:
                parsed = 0.0
            merged["cooldowns"][key] = parsed

    return merged


def load_telegram_config():
    global telegram_config
    loaded = {}
    if os.path.exists(TELEGRAM_CONFIG_FILE):
        try:
            with open(TELEGRAM_CONFIG_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception as exc:
            print(f"{t('telegram_config_load_error')} ({exc})")
    telegram_config = _normalize_telegram_config(loaded)
    save_telegram_config()
    return telegram_config


def save_telegram_config():
    try:
        with open(TELEGRAM_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(telegram_config, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"{t('telegram_config_save_error')} ({exc})")


def get_telegram_cooldowns():
    return telegram_config.get("cooldowns", {}).copy()


load_yolo_models_config()
load_telegram_config()
telegram_notifier = TelegramNotifier(
    enabled=telegram_config.get("enabled", False),
    token=telegram_config.get("token", ""),
    chat_id=telegram_config.get("chat_id", ""),
    cooldowns=get_telegram_cooldowns(),
)


def refresh_telegram_notifier_settings():
    telegram_notifier.update_settings(
        enabled=telegram_config.get("enabled", False),
        token=telegram_config.get("token", ""),
        chat_id=telegram_config.get("chat_id", ""),
        cooldowns=get_telegram_cooldowns(),
    )


def get_telegram_ui_config():
    return copy.deepcopy(telegram_config)


def save_telegram_ui_config(new_config):
    global telegram_config
    telegram_config = _normalize_telegram_config(new_config)
    save_telegram_config()
    refresh_telegram_notifier_settings()
    return True


def detect_telegram_chat_id(token):
    token = str(token or "").strip()
    if not token:
        return None, t("telegram_chat_id_missing_token")

    try:
        response = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok", False):
            return None, payload.get("description", t("telegram_chat_id_detect_error"))

        results = payload.get("result", [])
        for update in reversed(results):
            message = update.get("message") or update.get("edited_message") or update.get("channel_post")
            if not isinstance(message, dict):
                continue
            chat = message.get("chat", {})
            chat_id = chat.get("id")
            if chat_id is not None:
                return str(chat_id), None

        return None, t("telegram_chat_id_not_found")
    except Exception as exc:
        return None, f"{t('telegram_chat_id_detect_error')}: {exc}"



def get_yolo_settings_icon():
    """Wrapper del icono de ajustes (implementación en ui_assets.py)."""
    global yolo_settings_icon
    yolo_settings_icon = get_yolo_settings_icon_core(SETTINGS_ICON_PATH, yolo_settings_icon)
    return yolo_settings_icon

def get_audio_volume_icon(muted=True):
    """Wrapper de iconos de volumen (implementación en ui_assets.py)."""
    global mute_icon, vol_icon
    mute_icon, vol_icon, icon = get_audio_volume_icon_core(
        muted=muted,
        mute_icon=mute_icon,
        vol_icon=vol_icon,
        mute_icon_path=MUTE_ICON_PATH,
        vol_icon_path=VOL_ICON_PATH,
    )
    return icon

def cargar_ip():
    """Wrapper de carga de IP (connection_runtime.py)."""
    return cargar_ip_core(CONFIG_FILE)

def guardar_ip(ip):
    """Wrapper de guardado de IP (connection_runtime.py)."""
    return guardar_ip_core(CONFIG_FILE, ip)

# --- FUNCIONES TAILSCALE ---
tailscale_running = False

# NOTA: No guardamos credenciales de Tailscale porque:
# 1. Tailscale maneja la autenticación de forma persistente después del primer login
# 2. El usuario se autentica mediante OAuth en el navegador cuando ejecuta 'tailscale up'
# 3. Windows/Linux guardan la sesión automáticamente
# 4. Guardar credenciales en texto plano es un riesgo de seguridad

def verificar_estado_tailscale():
    """Wrapper de verificación Tailscale (tailscale_runtime.py)."""
    global tailscale_running
    tailscale_running = verificar_estado_tailscale_core(subprocess)
    return tailscale_running

def get_tailscale_username():
    """Wrapper de username Tailscale (tailscale_runtime.py)."""
    return get_tailscale_username_core(subprocess)

def get_tailscale_ip():
    """Wrapper de IP Tailscale (tailscale_runtime.py)."""
    return get_tailscale_ip_core(subprocess)

def get_tailscale_connected_devices():
    """Wrapper de dispositivos Tailscale (tailscale_runtime.py)."""
    return get_tailscale_connected_devices_core(subprocess)

def get_tailscale_path():
    """Wrapper de ruta de Tailscale (tailscale_runtime.py)."""
    return get_tailscale_path_core()

def tailscale_installed():
    """Wrapper de verificación de instalación Tailscale (tailscale_runtime.py)."""
    return tailscale_installed_core(subprocess)

def install_tailscale():
    """Wrapper de instalación Tailscale (tailscale_control.py)."""
    return install_tailscale_core(
        t_func=t,
        tailscale_installer_win=TAILSCALE_INSTALLER_WIN,
        tailscale_installer_linux=TAILSCALE_INSTALLER_LINUX,
        tailscale_installed_fn=tailscale_installed,
    )

def toggle_tailscale():
    """Wrapper de activación/desactivación Tailscale (tailscale_control.py)."""
    def _get_running():
        return tailscale_running

    def _set_running(value):
        global tailscale_running
        tailscale_running = value

    toggle_tailscale_core(
        t_func=t,
        get_running_fn=_get_running,
        set_running_fn=_set_running,
        tailscale_installed_fn=tailscale_installed,
        get_tailscale_path_fn=get_tailscale_path,
    )

# Runtime de idioma/sensibilidad gestionado en i18n_runtime.py

def show_language_selection_dialog():
    """Wrapper del diálogo de idioma (implementación en ui_language_options.py)."""
    return show_language_selection_dialog_ui(
        base_dir=BASE_DIR,
        t_func=t,
        translate_for_language_fn=translate_for_language,
        get_current_language_fn=get_current_language,
        get_audio_confidence_threshold_fn=get_audio_confidence_threshold,
        guardar_idioma_fn=guardar_idioma,
        guardar_audio_threshold_fn=guardar_audio_threshold,
    )


def show_telegram_options_dialog():
    """Wrapper del diálogo de Telegram (implementación en ui_telegram_options.py)."""
    return show_telegram_options_dialog_ui(
        base_dir=BASE_DIR,
        t_func=t,
        get_telegram_config_fn=get_telegram_ui_config,
        save_telegram_config_fn=save_telegram_ui_config,
        detect_telegram_chat_id_fn=detect_telegram_chat_id,
    )

def draw_tailscale_indicator(frame, mouse_pos, click_pos):
    """Wrapper del indicador Tailscale (implementación en ui_indicators.py)."""
    return draw_tailscale_indicator_ui(
        frame=frame,
        mouse_pos=mouse_pos,
        click_pos=click_pos,
        tailscale_running=tailscale_running,
        t_func=t,
    )

def draw_tailscale_settings_icon(frame, mouse_pos, click_pos):
    """Wrapper del icono de ajustes Tailscale (implementación en ui_indicators.py)."""
    return draw_tailscale_settings_icon_ui(
        frame=frame,
        mouse_pos=mouse_pos,
        click_pos=click_pos,
        icon=get_yolo_settings_icon(),
    )

def show_tailscale_config_dialog():
    """Wrapper del diálogo de Tailscale (implementación en ui_tailscale_options.py)."""
    return show_tailscale_config_dialog_ui(
        t_func=t,
        tailscale_installed_fn=tailscale_installed,
        tailscale_installer_win=TAILSCALE_INSTALLER_WIN,
        tailscale_installer_linux=TAILSCALE_INSTALLER_LINUX,
        install_tailscale_fn=install_tailscale,
        get_tailscale_username_fn=get_tailscale_username,
        get_tailscale_ip_fn=get_tailscale_ip,
        get_tailscale_connected_devices_fn=get_tailscale_connected_devices,
    )

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

def open_language_options_dialog():
    """Abre la ventana de idioma en un hilo aparte (evita duplicados)."""
    global language_options_thread
    if language_options_thread and language_options_thread.is_alive():
        return

    def runner():
        global language_options_thread
        try:
            show_language_selection_dialog()
        finally:
            language_options_thread = None

    language_options_thread = threading.Thread(target=runner, daemon=True)
    language_options_thread.start()


def open_telegram_options_dialog():
    """Abre la ventana de Telegram en un hilo aparte (evita duplicados)."""
    global telegram_options_thread
    if telegram_options_thread and telegram_options_thread.is_alive():
        return

    def runner():
        global telegram_options_thread
        try:
            show_telegram_options_dialog()
        finally:
            telegram_options_thread = None

    telegram_options_thread = threading.Thread(target=runner, daemon=True)
    telegram_options_thread.start()

def draw_language_indicator(frame, mouse_pos, click_pos):
    """Wrapper del indicador de idioma (implementación en ui_indicators.py)."""
    return draw_language_indicator_ui(
        frame=frame,
        mouse_pos=mouse_pos,
        click_pos=click_pos,
        t_func=t,
    )


def draw_telegram_indicator(frame, mouse_pos, click_pos):
    """Wrapper del indicador de Telegram (implementación en ui_indicators.py)."""
    return draw_telegram_indicator_ui(
        frame=frame,
        mouse_pos=mouse_pos,
        click_pos=click_pos,
        t_func=t,
    )

def solicitar_nueva_ip(ip_actual):
    """Wrapper del diálogo de IP (implementación en ui_helpers.py)."""
    return solicitar_nueva_ip_ui(ip_actual=ip_actual, t_func=t)

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
video_connection_manager = VideoConnectionManager()
windows_cursor_fixed = False
windows_cursor_warning = False

# --- CONFIGURACIÓN DE AUDIO ---
CHUNK = 1024
p = None
pyaudio_init_lock = threading.Lock()
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
audio_recent_chunks = deque()
audio_recent_lock = threading.Lock()
audio_recent_total_bytes = 0
audio_stream_sample_rate = 44100
audio_stream_channels = 1
audio_recent_max_seconds = 6
audio_detection_result = {"is_drone": False, "confidence": 0.0}
audio_detection_lock = threading.Lock()
# Sistema de alerta persistente
audio_detection_alert_time = None  # Timestamp de la última detección que superó el umbral
audio_detection_max_confidence = 0.0  # Máximo porcentaje alcanzado durante la alerta actual
AUDIO_ALERT_DURATION = 30  # Duración de la alerta en segundos

AUDIO_SAMPLE_RATE = 22050
AUDIO_DURATION = 2  # Segundos (entero para evitar errores de slice)
AUDIO_VISUAL_MULTIPLIER = 3  # Multiplicador visual para mostrar porcentajes más altos (hasta 100% máximo)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048


def get_pyaudio_instance():
    """Inicializa PyAudio bajo demanda, minimizando ruido ALSA/JACK en Linux."""
    global p
    if p is not None:
        return p

    with pyaudio_init_lock:
        if p is not None:
            return p
        if os.name == "posix":
            # PyAudio/PortAudio enumera dispositivos al iniciar y ALSA puede imprimir
            # avisos ruidosos en stderr. Silenciamos SOLO durante esta inicialización.
            stderr_fd = None
            stderr_backup = None
            devnull_fd = None
            try:
                stderr_fd = sys.stderr.fileno()
                stderr_backup = os.dup(stderr_fd)
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, stderr_fd)
                p = pyaudio.PyAudio()
            except Exception:
                p = pyaudio.PyAudio()
            finally:
                if stderr_fd is not None and stderr_backup is not None:
                    try:
                        os.dup2(stderr_backup, stderr_fd)
                    except Exception:
                        pass
                if stderr_backup is not None:
                    try:
                        os.close(stderr_backup)
                    except Exception:
                        pass
                if devnull_fd is not None:
                    try:
                        os.close(devnull_fd)
                    except Exception:
                        pass
        else:
            p = pyaudio.PyAudio()
    return p


def _get_telegram_temp_dir():
    temp_dir = os.path.join(tempfile.gettempdir(), "adas3_telegram")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def _append_audio_recent_chunk(chunk):
    global audio_recent_total_bytes
    if not chunk:
        return

    with audio_recent_lock:
        audio_recent_chunks.append(chunk)
        audio_recent_total_bytes += len(chunk)

        max_bytes = int(
            max(audio_stream_sample_rate, 8000)
            * max(audio_stream_channels, 1)
            * 2
            * max(audio_recent_max_seconds, 1)
        )
        while audio_recent_total_bytes > max_bytes and audio_recent_chunks:
            removed = audio_recent_chunks.popleft()
            audio_recent_total_bytes -= len(removed)


def _build_telegram_message(event_type, *, timestamp, confidence=None, frequency_hz=None):
    hour_text = time.strftime("%H:%M:%S", time.localtime(timestamp))

    if event_type == "yolo":
        if confidence is None:
            return t("telegram_yolo_message", hour_text)
        return t("telegram_yolo_message_conf", hour_text, int(confidence * 100))

    if event_type == "rf":
        if frequency_hz:
            return t("telegram_rf_message_freq", hour_text, frequency_hz / 1e6, int((confidence or 0.0) * 100))
        return t("telegram_rf_message", hour_text, int((confidence or 0.0) * 100))

    return t("telegram_audio_message", hour_text, int((confidence or 0.0) * 100))


def _save_frame_for_telegram(frame, event_type):
    if frame is None:
        return None
    temp_dir = _get_telegram_temp_dir()
    timestamp_ms = int(time.time() * 1000)
    output_path = os.path.join(temp_dir, f"{event_type}_{timestamp_ms}.png")
    success = cv2.imwrite(output_path, frame)
    if not success:
        return None
    return output_path


def _save_rf_image_for_telegram():
    with tinysa_render_lock:
        rf_image = None if tinysa_image_ready is None else tinysa_image_ready.copy()

    if rf_image is None:
        return None

    try:
        bgr_image = cv2.cvtColor(rf_image, cv2.COLOR_RGBA2BGR)
    except Exception:
        return None

    return _save_frame_for_telegram(bgr_image, "rf")


def _save_audio_clip_for_telegram(clip_seconds=5):
    with audio_recent_lock:
        if not audio_recent_chunks:
            return None
        raw_audio = b"".join(audio_recent_chunks)
        sample_rate = int(max(audio_stream_sample_rate, 8000))
        channels = int(max(audio_stream_channels, 1))

    max_bytes = int(sample_rate * channels * 2 * max(clip_seconds, 1))
    if len(raw_audio) > max_bytes:
        raw_audio = raw_audio[-max_bytes:]

    if not raw_audio:
        return None

    temp_dir = _get_telegram_temp_dir()
    timestamp_ms = int(time.time() * 1000)
    output_path = os.path.join(temp_dir, f"audio_{timestamp_ms}.wav")
    try:
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio)
    except Exception as exc:
        print(f"[TELEGRAM] No se pudo crear clip WAV: {exc}")
        return None

    return output_path


def enqueue_telegram_notification(
    event_type,
    *,
    timestamp,
    confidence=None,
    frequency_hz=None,
    frame_path=None,
    audio_path=None,
):
    text = _build_telegram_message(
        event_type,
        timestamp=timestamp,
        confidence=confidence,
        frequency_hz=frequency_hz,
    )
    event = TelegramEvent(
        event_type=event_type,
        text=text,
        photo_path=frame_path,
        audio_path=audio_path,
        remove_after_send=True,
    )
    telegram_notifier.enqueue(event)

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


def show_tinysa_menu():
    """Wrapper del menú TinySA (implementación en ui_tinysa_options.py)."""
    global last_advanced_intervals
    result, loaded_intervals = show_tinysa_menu_ui(
        t_func=t,
        advanced_intervals_file=ADVANCED_INTERVALS_FILE,
        default_sweeps=TIN_YSA_SWEEPS_PER_RANGE,
        last_advanced_intervals=last_advanced_intervals,
    )
    last_advanced_intervals = loaded_intervals
    return result

# --- FUNCIONES TINYSA HARDWARE ---

def find_tinysa_port():
    """Busca el puerto COM del TinySA Ultra."""
    return find_tinysa_port_engine(serial.tools.list_ports)


def send_tinysa_command(command_json):
    """Envía un comando JSON al servidor Android para controlar TinySA."""
    return send_tinysa_command_engine(base_url, command_json, timeout=10)

def tinysa_hardware_worker_serial():
    """Wrapper del worker serial TinySA (implementación en tinysa_hardware_engine.py)."""
    global tinysa_data_ready, tinysa_sequence_index, tinysa_current_label

    def _get_running():
        return tinysa_running

    def _get_sequence():
        return tinysa_sequence

    def _get_sequence_index():
        return tinysa_sequence_index

    def _set_sequence_index(new_idx):
        global tinysa_sequence_index
        tinysa_sequence_index = new_idx

    def _set_current_label(label):
        global tinysa_current_label
        tinysa_current_label = label

    def _get_serial():
        return tinysa_serial

    def _set_data_ready(freqs_dynamic, levels):
        global tinysa_data_ready
        with tinysa_data_lock:
            tinysa_data_ready = (freqs_dynamic, levels)

    run_tinysa_hardware_worker_serial(
        get_running_fn=_get_running,
        get_sequence_fn=_get_sequence,
        get_sequence_index_fn=_get_sequence_index,
        set_sequence_index_fn=_set_sequence_index,
        set_current_label_fn=_set_current_label,
        get_serial_fn=_get_serial,
        set_data_ready_fn=_set_data_ready,
        tinysa_points_default=TINYSA_POINTS,
        sweeps_per_range_default=TIN_YSA_SWEEPS_PER_RANGE,
    )

def tinysa_hardware_worker():
    """Wrapper del worker HTTP TinySA (implementación en tinysa_hardware_engine.py)."""
    global tinysa_data_ready, tinysa_http_response, tinysa_current_label

    def _get_running():
        return tinysa_running

    def _set_running(value):
        global tinysa_running
        tinysa_running = value

    def _get_use_http():
        return tinysa_use_http

    def _set_use_http(value):
        global tinysa_use_http
        tinysa_use_http = value

    def _get_last_sequence_payload():
        return tinysa_last_sequence_payload

    def _send_command(command_json):
        return send_tinysa_command(command_json)

    def _get_http_response():
        return tinysa_http_response

    def _set_http_response(response):
        global tinysa_http_response
        tinysa_http_response = response

    def _set_data_ready(freqs, levels):
        global tinysa_data_ready
        with tinysa_data_lock:
            tinysa_data_ready = (freqs, levels)

    def _set_current_label(label):
        global tinysa_current_label
        tinysa_current_label = label

    run_tinysa_hardware_worker_http(
        get_running_fn=_get_running,
        set_running_fn=_set_running,
        get_use_http_fn=_get_use_http,
        set_use_http_fn=_set_use_http,
        get_last_sequence_payload_fn=_get_last_sequence_payload,
        send_command_fn=_send_command,
        base_url=base_url,
        headers=headers,
        connect_timeout=TINYSA_HTTP_CONNECT_TIMEOUT,
        read_timeout=TINYSA_HTTP_READ_TIMEOUT,
        stream_chunk_size=TINYSA_STREAM_CHUNK_SIZE,
        get_http_response_fn=_get_http_response,
        set_http_response_fn=_set_http_response,
        set_data_ready_fn=_set_data_ready,
        set_current_label_fn=_set_current_label,
    )

def detect_drone_rf(freqs, levels):
    """Wrapper local de detección RF (implementación en rf_detection.py)."""
    global rf_drone_detection_history

    with rf_detection_params_lock:
        peak_threshold = rf_peak_threshold
        min_peak_height_db = rf_min_peak_height_db
        min_peak_width_mhz = rf_min_peak_width_mhz
        max_peak_width_mhz = rf_max_peak_width_mhz

    detection, new_history = detect_drone_rf_core(
        freqs=freqs,
        levels=levels,
        rf_history=rf_drone_detection_history,
        peak_threshold=peak_threshold,
        min_peak_height_db=min_peak_height_db,
        min_peak_width_mhz=min_peak_width_mhz,
        max_peak_width_mhz=max_peak_width_mhz,
    )
    rf_drone_detection_history = new_history
    return detection

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
                    tinysa_last_sequence_payload = copy.deepcopy(sequence_json)
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
    """Wrapper local de carga de modelo (implementación en audio_detection_engine.py)."""
    global audio_model, audio_mean, audio_std
    ok, loaded_model, loaded_mean, loaded_std = load_audio_model(
        audio_model_path=AUDIO_MODEL_PATH,
        audio_mean_path=AUDIO_MEAN_PATH,
        audio_std_path=AUDIO_STD_PATH,
    )
    if ok:
        audio_model = loaded_model
        audio_mean = loaded_mean
        audio_std = loaded_std
    return ok

def extract_features_realtime(audio_chunk):
    """Wrapper local para extracción de features (implementación en audio_features.py)."""
    return extract_audio_features(
        audio_chunk=audio_chunk,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_duration=AUDIO_DURATION,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        audio_mean=audio_mean,
        audio_std=audio_std,
    )

def audio_detection_worker():
    """Wrapper local del worker (implementación en audio_detection_engine.py)."""
    global audio_detection_result, audio_detection_alert_time, audio_detection_max_confidence

    def _is_enabled():
        return audio_detection_enabled

    def _get_audio_model():
        return audio_model

    def _get_alert_state():
        return audio_detection_alert_time, audio_detection_max_confidence

    def _set_alert_state(alert_time, max_confidence):
        global audio_detection_alert_time, audio_detection_max_confidence
        audio_detection_alert_time = alert_time
        audio_detection_max_confidence = max_confidence

    def _set_detection_result(is_drone, confidence):
        global audio_detection_result
        with audio_detection_lock:
            audio_detection_result = {
                "is_drone": is_drone,
                "confidence": confidence,
            }

    def _on_detection_event(event_data):
        timestamp = event_data.get("timestamp", time.time())
        confidence = float(event_data.get("visual_confidence", 0.0))
        audio_path = None
        if telegram_config.get("send_audio_clip", True):
            audio_path = _save_audio_clip_for_telegram(clip_seconds=5)
        enqueue_telegram_notification(
            "audio",
            timestamp=timestamp,
            confidence=confidence,
            audio_path=audio_path,
        )

    run_audio_detection_worker(
        is_detection_enabled_fn=_is_enabled,
        audio_buffer=audio_buffer,
        audio_duration_seconds=AUDIO_DURATION,
        extract_features_fn=extract_features_realtime,
        get_audio_model_fn=_get_audio_model,
        get_threshold_fn=get_audio_confidence_threshold,
        visual_multiplier=AUDIO_VISUAL_MULTIPLIER,
        alert_duration_seconds=AUDIO_ALERT_DURATION,
        get_alert_state_fn=_get_alert_state,
        set_alert_state_fn=_set_alert_state,
        set_detection_result_fn=_set_detection_result,
        on_detection_event_fn=_on_detection_event,
    )

def toggle_audio_detection():
    """Activa/desactiva la detección de audio."""
    global audio_detection_enabled, audio_detection_thread
    
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
        print("Detección activada")
    else:
        # Apagar detección
        audio_detection_enabled = False
        if audio_detection_thread:
            audio_detection_thread.join(timeout=2)
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
    """Wrapper local de carga (implementación en yolo_engine.py)."""
    global yolo_model
    ok, model = load_yolo_model(yolo_model_path)
    if ok:
        yolo_model = model
    return ok

def yolo_inference_worker():
    """Wrapper local del worker (implementación en yolo_engine.py)."""
    global ultimo_resultado_yolo

    def _is_running():
        return yolo_worker_running

    def _get_model():
        return yolo_model

    def _get_thresholds():
        with yolo_threshold_lock:
            return yolo_conf_threshold, yolo_iou_threshold

    def _set_result(frame_original, detecciones, boxes_data):
        global ultimo_resultado_yolo
        with yolo_result_lock:
            ultimo_resultado_yolo = {
                "frame": frame_original,
                "detecciones": detecciones,
                "boxes_data": boxes_data,
            }

    run_yolo_inference_worker(
        is_running_fn=_is_running,
        frame_queue=yolo_frame_queue,
        get_model_fn=_get_model,
        get_thresholds_fn=_get_thresholds,
        yolo_scale=YOLO_SCALE,
        set_result_fn=_set_result,
    )

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
    
    # Limpiar cola de frames pendientes
    clear_queue_safely(yolo_frame_queue)
    
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
    """Wrapper local de dibujo (implementación en yolo_engine.py)."""
    return draw_yolo_detections(frame, boxes_data)

# La gestión de reconexión/captura de video vive en video_connection.py


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

# --- AUDIO STREAMING ---
def stream_audio():
    global audio_stream, stop_audio_thread, audio_stream_sample_rate, audio_stream_channels
    
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
                audio_stream_sample_rate = parsed_sample_rate
                audio_stream_channels = parsed_channels

                print(f"[AUDIO] Stream configurado: {parsed_channels} canal(es) @ {parsed_sample_rate} Hz")

                # El servidor Android envía PCM crudo directamente, sin header WAV
                # Inicializar PyAudio antes de leer datos
                pa = get_pyaudio_instance()
                audio_stream = pa.open(format=pyaudio.paInt16,
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
                            _append_audio_recent_chunk(chunk)
                                    
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
    return cambiar_ip_camara_core(
        cap_actual=cap_actual,
        nueva_ip=nueva_ip,
        audio_enabled=audio_enabled,
        stop_audio_fn=stop_audio,
        ask_new_ip_fn=solicitar_nueva_ip,
        current_ip=ip_y_puerto,
        update_stream_endpoints_fn=update_stream_endpoints,
        schedule_video_connection_fn=lambda force: video_connection_manager.schedule(video_url, force=force),
    )

# --- INDICADORES ---
def draw_interactive_button(frame, text, x_start, y_center, w, h, text_color, mouse_pos, click_pos, align_right=False):
    return draw_interactive_button_ui(
        frame,
        text,
        x_start,
        y_center,
        w,
        h,
        text_color,
        mouse_pos,
        click_pos,
        align_right=align_right,
    )


def draw_yolo_indicator(frame, mouse_pos, click_pos, detecciones=0):
    return draw_yolo_indicator_ui(frame, mouse_pos, click_pos, yolo_enabled, detecciones, t)

def draw_yolo_settings_icon(frame, mouse_pos, click_pos):
    """Wrapper del icono de ajustes YOLO (implementación en ui_indicators.py)."""
    return draw_yolo_settings_icon_ui(
        frame=frame,
        mouse_pos=mouse_pos,
        click_pos=click_pos,
        icon=get_yolo_settings_icon(),
    )

def draw_tinysa_indicator(frame, mouse_pos, click_pos):
    return draw_tinysa_indicator_ui(frame, mouse_pos, click_pos, tinysa_running, t)

def draw_tinysa_settings_icon(frame, mouse_pos, click_pos):
    """Wrapper del icono de ajustes TinySA (implementación en ui_indicators.py)."""
    return draw_tinysa_settings_icon_ui(
        frame=frame,
        mouse_pos=mouse_pos,
        click_pos=click_pos,
        icon=get_yolo_settings_icon(),
    )

def draw_audio_volume_icon(frame, mouse_pos, click_pos):
    """Wrapper del icono de volumen (implementación en ui_indicators.py)."""
    return draw_audio_volume_icon_ui(
        frame=frame,
        mouse_pos=mouse_pos,
        click_pos=click_pos,
        icon=get_audio_volume_icon(muted=(not audio_enabled) or audio_playback_muted),
    )

def draw_audio_detection_toggle(frame, mouse_pos, click_pos):
    return draw_audio_detection_toggle_ui(frame, mouse_pos, click_pos, audio_detection_enabled, t)


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
    return draw_ip_indicator_ui(frame, ip_y_puerto, t)


def draw_ip_settings_icon(frame, mouse_pos, click_pos):
    """Wrapper del icono de ajustes IP (implementación en ui_indicators.py)."""
    return draw_ip_settings_icon_ui(
        frame=frame,
        mouse_pos=mouse_pos,
        click_pos=click_pos,
        icon=get_yolo_settings_icon(),
        ip_text=f"IP: {ip_y_puerto}",
    )


def open_ip_change_dialog():
    global ip_dialog_thread
    def _get_current_ip():
        return ip_y_puerto

    def _set_pending_ip(value):
        global pending_ip_change
        pending_ip_change = value

    def _clear_thread():
        global ip_dialog_thread
        ip_dialog_thread = None

    ip_dialog_thread = open_ip_change_dialog_core(
        current_thread=ip_dialog_thread,
        get_current_ip_fn=_get_current_ip,
        ask_new_ip_fn=solicitar_nueva_ip,
        set_pending_ip_fn=_set_pending_ip,
        clear_thread_fn=_clear_thread,
    )


def apply_pending_ip_change(cap_actual):
    global pending_ip_change
    cap_actual, pending_ip_change = apply_pending_ip_change_core(
        pending_ip=pending_ip_change,
        cap_actual=cap_actual,
        cambiar_ip_fn=cambiar_ip_camara,
    )
    return cap_actual


def poll_adb_connection():
    global last_adb_check, adb_connected, pending_ip_change, last_wifi_ip
    new_state = poll_adb_connection_core(
        last_adb_check=last_adb_check,
        adb_check_interval=ADB_CHECK_INTERVAL,
        adb_connected=adb_connected,
        pending_ip_change=pending_ip_change,
        last_wifi_ip=last_wifi_ip,
        current_ip=ip_y_puerto,
        adb_target_ip=ADB_TARGET_IP,
        subprocess_module=subprocess,
        shutil_module=shutil,
        time_module=time,
    )
    last_adb_check = new_state["last_adb_check"]
    adb_connected = new_state["adb_connected"]
    pending_ip_change = new_state["pending_ip_change"]
    last_wifi_ip = new_state["last_wifi_ip"]
    for msg in new_state["messages"]:
        print(msg)


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
    """Wrapper de la ventana de modelos YOLO (implementación en ui_yolo_options.py)."""
    show_yolo_options_window_ui(
        yolo_model_slots=yolo_model_slots,
        yolo_default_slot=yolo_default_slot,
        yolo_default_model_path=YOLO_DEFAULT_MODEL,
        translate_fn=t,
        normalize_model_path_fn=normalize_model_path,
        apply_yolo_model_fn=apply_yolo_model,
    )


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
    # El panel y la línea del track se componen en un solo overlay para reducir copias.

    # Textos
    text_y = panel_y1 + 18
    cv2.putText(overlay, label, (x + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)
    cv2.putText(overlay, f"{value:.2f}", (x + width - 55, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)

    slider_offset = 10
    panel_y1 += slider_offset

    # Slider track semitransparente
    track_x1 = x + 20
    track_x2 = x + width - 20
    track_y = panel_y1 + height - 25
    cv2.line(overlay, (track_x1, track_y), (track_x2, track_y), (210, 210, 210), 6, cv2.LINE_AA)

    # Handle
    ratio = (value - min_val) / (max_val - min_val)
    ratio = max(0.0, min(1.0, ratio))
    handle_x = int(track_x1 + ratio * (track_x2 - track_x1))
    cv2.circle(overlay, (handle_x, track_y), 10, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, (handle_x, track_y), 10, (0, 102, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

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
    return draw_audio_detection_indicator_ui(
        frame=frame,
        audio_detection_enabled=audio_detection_enabled,
        audio_detection_lock=audio_detection_lock,
        audio_detection_result=audio_detection_result,
        audio_detection_alert_time=audio_detection_alert_time,
        audio_detection_max_confidence=audio_detection_max_confidence,
        audio_visual_multiplier=AUDIO_VISUAL_MULTIPLIER,
        t_func=t,
    )


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
yolo_prev_detected = False
rf_prev_detected = False

video_connection_manager.schedule(video_url, force=True)
refresh_telegram_notifier_settings()
telegram_notifier.start()
if telegram_config.get("enabled", False):
    print(t("telegram_enabled_start"))
else:
    print(t("telegram_disabled_start"))

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

    cap, new_cap_ready = video_connection_manager.process_pending(cap, video_url)
    if new_cap_ready:
        fps_start_time = time.time()
        last_reconnect_try = time.time()

    if cap is None:
        yolo_prev_detected = False
        rf_prev_detected = False
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
        frame_negro = draw_adb_message(frame_negro, t, adb_connected)
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
            show_warning_async(t, 'no_streaming', 'no_streaming_yolo')
            current_click = None
        frame_negro, yolo_settings_clicked = draw_yolo_settings_icon(frame_negro, current_mouse, current_click)
        if yolo_settings_clicked:
            open_yolo_options_dialog()
            current_click = None
        
        # Icono de volumen de audio
        frame_negro, volume_icon_clicked = draw_audio_volume_icon(frame_negro, current_mouse, current_click)
        if volume_icon_clicked:
            if cap is None:
                show_warning_async(t, 'no_streaming', 'no_streaming')
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

        # Bot Telegram
        frame_negro, telegram_clicked = draw_telegram_indicator(frame_negro, current_mouse, current_click)
        if telegram_clicked:
            open_telegram_options_dialog()
            current_click = None
        
        # Idioma APP
        frame_negro, language_clicked = draw_language_indicator(frame_negro, current_mouse, current_click)
        if language_clicked:
            open_language_options_dialog()
            current_click = None

        process_pending_yolo_reload()
        cap = apply_pending_ip_change(cap)
        frame_negro = draw_tinysa_message(
            frame_negro,
            t,
            tinysa_detected,
            tinysa_use_http,
            rf_drone_detection_lock,
            rf_drone_detection_result,
            rf_drone_detection_enabled,
        )
        
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
            video_connection_manager.schedule(video_url)
        
    else:
        # MODO CONECTADO
        ret, frame, current_id = cap.read()
        
        if not ret:
            print("Señal perdida. Cerrando captura...")
            cap.release()
            cap = None
            last_reconnect_try = time.time()
            video_connection_manager.schedule(video_url, force=True)
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
        yolo_detected = bool(resultado_yolo["boxes_data"])
        if resultado_yolo["boxes_data"]:
            # Las detecciones ya están en el tamaño correcto (1280x720) porque YOLO procesó el frame escalado
            frame = dibujar_detecciones_yolo(frame, resultado_yolo["boxes_data"])
            detecciones_count = resultado_yolo["detecciones"]
        else:
            detecciones_count = 0
        
        # Renderizado capas
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
                show_warning_async(t, 'no_streaming', 'no_streaming')
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
                show_warning_async(t, 'no_streaming', 'no_streaming_yolo')
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

        # 6. Bot Telegram
        frame, telegram_clicked = draw_telegram_indicator(frame, current_mouse, current_click)
        if telegram_clicked:
            open_telegram_options_dialog()
            current_click = None

        # 7. Idioma APP
        frame, language_clicked = draw_language_indicator(frame, current_mouse, current_click)
        if language_clicked:
            open_language_options_dialog()
            current_click = None

        # 8. IP
        frame, _ = draw_ip_indicator(frame, current_mouse, current_click)
        frame = draw_adb_message(frame, t, adb_connected)
        frame, ip_settings_clicked = draw_ip_settings_icon(frame, current_mouse, current_click)
        if ip_settings_clicked:
            open_ip_change_dialog()
            current_click = None

        process_pending_yolo_reload()
        cap = apply_pending_ip_change(cap)
        frame = draw_tinysa_message(
            frame,
            t,
            tinysa_detected,
            tinysa_use_http,
            rf_drone_detection_lock,
            rf_drone_detection_result,
            rf_drone_detection_enabled,
        )

        frame = draw_audio_detection_indicator(frame)
        frame = draw_fps_indicator(frame, current_fps, t)

        if yolo_detected and not yolo_prev_detected:
            best_confidence = max((box.get("conf", 0.0) for box in resultado_yolo["boxes_data"]), default=0.0)
            yolo_photo_path = None
            if telegram_config.get("send_yolo_photo", True):
                yolo_photo_path = _save_frame_for_telegram(frame, "yolo")
            enqueue_telegram_notification(
                "yolo",
                timestamp=time.time(),
                confidence=best_confidence,
                frame_path=yolo_photo_path,
            )
        yolo_prev_detected = yolo_detected

        with rf_drone_detection_lock:
            rf_snapshot = rf_drone_detection_result.copy()
        rf_detected = bool(rf_snapshot.get("is_drone", False)) and rf_drone_detection_enabled
        if rf_detected and not rf_prev_detected:
            rf_photo_path = None
            if telegram_config.get("send_rf_image", True):
                rf_photo_path = _save_rf_image_for_telegram()
            enqueue_telegram_notification(
                "rf",
                timestamp=time.time(),
                confidence=rf_snapshot.get("confidence", 0.0),
                frequency_hz=rf_snapshot.get("frequency"),
                frame_path=rf_photo_path,
            )
        rf_prev_detected = rf_detected
        
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
                show_warning_async(t, 'no_streaming', 'no_streaming_yolo')
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

if p is not None:
    p.terminate()
telegram_notifier.stop()
cv2.destroyAllWindows()
print("Programa finalizado.")
