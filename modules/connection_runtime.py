import json
import os
import threading


def cargar_ip(config_file, default_ip="192.168.1.129:8080"):
    """Carga la última IP guardada o devuelve la predeterminada."""
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("ip", default_ip)
        except Exception:
            pass
    return default_ip


def guardar_ip(config_file, ip):
    """Guarda la IP en el archivo de configuración."""
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"ip": ip}, f)
        return True
    except Exception as e:
        print(f"Error al guardar IP: {e}")
        return False


def compute_stream_endpoints(ip_with_port):
    """Calcula base_url, video_url y audio_url para la IP dada."""
    base_url = f"http://{ip_with_port}"
    return base_url, base_url + "/video", base_url + "/audio"


def update_stream_endpoints_state(
    *,
    ip_with_port,
    record_wifi,
    adb_target_ip,
    last_wifi_ip,
):
    """
    Calcula el nuevo estado de endpoints sin tocar globales.
    """
    if record_wifi and ip_with_port != adb_target_ip:
        last_wifi_ip = ip_with_port
    base_url, video_url, audio_url = compute_stream_endpoints(ip_with_port)
    return {
        "ip_y_puerto": ip_with_port,
        "base_url": base_url,
        "video_url": video_url,
        "audio_url": audio_url,
        "last_wifi_ip": last_wifi_ip,
    }


def cambiar_ip_camara(
    *,
    cap_actual,
    nueva_ip,
    audio_enabled,
    stop_audio_fn,
    ask_new_ip_fn,
    current_ip,
    update_stream_endpoints_fn,
    schedule_video_connection_fn,
):
    """Gestiona cambio de IP de cámara y reinicio de conexión de video."""
    if audio_enabled:
        stop_audio_fn()

    if cap_actual is not None:
        cap_actual.release()

    if nueva_ip is None:
        nueva_ip = ask_new_ip_fn(current_ip)

    if nueva_ip:
        update_stream_endpoints_fn(nueva_ip, record_wifi=True)
        schedule_video_connection_fn(force=True)

    return None


def open_ip_change_dialog(
    *,
    current_thread,
    get_current_ip_fn,
    ask_new_ip_fn,
    set_pending_ip_fn,
    clear_thread_fn,
):
    """Abre diálogo de cambio de IP en hilo separado."""
    if current_thread and current_thread.is_alive():
        return current_thread

    def runner():
        nueva_ip = ask_new_ip_fn(get_current_ip_fn())
        if nueva_ip and nueva_ip.strip():
            set_pending_ip_fn(nueva_ip.strip())
        clear_thread_fn()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread


def apply_pending_ip_change(*, pending_ip, cap_actual, cambiar_ip_fn):
    """Aplica IP pendiente si existe y limpia estado pendiente."""
    if pending_ip:
        cap_actual = cambiar_ip_fn(cap_actual, nueva_ip=pending_ip)
        return cap_actual, None
    return cap_actual, pending_ip
