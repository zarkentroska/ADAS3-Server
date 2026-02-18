import os
import re
import subprocess
import threading
import time
import webbrowser
from tkinter import Tk, messagebox


def _show_message_async(kind, title, text):
    def _runner():
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if kind == "error":
            messagebox.showerror(title, text)
        else:
            messagebox.showinfo(title, text)
        root.destroy()

    threading.Thread(target=_runner, daemon=True).start()


def install_tailscale(
    *,
    t_func,
    tailscale_installer_win,
    tailscale_installer_linux,
    tailscale_installed_fn,
):
    """Instala Tailscale en modo silencioso."""
    if os.name == "nt":
        installer_path = tailscale_installer_win
        if not os.path.exists(installer_path):
            _show_message_async("error", t_func("error"), t_func("tailscale_installer_not_found"))
            return False

        def install_thread():
            try:
                cmd = f'"{installer_path}" /S'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    time.sleep(3)
                    if tailscale_installed_fn():
                        _show_message_async(
                            "info",
                            t_func("tailscale_install_success"),
                            t_func("tailscale_install_success"),
                        )
                    else:
                        _show_message_async(
                            "info",
                            t_func("tailscale_install_success"),
                            t_func("tailscale_install_success")
                            + "\n\n"
                            + "Si no se detecta, reinicia la aplicación.",
                        )
                else:
                    raise Exception(result.stderr)
            except Exception as e:
                print(f"Error instalando Tailscale: {e}")
                _show_message_async("error", t_func("error"), t_func("tailscale_install_error"))

        threading.Thread(target=install_thread, daemon=True).start()
        return True

    installer_path = tailscale_installer_linux
    if not os.path.exists(installer_path):
        _show_message_async("error", t_func("error"), t_func("tailscale_installer_not_found"))
        return False

    def install_thread():
        try:
            os.chmod(installer_path, 0o755)
            cmd = f'sudo bash "{installer_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                _show_message_async(
                    "info",
                    t_func("tailscale_install_success"),
                    t_func("tailscale_install_success"),
                )
            else:
                raise Exception(result.stderr)
        except Exception as e:
            print(f"Error instalando Tailscale: {e}")
            _show_message_async("error", t_func("error"), t_func("tailscale_install_error"))

    threading.Thread(target=install_thread, daemon=True).start()
    return True


def toggle_tailscale(
    *,
    t_func,
    get_running_fn,
    set_running_fn,
    tailscale_installed_fn,
    get_tailscale_path_fn,
):
    """Activa o desactiva Tailscale."""
    print("[TAILSCALE] toggle_tailscale() llamado")

    if not tailscale_installed_fn():
        print("[TAILSCALE] Tailscale no está instalado")
        _show_message_async("error", t_func("error"), t_func("tailscale_not_installed"))
        return

    def connect_tailscale():
        print("[TAILSCALE] connect_tailscale() iniciado")
        is_windows = os.name == "nt"
        tailscale_cmd = get_tailscale_path_fn()
        print(f"[TAILSCALE] Usando comando: {tailscale_cmd}")

        try:
            print("[TAILSCALE] Verificando estado actual...")
            status_cmd = f'"{tailscale_cmd}" status' if is_windows else f"{tailscale_cmd} status"
            status_result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True, timeout=5)
            print(f"[TAILSCALE] tailscale status returncode: {status_result.returncode}")
            print(f"[TAILSCALE] tailscale status stdout: {status_result.stdout[:200]}")

            auth_url = None
            if status_result.stdout:
                url_match = re.search(r"https://login\.tailscale\.com/a/[^\s\n]+", status_result.stdout)
                if url_match:
                    auth_url = url_match.group(0).strip()
                    print(f"[TAILSCALE] URL encontrada en tailscale status: {auth_url}")

            if status_result.returncode == 0:
                if "Logged in" in status_result.stdout or "100." in status_result.stdout:
                    print("[TAILSCALE] Ya está conectado")
                    set_running_fn(True)
                    return

            if is_windows and not auth_url:
                print("[TAILSCALE] Windows: Ejecutando tailscale up en background...")
                try:
                    up_cmd = f'"{tailscale_cmd}" up'
                    subprocess.Popen(
                        up_cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                    )
                    print("[TAILSCALE] tailscale up iniciado en background")
                    print("[TAILSCALE] Esperando unos segundos y verificando status para URL...")
                    time.sleep(2)

                    status_check_cmd = f'"{tailscale_cmd}" status'
                    for _ in range(3):
                        status_check = subprocess.run(
                            status_check_cmd,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=3,
                        )
                        if status_check.stdout:
                            url_match = re.search(
                                r"https://login\.tailscale\.com/a/[^\s\n]+",
                                status_check.stdout,
                            )
                            if url_match:
                                auth_url = url_match.group(0).strip()
                                print(
                                    f"[TAILSCALE] URL encontrada en status después de tailscale up: {auth_url}"
                                )
                                break
                        time.sleep(1)
                except Exception as e:
                    print(f"[TAILSCALE] Error ejecutando tailscale up: {e}")

            if auth_url:
                try:
                    webbrowser.open(auth_url)
                    print(f"[TAILSCALE] Navegador abierto con URL: {auth_url}")
                except Exception as e:
                    print(f"[TAILSCALE] Error abriendo navegador: {e}")

            if not is_windows:
                print("[TAILSCALE] Linux: Ejecutando tailscale up en background...")
                try:
                    up_cmd = f"{tailscale_cmd} up"
                    subprocess.Popen(
                        up_cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                    )
                    print("[TAILSCALE] Proceso tailscale up iniciado en background")
                except Exception as e:
                    print(f"[TAILSCALE] Error ejecutando tailscale up: {e}")

            def check_connection_periodic():
                print("[TAILSCALE] Iniciando verificación periódica de conexión...")
                max_attempts = 30 if auth_url else 10
                status_cmd_check = f'"{tailscale_cmd}" status' if is_windows else f"{tailscale_cmd} status"
                for i in range(max_attempts):
                    time.sleep(1)
                    try:
                        status_check = subprocess.run(
                            status_cmd_check,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if status_check.returncode == 0:
                            if "Logged in" in status_check.stdout or "100." in status_check.stdout:
                                set_running_fn(True)
                                print(f"[TAILSCALE] Conectado exitosamente (intento {i + 1})")
                                return
                        elif i % 5 == 0:
                            print(f"[TAILSCALE] Esperando conexión... (intento {i + 1}/{max_attempts})")
                    except Exception as e:
                        print(f"[TAILSCALE] Error verificando conexión: {e}")
                print(f"[TAILSCALE] No se detectó conexión después de {max_attempts} intentos")

            threading.Thread(target=check_connection_periodic, daemon=True).start()
        except Exception as e:
            print(f"Error conectando Tailscale: {e}")

    def disconnect_tailscale():
        try:
            tailscale_cmd = get_tailscale_path_fn()
            is_windows = os.name == "nt"
            cmd = f'"{tailscale_cmd}" down' if is_windows else f"{tailscale_cmd} down"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                set_running_fn(False)
            else:
                raise Exception(result.stderr)
        except Exception as e:
            print(f"Error desconectando Tailscale: {e}")

    if get_running_fn():
        threading.Thread(target=disconnect_tailscale, daemon=True).start()
    else:
        threading.Thread(target=connect_tailscale, daemon=True).start()
