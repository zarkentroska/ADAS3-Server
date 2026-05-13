import os
import re
import shlex
import subprocess
import sys
import threading
import tempfile
import time
import webbrowser
from tkinter import Tk, messagebox

from modules.mainthread_dispatch import schedule_dialog
from modules.ui_window_icon import apply_window_icon

IS_MACOS = sys.platform == "darwin"


def _show_message_async(kind, title, text):
    def _runner():
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        apply_window_icon(root)
        if kind == "error":
            messagebox.showerror(title, text)
        else:
            messagebox.showinfo(title, text)
        root.destroy()

    schedule_dialog(_runner)


def _looks_like_state_store_error(raw_text):
    text = str(raw_text or "").lower()
    return (
        "state store failed to initialize" in text
        or "state-store-init-error" in text
        or "failed to migrate existing tpm-seal" in text
        or "failed to unseal" in text
    )


def _build_state_store_error_message(t_func, status_text):
    details = str(status_text or "").strip()
    first_line = ""
    if details:
        first_line = details.splitlines()[0].strip()
    base = t_func("tailscale_state_store_error")
    if first_line:
        return f"{base}\n\n{first_line}"
    return base


def _build_cli_command(tailscale_cmd, subcommand, *, is_windows):
    if is_windows:
        return f'"{tailscale_cmd}" {subcommand}'
    return f"{shlex.quote(tailscale_cmd)} {subcommand}"


def _run_macos_admin_shell_command(command, timeout=180):
    """Ejecuta un comando shell en macOS solicitando credenciales de admin."""
    escaped = command.replace("\\", "\\\\").replace('"', '\\"')
    script = f'do shell script "{escaped}" with administrator privileges'
    return subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def install_tailscale(
    *,
    t_func,
    tailscale_installer_win,
    tailscale_installer_linux,
    tailscale_installed_fn,
):
    """Instala Tailscale en modo silencioso y devuelve (ok, error_text)."""
    if IS_MACOS:
        if tailscale_installed_fn():
            return True, ""

        pkg_url = "https://pkgs.tailscale.com/stable/Tailscale-latest-macos.pkg"
        pkg_path = os.path.join(tempfile.gettempdir(), "tailscale-latest-macos.pkg")
        try:
            download = subprocess.run(
                ["curl", "-L", "--fail", "-o", pkg_path, pkg_url],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if download.returncode != 0:
                details = (download.stderr or download.stdout or "").strip()
                if details:
                    return False, f"{t_func('tailscale_install_error')}\n\n{details[:300]}"
                return False, t_func("tailscale_install_error")

            install_cmd = f"installer -pkg {shlex.quote(pkg_path)} -target /"
            install = _run_macos_admin_shell_command(install_cmd, timeout=300)
            if install.returncode != 0:
                details = (install.stderr or install.stdout or "").strip()
                if details:
                    return False, f"{t_func('tailscale_install_error')}\n\n{details[:300]}"
                return False, t_func("tailscale_install_error")

            # Levantar la app para que el daemon quede listo para "tailscale up".
            subprocess.run(["open", "-a", "Tailscale"], capture_output=True, text=True, timeout=10)
            time.sleep(2)

            if tailscale_installed_fn():
                return True, ""
            return False, t_func("tailscale_install_error")
        except Exception as e:
            print(f"Error instalando Tailscale (macOS): {e}")
            return False, t_func("tailscale_install_error")
        finally:
            try:
                if os.path.exists(pkg_path):
                    os.remove(pkg_path)
            except Exception:
                pass

    if os.name == "nt":
        installer_path = tailscale_installer_win
        if not os.path.exists(installer_path):
            return False, t_func("tailscale_installer_not_found")

        try:
            cmd = f'"{installer_path}" /S'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                details = (result.stderr or result.stdout or "").strip()
                if details:
                    return False, f"{t_func('tailscale_install_error')}\n\n{details[:300]}"
                return False, t_func("tailscale_install_error")

            time.sleep(3)
            if tailscale_installed_fn():
                return True, ""
            return False, t_func("tailscale_install_error")
        except Exception as e:
            print(f"Error instalando Tailscale: {e}")
            return False, t_func("tailscale_install_error")

    installer_path = tailscale_installer_linux
    if not os.path.exists(installer_path):
        return False, t_func("tailscale_installer_not_found")

    try:
        os.chmod(installer_path, 0o755)
        cmd = f'sudo bash "{installer_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            if details:
                return False, f"{t_func('tailscale_install_error')}\n\n{details[:300]}"
            return False, t_func("tailscale_install_error")
        if tailscale_installed_fn():
            return True, ""
        return False, t_func("tailscale_install_error")
    except Exception as e:
        print(f"Error instalando Tailscale: {e}")
        return False, t_func("tailscale_install_error")


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
        is_macos = IS_MACOS
        tailscale_cmd = get_tailscale_path_fn()
        print(f"[TAILSCALE] Usando comando: {tailscale_cmd}")

        try:
            print("[TAILSCALE] Verificando estado actual...")
            status_cmd = _build_cli_command(tailscale_cmd, "status", is_windows=is_windows)
            status_result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True, timeout=5)
            print(f"[TAILSCALE] tailscale status returncode: {status_result.returncode}")
            print(f"[TAILSCALE] tailscale status stdout: {status_result.stdout[:200]}")
            status_text = f"{status_result.stdout}\n{status_result.stderr}".strip()

            if is_windows and _looks_like_state_store_error(status_text):
                print("[TAILSCALE] Detectado error de state store/TPM. Abortando intento de conexión.")
                _show_message_async(
                    "error",
                    t_func("tailscale_error"),
                    _build_state_store_error_message(t_func, status_text),
                )
                set_running_fn(False)
                return

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
                        status_check_text = f"{status_check.stdout}\n{status_check.stderr}".strip()
                        if _looks_like_state_store_error(status_check_text):
                            print("[TAILSCALE] Error de state store detectado tras tailscale up.")
                            _show_message_async(
                                "error",
                                t_func("tailscale_error"),
                                _build_state_store_error_message(t_func, status_check_text),
                            )
                            set_running_fn(False)
                            return
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

            if is_macos:
                # En macOS, asegurar que la app/daemon esté levantada.
                try:
                    subprocess.run(["open", "-a", "Tailscale"], capture_output=True, text=True, timeout=10)
                except Exception as e:
                    print(f"[TAILSCALE] Error abriendo app de Tailscale en macOS: {e}")

            if not is_windows:
                print("[TAILSCALE] POSIX: Ejecutando tailscale up en background...")
                try:
                    up_cmd = _build_cli_command(tailscale_cmd, "up", is_windows=is_windows)
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
                status_cmd_check = _build_cli_command(tailscale_cmd, "status", is_windows=is_windows)
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
                        status_check_text = f"{status_check.stdout}\n{status_check.stderr}".strip()
                        if is_windows and _looks_like_state_store_error(status_check_text):
                            print("[TAILSCALE] Error persistente de state store detectado en verificación.")
                            _show_message_async(
                                "error",
                                t_func("tailscale_error"),
                                _build_state_store_error_message(t_func, status_check_text),
                            )
                            set_running_fn(False)
                            return
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
            cmd = _build_cli_command(tailscale_cmd, "down", is_windows=is_windows)
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
