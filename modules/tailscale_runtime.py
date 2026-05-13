import os
import sys


def get_tailscale_path():
    """Obtiene la ruta completa del ejecutable de Tailscale."""
    if os.name == "nt":
        possible_paths = [
            r"C:\Program Files\Tailscale\tailscale.exe",
            r"C:\Program Files (x86)\Tailscale\tailscale.exe",
            os.path.expanduser(r"~\AppData\Local\Tailscale\tailscale.exe"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return "tailscale"
    if sys.platform == "darwin":
        possible_paths = [
            "/usr/local/bin/tailscale",
            "/opt/homebrew/bin/tailscale",
            "/Applications/Tailscale.app/Contents/MacOS/tailscale",
            "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return "tailscale"
    return "tailscale"


def tailscale_installed(subprocess_module):
    """Verifica si Tailscale está instalado."""
    if os.name == "nt":
        try:
            result = subprocess_module.run(
                "tailscale --version",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

        common_paths = [
            r"C:\Program Files\Tailscale\tailscale.exe",
            r"C:\Program Files (x86)\Tailscale\tailscale.exe",
            os.path.expanduser(r"~\AppData\Local\Tailscale\tailscale.exe"),
        ]
        for path in common_paths:
            if os.path.exists(path):
                return True

        try:
            result = subprocess_module.run(
                "sc query Tailscale",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "SERVICE_NAME: Tailscale" in result.stdout:
                return True
        except Exception:
            pass
        return False

    try:
        if sys.platform == "darwin":
            # En macOS muchos entornos GUI arrancan sin PATH completo.
            tailscale_path = get_tailscale_path()
            if tailscale_path != "tailscale" and os.path.exists(tailscale_path):
                return True
        result = subprocess_module.run(
            "tailscale version",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def verificar_estado_tailscale(subprocess_module):
    """Devuelve True si Tailscale parece conectado."""
    if not tailscale_installed(subprocess_module):
        return False

    try:
        tailscale_cmd = get_tailscale_path()
        is_windows = os.name == "nt"
        status_cmd = f'"{tailscale_cmd}" status' if is_windows else f"{tailscale_cmd} status"
        status_result = subprocess_module.run(
            status_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status_result.returncode == 0:
            if "Logged in" in status_result.stdout or "100." in status_result.stdout:
                return True
        return False
    except Exception as e:
        print(f"Error verificando estado de Tailscale: {e}")
        return False


def get_tailscale_username(subprocess_module):
    """Obtiene el nombre de usuario de Tailscale."""
    try:
        tailscale_cmd = get_tailscale_path()
        is_windows = os.name == "nt"
        cmd = f'"{tailscale_cmd}" whoami' if is_windows else f"{tailscale_cmd} whoami"
        result = subprocess_module.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_tailscale_ip(subprocess_module):
    """Obtiene la IP de Tailscale de este dispositivo."""
    try:
        tailscale_cmd = get_tailscale_path()
        is_windows = os.name == "nt"
        cmd = f'"{tailscale_cmd}" ip -4' if is_windows else f"{tailscale_cmd} ip -4"
        result = subprocess_module.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_tailscale_connected_devices(subprocess_module):
    """Obtiene dispositivos conectados (online) en Tailscale."""
    devices = []
    try:
        tailscale_cmd = get_tailscale_path()
        is_windows = os.name == "nt"
        cmd = f'"{tailscale_cmd}" status' if is_windows else f"{tailscale_cmd} status"
        result = subprocess_module.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if not line.strip() or not line[0].isdigit():
                    continue
                if "offline" in line.lower():
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    ip = parts[0].strip()
                    device_name = parts[1].strip()
                    if "." in ip and len(ip.split(".")) == 4:
                        devices.append({"ip": ip, "name": device_name})
    except Exception as e:
        print(f"Error obteniendo dispositivos de Tailscale: {e}")
    return devices
