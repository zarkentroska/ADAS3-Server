import os
import sys


def _platform_tools_adb(sdk_root, *, windows):
    name = "adb.exe" if windows else "adb"
    return os.path.join(sdk_root, "platform-tools", name)


def _sdk_adb_candidates(home, *, windows):
    roots = []
    for key in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        value = os.environ.get(key, "").strip()
        if value:
            roots.append(value)
    if sys.platform == "darwin":
        roots.append(os.path.join(home, "Library", "Android", "sdk"))
    elif os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app:
            roots.append(os.path.join(local_app, "Android", "Sdk"))
        roots.append(os.path.join(home, "AppData", "Local", "Android", "Sdk"))
    else:
        roots.append(os.path.join(home, "Android", "Sdk"))

    seen = set()
    candidates = []
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        candidates.append(_platform_tools_adb(root, windows=windows))
    return candidates


def get_adb_path(shutil_module):
    """
    Resuelve el ejecutable ``adb``.

    En macOS las apps GUI suelen arrancar sin PATH completo (Homebrew, SDK, etc.),
    por eso se comprueban rutas habituales además de ``shutil.which``.
    """
    found = shutil_module.which("adb")
    if found:
        return found

    home = os.path.expanduser("~")
    windows = os.name == "nt"
    candidates = []

    if sys.platform == "darwin":
        candidates.extend(
            [
                "/opt/homebrew/bin/adb",
                "/usr/local/bin/adb",
            ]
        )
    elif windows:
        candidates.extend(
            [
                r"C:\platform-tools\adb.exe",
                os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Android", "android-sdk", "platform-tools", "adb.exe"),
            ]
        )
    else:
        candidates.extend(["/usr/local/bin/adb", "/usr/bin/adb"])

    candidates.extend(_sdk_adb_candidates(home, windows=windows))

    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def setup_adb_forward(subprocess_module, adb_cmd, timeout=3):
    """Configura el túnel ADB local tcp:8080 -> tcp:8080."""
    try:
        subprocess_module.run(
            [adb_cmd, "forward", "--remove", "tcp:8080"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        pass

    try:
        subprocess_module.run(
            [adb_cmd, "forward", "tcp:8080", "tcp:8080"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return True
    except Exception as e:
        print(f"[ADB] Error configurando forward: {e}")
        return False


def teardown_adb_forward(subprocess_module, adb_cmd, timeout=3):
    """Elimina el túnel ADB local."""
    try:
        subprocess_module.run(
            [adb_cmd, "forward", "--remove", "tcp:8080"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        pass


def poll_adb_connection(
    *,
    last_adb_check,
    adb_check_interval,
    adb_connected,
    pending_ip_change,
    last_wifi_ip,
    current_ip,
    adb_target_ip,
    subprocess_module,
    shutil_module,
    time_module,
):
    """
    Evalúa el estado de ADB y devuelve el nuevo estado sin tocar globales.
    """
    now = time_module.time()
    if now - last_adb_check < adb_check_interval:
        return {
            "last_adb_check": last_adb_check,
            "adb_connected": adb_connected,
            "pending_ip_change": pending_ip_change,
            "last_wifi_ip": last_wifi_ip,
            "messages": [],
        }

    last_adb_check = now

    adb_cmd = get_adb_path(shutil_module)
    if adb_cmd is None:
        if adb_connected:
            adb_connected = False
        return {
            "last_adb_check": last_adb_check,
            "adb_connected": adb_connected,
            "pending_ip_change": pending_ip_change,
            "last_wifi_ip": last_wifi_ip,
            "messages": [],
        }

    try:
        result = subprocess_module.run(
            [adb_cmd, "devices"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        lines = result.stdout.strip().splitlines()
        connected = any("\tdevice" in line for line in lines[1:])
    except Exception:
        connected = False

    messages = []

    if connected and not adb_connected:
        if last_wifi_ip is None or (not last_wifi_ip or last_wifi_ip == adb_target_ip):
            last_wifi_ip = current_ip
        if setup_adb_forward(subprocess_module, adb_cmd, timeout=3):
            adb_connected = True
            pending_ip_change = adb_target_ip
            messages.append("[ADB] Conectado. Cambiando IP a 127.0.0.1.")
        else:
            messages.append("[ADB] Fallo configurando el túnel. Mantengo IP actual.")
    elif not connected and adb_connected:
        adb_connected = False
        teardown_adb_forward(subprocess_module, adb_cmd, timeout=3)
        if last_wifi_ip and last_wifi_ip != adb_target_ip:
            pending_ip_change = last_wifi_ip
            messages.append("[ADB] Desconectado. Volviendo a IP anterior.")

    return {
        "last_adb_check": last_adb_check,
        "adb_connected": adb_connected,
        "pending_ip_change": pending_ip_change,
        "last_wifi_ip": last_wifi_ip,
        "messages": messages,
    }
