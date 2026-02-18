def setup_adb_forward(subprocess_module, timeout=3):
    """Configura el túnel ADB local tcp:8080 -> tcp:8080."""
    try:
        subprocess_module.run(
            ["adb", "forward", "--remove", "tcp:8080"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        pass

    try:
        subprocess_module.run(
            ["adb", "forward", "tcp:8080", "tcp:8080"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return True
    except Exception as e:
        print(f"[ADB] Error configurando forward: {e}")
        return False


def teardown_adb_forward(subprocess_module, timeout=3):
    """Elimina el túnel ADB local."""
    try:
        subprocess_module.run(
            ["adb", "forward", "--remove", "tcp:8080"],
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

    if shutil_module.which("adb") is None:
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
            ["adb", "devices"],
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
        if setup_adb_forward(subprocess_module, timeout=3):
            adb_connected = True
            pending_ip_change = adb_target_ip
            messages.append("[ADB] Conectado. Cambiando IP a 127.0.0.1.")
        else:
            messages.append("[ADB] Fallo configurando el túnel. Mantengo IP actual.")
    elif not connected and adb_connected:
        adb_connected = False
        teardown_adb_forward(subprocess_module, timeout=3)
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
