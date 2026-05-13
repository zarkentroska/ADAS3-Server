import threading
import time

import requests


EP32_SUPPORTED_COMMANDS = {
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "A",
    "B",
    "MENU",
    "AUTO",
}

EP32_ACTION_TO_COMMAND = {
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "a": "A",
    "b": "B",
    "menu": "MENU",
    "auto": "AUTO",
}

EP32_ACTION_TO_SEQUENCE = {
    "fix_a": ["MENU", "A"],
    "fix_b": ["MENU", "B"],
    "auto_a": ["AUTO", "A"],
    "auto_b": ["AUTO", "B"],
}


class Ep32ClientController:
    """Gestiona envío de comandos EP32 hacia el cliente móvil por HTTP."""

    def __init__(
        self,
        *,
        base_url_supplier,
        endpoint_path="/adas3/ep32-command",
        timeout_seconds=1.8,
        default_delay_ms=180,
    ):
        self._base_url_supplier = base_url_supplier
        self._endpoint_path = str(endpoint_path or "/adas3/ep32-command")
        self._timeout = float(timeout_seconds)
        self._default_delay_ms = int(default_delay_ms)
        self._enabled = False
        self._status = "off"
        self._last_error = ""
        self._last_update_ts = time.time()
        self._lock = threading.Lock()

    def is_enabled(self):
        with self._lock:
            return bool(self._enabled)

    def get_state(self):
        with self._lock:
            return {
                "enabled": bool(self._enabled),
                "status": str(self._status),
                "last_error": str(self._last_error),
                "updated_at": float(self._last_update_ts),
            }

    def set_enabled(self, enabled):
        with self._lock:
            self._enabled = bool(enabled)
            if self._enabled:
                # El cliente móvil se encarga de escanear/conectar automáticamente.
                self._status = "scanning"
            else:
                self._status = "off"
            self._last_error = ""
            self._last_update_ts = time.time()

    def toggle_enabled(self):
        new_value = not self.is_enabled()
        self.set_enabled(new_value)
        return new_value

    def send_action(self, action_id):
        action = str(action_id or "").strip().lower()
        if action in EP32_ACTION_TO_COMMAND:
            return self.send_command(EP32_ACTION_TO_COMMAND[action])
        if action in EP32_ACTION_TO_SEQUENCE:
            return self.send_sequence(EP32_ACTION_TO_SEQUENCE[action], delay_ms=self._default_delay_ms)
        return self._mark_error("invalid_action", f"Acción EP32 no soportada: {action_id}")

    def send_command(self, command):
        token = str(command or "").strip().upper()
        if token not in EP32_SUPPORTED_COMMANDS:
            return self._mark_error("invalid_command", f"Comando EP32 inválido: {command}")
        payload = {
            "type": "adas3-ep32-command",
            "command": token,
        }
        return self._post(payload)

    def send_sequence(self, sequence, delay_ms=None):
        safe_seq = []
        for token in sequence or []:
            normalized = str(token or "").strip().upper()
            if normalized not in EP32_SUPPORTED_COMMANDS:
                return self._mark_error("invalid_command", f"Comando EP32 inválido en secuencia: {token}")
            safe_seq.append(normalized)
        if not safe_seq:
            return self._mark_error("invalid_sequence", "Secuencia EP32 vacía.")
        payload = {
            "type": "adas3-ep32-command",
            "sequence": safe_seq,
            "delay_ms": int(delay_ms if delay_ms is not None else self._default_delay_ms),
        }
        return self._post(payload)

    def _get_url(self):
        base_url = str(self._base_url_supplier() or "").strip().rstrip("/")
        if not base_url:
            return ""
        return f"{base_url}{self._endpoint_path}"

    def _post(self, payload):
        if not self.is_enabled():
            return self._mark_error("disabled", "EP32 BT desactivado.")

        url = self._get_url()
        if not url:
            return self._mark_error("invalid_url", "No hay URL base del cliente móvil.")

        try:
            response = requests.post(url, json=payload, timeout=self._timeout)
            status_code = int(response.status_code)
            body = {}
            try:
                body = response.json() if response.content else {}
            except Exception:
                body = {}
        except Exception as e:
            return self._mark_error("error", f"Error HTTP EP32: {e}")

        if status_code == 200:
            return self._mark_ok("connected", body)
        if status_code == 409:
            return self._mark_error("not_connected", "EP32 no conectada en el cliente móvil.", body=body)
        if status_code == 400:
            return self._mark_error("invalid_payload", "Payload EP32 inválido.", body=body)
        if status_code == 405:
            return self._mark_error("method_not_allowed", "Método HTTP no permitido.", body=body)
        return self._mark_error("error", f"Respuesta HTTP inesperada: {status_code}", body=body)

    def _mark_ok(self, status, body):
        with self._lock:
            self._status = status
            self._last_error = ""
            self._last_update_ts = time.time()
        return {"ok": True, "status": status, "body": body}

    def _mark_error(self, status, message, body=None):
        with self._lock:
            self._status = status
            self._last_error = str(message or "")
            self._last_update_ts = time.time()
        return {"ok": False, "status": status, "error": str(message or ""), "body": body or {}}
