"""
HTTP bridge from the ADAS3 server to the Android client, which in turn
controls the ESP32 over Bluetooth SPP.

Flow:

    Server (this module)
        |  POST http(s)://<phone-ip>:<port>/adas3/ep32-command
        v
    Android client (StreamingServerHelper)
        |  Bluetooth SPP "UP\\n" / "DOWN\\n" / ...
        v
    ESP32 firmware (esp32-adas3.ino)

The server NEVER scans Bluetooth locally. The Android phone is the
Bluetooth master; this module only talks HTTP to it. Status values are
designed to reflect that bridge accurately so the user can tell whether
a failure is "phone offline", "phone reachable but ESP32 not paired",
"wrong base URL", etc., instead of a generic "scanning..." that hides
the real reason.
"""

import logging
import threading
import time

import requests


log = logging.getLogger("adas3.ep32_client")


# Whitelist of tokens the firmware understands. Anything else is rejected
# locally so we don't waste a round-trip to the phone for a command the
# ESP32 would ignore.
#
# The unified firmware (firmware/esp32-adas3/esp32-adas3.ino) accepts:
#     UP / DOWN / LEFT / RIGHT / TEST / STATUS
# A, B, MENU, AUTO are kept here for backwards compatibility with older
# firmware revisions that exposed them; the current firmware will simply
# log "Comando desconocido" if it receives one of them, which is
# harmless.
EP32_SUPPORTED_COMMANDS = {
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "TEST",
    "STATUS",
    # Legacy / forward-compat:
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
    "test": "TEST",
    "status": "STATUS",
    # Legacy / forward-compat:
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


# Status vocabulary used both in `state["status"]` and as the
# `ep32_status_<value>` translation key. Keep in sync with
# modules/translations_data.py.
STATUS_OFF = "off"
STATUS_IDLE = "idle"                       # enabled, never tried a command yet
STATUS_CONNECTED = "connected"
STATUS_NOT_CONNECTED = "not_connected"     # bridge reachable, ESP32 not paired
STATUS_BRIDGE_UNREACHABLE = "bridge_unreachable"
STATUS_INVALID_URL = "invalid_url"
STATUS_INVALID_PAYLOAD = "invalid_payload"
STATUS_METHOD_NOT_ALLOWED = "method_not_allowed"
STATUS_INVALID_COMMAND = "invalid_command"
STATUS_INVALID_SEQUENCE = "invalid_sequence"
STATUS_INVALID_ACTION = "invalid_action"
STATUS_ERROR = "error"
# Specific to the optional /adas3/ep32-control + /adas3/ep32-status endpoints
# that newer Android clients expose. If the bridge returns 404 (or any sign
# that those endpoints don't exist) we treat the bridge as "legacy" — the
# probe + send_command path still works.
STATUS_LEGACY_BRIDGE = "legacy_bridge"

# Whitelist of actions accepted by POST /adas3/ep32-control. Mirror of the
# Android side, kept small and explicit.
EP32_CONTROL_ACTIONS = {
    "enable",
    "disable",
    "reconnect",
    "stop",
}


class Ep32ClientController:
    """Posts EP32 commands to the Android client over HTTP."""

    def __init__(
        self,
        *,
        base_url_supplier,
        endpoint_path="/adas3/ep32-command",
        control_endpoint_path="/adas3/ep32-control",
        status_endpoint_path="/adas3/ep32-status",
        timeout_seconds=1.8,
        default_delay_ms=180,
    ):
        self._base_url_supplier = base_url_supplier
        self._endpoint_path = str(endpoint_path or "/adas3/ep32-command")
        self._control_endpoint_path = str(control_endpoint_path or "/adas3/ep32-control")
        self._status_endpoint_path = str(status_endpoint_path or "/adas3/ep32-status")
        self._timeout = float(timeout_seconds)
        self._default_delay_ms = int(default_delay_ms)
        self._enabled = False
        self._status = STATUS_OFF
        self._last_error = ""
        self._last_update_ts = time.time()
        self._last_command = ""
        # Latest snapshot from /adas3/ep32-status. Empty until we either
        # receive one or detect the endpoint is missing.
        self._bridge_status = {}
        # Capability flags. Lazily flipped to False on the first 404 / 405
        # so we don't keep hammering an endpoint a legacy APK doesn't have.
        # None = unknown, True/False = remembered.
        self._control_supported = None
        self._status_supported = None
        # Marca puesta a True tras el primer log WARNING de "unreachable"
        # para que los siguientes fallos consecutivos pasen a DEBUG y no
        # spameen la consola cada poll. Se resetea a False en cuanto el
        # bridge vuelve a responder o se reinicia el controller.
        self._unreachable_logged = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_enabled(self):
        with self._lock:
            return bool(self._enabled)

    def get_state(self):
        with self._lock:
            return {
                "enabled": bool(self._enabled),
                "status": str(self._status),
                "last_error": str(self._last_error),
                "last_command": str(self._last_command),
                "updated_at": float(self._last_update_ts),
                "bridge_status": dict(self._bridge_status),
                "control_supported": self._control_supported,
                "status_supported": self._status_supported,
            }

    def set_enabled(self, enabled):
        with self._lock:
            self._enabled = bool(enabled)
            if self._enabled:
                # Enabling EP32 BT from the server does NOT mean the server
                # itself talks Bluetooth. It just means "the Android client
                # is allowed to control the ESP32 on our behalf". Start in
                # `idle`; the first command (or an optional probe) will
                # promote the state to `connected`/`not_connected`/etc.
                self._status = STATUS_IDLE
            else:
                self._status = STATUS_OFF
            self._last_error = ""
            self._last_update_ts = time.time()
            self._last_command = ""

    def toggle_enabled(self):
        new_value = not self.is_enabled()
        self.set_enabled(new_value)
        return new_value

    # ------------------------------------------------------------------
    # Bridge probe (optional, lightweight)
    # ------------------------------------------------------------------

    def probe_bridge(self):
        """Ping the Android client to see if it answers. Uses a short GET
        against the EP32 endpoint, which returns 405 Method Not Allowed
        — that is the proof that the bridge is reachable AND the right
        endpoint exists. Refreshes the status accordingly.

        Returns the resulting status string.
        """
        url = self._get_url()
        if not url:
            return self._mark_error(STATUS_INVALID_URL,
                                    "No hay URL base del cliente movil.")["status"]
        try:
            response = requests.get(url, timeout=self._timeout)
            code = int(response.status_code)
        except (requests.ConnectionError, requests.Timeout) as e:
            return self._mark_error(STATUS_BRIDGE_UNREACHABLE,
                                    f"Cliente Android no responde: {e}")["status"]
        except Exception as e:
            return self._mark_error(STATUS_ERROR,
                                    f"Error probando puente: {e}")["status"]
        if code in (405, 200, 409):
            # 405 is the documented response of /adas3/ep32-command for GET;
            # 200/409 mean the endpoint exists too. Anything else = bad
            # routing / wrong port / not the Android client.
            with self._lock:
                self._status = STATUS_IDLE if self._enabled else STATUS_OFF
                self._last_error = ""
                self._last_update_ts = time.time()
            return self._status
        return self._mark_error(STATUS_ERROR,
                                f"Respuesta inesperada al probar puente: {code}")["status"]

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def send_action(self, action_id):
        action = str(action_id or "").strip().lower()
        if action in EP32_ACTION_TO_COMMAND:
            return self.send_command(EP32_ACTION_TO_COMMAND[action])
        if action in EP32_ACTION_TO_SEQUENCE:
            return self.send_sequence(EP32_ACTION_TO_SEQUENCE[action],
                                      delay_ms=self._default_delay_ms)
        return self._mark_error(STATUS_INVALID_ACTION,
                                f"Accion EP32 no soportada: {action_id}")

    def send_command(self, command):
        token = str(command or "").strip().upper()
        if token not in EP32_SUPPORTED_COMMANDS:
            return self._mark_error(STATUS_INVALID_COMMAND,
                                    f"Comando EP32 invalido: {command}")
        payload = {
            "type": "adas3-ep32-command",
            "command": token,
        }
        return self._post(payload, label=token)

    def send_sequence(self, sequence, delay_ms=None):
        safe_seq = []
        for token in sequence or []:
            normalized = str(token or "").strip().upper()
            if normalized not in EP32_SUPPORTED_COMMANDS:
                return self._mark_error(STATUS_INVALID_COMMAND,
                                        f"Comando EP32 invalido en secuencia: {token}")
            safe_seq.append(normalized)
        if not safe_seq:
            return self._mark_error(STATUS_INVALID_SEQUENCE,
                                    "Secuencia EP32 vacia.")
        payload = {
            "type": "adas3-ep32-command",
            "sequence": safe_seq,
            "delay_ms": int(delay_ms if delay_ms is not None else self._default_delay_ms),
        }
        return self._post(payload, label="+".join(safe_seq))

    # ------------------------------------------------------------------
    # /adas3/ep32-control + /adas3/ep32-status (new in Android client)
    # ------------------------------------------------------------------

    def request_control(self, action):
        """POST {"action": ...} to /adas3/ep32-control. Newer Android
        clients use this to flip the Bluetooth bridge on/off, reconnect, or
        stop from the server side, instead of the user toggling the switch
        in the Android UI.

        Accepts the literal action strings:
            "enable" | "disable" | "reconnect" | "stop"

        Behaviour against legacy APKs:
          - 404 / 405 from the endpoint → cache ``control_supported=False``
            and return ``{"ok": False, "status": "legacy_bridge", ...}``
            without raising. The caller is expected to fall back to
            ``probe_bridge()`` and direct command sends.

        Side effects: refreshes ``_status`` so the UI reflects the result
        (idle/connected/bridge_unreachable/not_connected/etc.) and stores
        the returned snapshot in ``_bridge_status``.
        """
        normalized = str(action or "").strip().lower()
        if normalized not in EP32_CONTROL_ACTIONS:
            return self._mark_error(
                STATUS_INVALID_ACTION,
                f"Accion EP32 (control) no soportada: {action}",
            )

        # Skip the round-trip if we already know the bridge is legacy.
        with self._lock:
            if self._control_supported is False:
                return {
                    "ok": False,
                    "status": STATUS_LEGACY_BRIDGE,
                    "error": "Endpoint /adas3/ep32-control no soportado por el cliente Android.",
                    "body": {},
                }

        url = self._build_url(self._control_endpoint_path)
        if not url:
            return self._mark_error(STATUS_INVALID_URL,
                                    "No hay URL base del cliente movil.")

        payload = {"type": "adas3-ep32-control", "action": normalized}
        try:
            response = requests.post(url, json=payload, timeout=self._timeout)
            status_code = int(response.status_code)
            body = self._safe_json(response)
        except (requests.ConnectionError, requests.Timeout) as e:
            log.warning("EP32 control bridge unreachable %s: %s", url, e)
            return self._mark_error(
                STATUS_BRIDGE_UNREACHABLE,
                f"Cliente Android no responde en {url}: {e}",
            )
        except Exception as e:
            log.exception("EP32 control HTTP error")
            return self._mark_error(STATUS_ERROR,
                                    f"Error HTTP EP32 control: {e}")

        if status_code in (200, 202):
            with self._lock:
                self._control_supported = True
                self._bridge_status = dict(body) if isinstance(body, dict) else {}
            return self._apply_control_outcome(normalized, body)

        if status_code in (404, 405):
            # Legacy APK: endpoint missing or not POST-able. Remember it
            # so we don't keep retrying.
            with self._lock:
                self._control_supported = False
                self._last_update_ts = time.time()
            return {
                "ok": False,
                "status": STATUS_LEGACY_BRIDGE,
                "error": "Cliente Android no expone /adas3/ep32-control.",
                "body": body,
            }
        if status_code == 400:
            return self._mark_error(STATUS_INVALID_PAYLOAD,
                                    "Payload EP32 (control) invalido.",
                                    body=body)
        if status_code == 409:
            return self._mark_error(
                STATUS_NOT_CONNECTED,
                "El cliente Android no puede activar el puente ahora mismo.",
                body=body,
            )
        return self._mark_error(STATUS_ERROR,
                                f"Respuesta HTTP inesperada en control: {status_code}",
                                body=body)

    def fetch_status(self):
        """GET /adas3/ep32-status and store the snapshot.

        Returns the parsed body on success, or ``{}`` on error. Updates
        ``_status`` to reflect what the bridge tells us (connected,
        scanning, error, …) and also flips ``_status_supported`` when the
        endpoint is missing.
        """
        with self._lock:
            if self._status_supported is False:
                return {}

        url = self._build_url(self._status_endpoint_path)
        if not url:
            self._mark_error(STATUS_INVALID_URL,
                             "No hay URL base del cliente movil.")
            return {}

        try:
            response = requests.get(url, timeout=self._timeout)
            status_code = int(response.status_code)
            body = self._safe_json(response)
        except (requests.ConnectionError, requests.Timeout) as e:
            # Rate-limit del log: en cuanto el puente está unreachable
            # de forma sostenida, el primer fallo se logea como WARNING
            # (para que el usuario lo vea) y los siguientes pasan a
            # DEBUG hasta que el bridge vuelva a responder. Antes esto
            # se logueaba como WARNING en cada poll y la consola se
            # llenaba.
            with self._lock:
                first = not self._unreachable_logged
                self._unreachable_logged = True
            if first:
                log.warning("EP32 status bridge unreachable %s: %s "
                            "(siguientes se silencian a DEBUG hasta "
                            "que vuelva a responder)", url, e)
            else:
                log.debug("EP32 status bridge unreachable %s: %s", url, e)
            self._mark_error(STATUS_BRIDGE_UNREACHABLE,
                             f"Cliente Android no responde en {url}: {e}")
            return {}
        except Exception as e:
            log.exception("EP32 status HTTP error")
            self._mark_error(STATUS_ERROR,
                             f"Error HTTP EP32 status: {e}")
            return {}

        if status_code == 200 and isinstance(body, dict):
            with self._lock:
                self._status_supported = True
                self._bridge_status = dict(body)
                # Si volvemos a tener respuesta, rehabilitamos el log
                # WARNING en el próximo fallo (señal genuina de que el
                # bridge se ha desconectado de nuevo).
                self._unreachable_logged = False
            self._apply_status_snapshot(body)
            return dict(body)
        if status_code in (404, 405):
            with self._lock:
                self._status_supported = False
                self._last_update_ts = time.time()
            return {}
        self._mark_error(STATUS_ERROR,
                         f"Respuesta HTTP inesperada en status: {status_code}",
                         body=body)
        return {}

    def supports_control(self):
        """True / False / None — None means we don't know yet."""
        with self._lock:
            return self._control_supported

    def supports_status(self):
        with self._lock:
            return self._status_supported

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_control_outcome(self, action, body):
        """Promote the controller status from the snapshot returned by
        /adas3/ep32-control. ``body`` follows the same shape as
        /adas3/ep32-status."""
        bridge_state = ""
        connected = False
        if isinstance(body, dict):
            bridge_state = str(body.get("state", "")).upper()
            connected = bool(body.get("connected", False))
        if action in ("disable", "stop"):
            with self._lock:
                self._enabled = False
                self._status = STATUS_OFF
                self._last_error = ""
                self._last_update_ts = time.time()
            return {"ok": True, "status": STATUS_OFF, "body": body}
        # enable / reconnect
        with self._lock:
            self._enabled = True
            if connected or bridge_state == "CONNECTED":
                self._status = STATUS_CONNECTED
            elif bridge_state in ("SCANNING", "CONNECTING"):
                self._status = STATUS_IDLE
            elif bridge_state == "ERROR":
                self._status = STATUS_ERROR
                self._last_error = str(body.get("detail", "")) if isinstance(body, dict) else ""
            else:
                self._status = STATUS_IDLE
            if self._status != STATUS_ERROR:
                self._last_error = ""
            self._last_update_ts = time.time()
        return {"ok": True, "status": self._status, "body": body}

    def _apply_status_snapshot(self, body):
        """Refresh ``_status`` from a /adas3/ep32-status snapshot."""
        bridge_state = str(body.get("state", "")).upper()
        connected = bool(body.get("connected", False))
        enabled_in_bridge = bool(body.get("enabled", False))
        with self._lock:
            if connected:
                self._status = STATUS_CONNECTED
                self._last_error = ""
            elif bridge_state in ("SCANNING", "CONNECTING"):
                self._status = STATUS_IDLE
            elif bridge_state == "ERROR":
                self._status = STATUS_ERROR
                self._last_error = str(body.get("detail", ""))
            elif not enabled_in_bridge:
                self._status = STATUS_NOT_CONNECTED
                self._last_error = "El puente EP32 esta desactivado en la app Android."
            else:
                self._status = STATUS_IDLE
            self._last_update_ts = time.time()

    def _build_url(self, path):
        base_url = str(self._base_url_supplier() or "").strip().rstrip("/")
        if not base_url:
            return ""
        return f"{base_url}{path}"

    def _safe_json(self, response):
        try:
            return response.json() if response.content else {}
        except Exception:
            return {}

    def _get_url(self):
        return self._build_url(self._endpoint_path)

    def _post(self, payload, label=""):
        if not self.is_enabled():
            return self._mark_error(STATUS_OFF, "EP32 BT desactivado.")

        url = self._get_url()
        if not url:
            return self._mark_error(STATUS_INVALID_URL,
                                    "No hay URL base del cliente movil.")

        try:
            response = requests.post(url, json=payload, timeout=self._timeout)
            status_code = int(response.status_code)
            body = {}
            try:
                body = response.json() if response.content else {}
            except Exception:
                body = {}
        except (requests.ConnectionError, requests.Timeout) as e:
            log.warning("EP32 bridge unreachable %s: %s", url, e)
            return self._mark_error(
                STATUS_BRIDGE_UNREACHABLE,
                f"Cliente Android no responde en {url}: {e}",
            )
        except Exception as e:
            log.exception("EP32 HTTP error")
            return self._mark_error(STATUS_ERROR, f"Error HTTP EP32: {e}")

        with self._lock:
            self._last_command = str(label)

        if status_code == 200:
            return self._mark_ok(STATUS_CONNECTED, body)
        if status_code == 409:
            return self._mark_error(
                STATUS_NOT_CONNECTED,
                "El cliente Android responde pero la ESP32 no esta emparejada.",
                body=body,
            )
        if status_code == 400:
            return self._mark_error(STATUS_INVALID_PAYLOAD,
                                    "Payload EP32 invalido.", body=body)
        if status_code == 405:
            return self._mark_error(STATUS_METHOD_NOT_ALLOWED,
                                    "Metodo HTTP no permitido.", body=body)
        return self._mark_error(STATUS_ERROR,
                                f"Respuesta HTTP inesperada: {status_code}", body=body)

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
        return {"ok": False, "status": status, "error": str(message or ""),
                "body": body or {}}
