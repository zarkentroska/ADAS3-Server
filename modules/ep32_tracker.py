"""
Auto-tracking de drones detectados por YOLO mediante el controlador EP32.

Calcula el desvío del centro del bounding box más confiable respecto
al centro del frame y emite comandos direccionales para recentrar el trípode.
Implementa zona muerta, cooldown entre comandos y suavizado para
evitar oscilaciones.
"""

import threading
import time


class Ep32AutoTracker:

    def __init__(
        self,
        ep32_controller,
        *,
        frame_width=1280,
        frame_height=720,
        dead_zone_x=0.08,
        dead_zone_y=0.08,
        command_cooldown=0.35,
    ):
        self._controller = ep32_controller
        self._frame_w = frame_width
        self._frame_h = frame_height
        # Fracción del frame que se considera "centrado" (sin corrección).
        self._dead_zone_x = dead_zone_x
        self._dead_zone_y = dead_zone_y
        self._cooldown = command_cooldown
        self._enabled = False
        self._last_cmd_time = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Activación
    # ------------------------------------------------------------------

    def is_enabled(self):
        with self._lock:
            return self._enabled

    def set_enabled(self, value):
        with self._lock:
            self._enabled = bool(value)

    def toggle(self):
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled

    # ------------------------------------------------------------------
    # Lógica principal
    # ------------------------------------------------------------------

    def update(self, boxes_data, frame_width=None, frame_height=None):
        """Recibe las detecciones YOLO del frame actual y envía un
        comando si es necesario.  Debe llamarse en cada frame.

        ``boxes_data`` es la lista de dicts con claves x1, y1, x2, y2, conf.
        """
        if not self.is_enabled():
            return
        if not self._controller.is_enabled():
            return
        if not boxes_data:
            return

        fw = frame_width or self._frame_w
        fh = frame_height or self._frame_h

        now = time.time()
        if now - self._last_cmd_time < self._cooldown:
            return

        best = max(boxes_data, key=lambda b: b.get("conf", 0.0))

        cx = (best["x1"] + best["x2"]) / 2.0
        cy = (best["y1"] + best["y2"]) / 2.0

        # Desvío normalizado [-1, 1] respecto al centro del frame.
        off_x = (cx - fw / 2.0) / (fw / 2.0)
        off_y = (cy - fh / 2.0) / (fh / 2.0)

        action = None

        # Priorizamos el eje con mayor desvío para no enviar dos comandos
        # simultáneos al Zifon (movimiento discreto, un eje por pulso).
        abs_x = abs(off_x)
        abs_y = abs(off_y)

        if abs_x > self._dead_zone_x or abs_y > self._dead_zone_y:
            if abs_x >= abs_y and abs_x > self._dead_zone_x:
                action = "right" if off_x > 0 else "left"
            elif abs_y > self._dead_zone_y:
                action = "down" if off_y > 0 else "up"

        if action is None:
            return

        self._last_cmd_time = now
        threading.Thread(
            target=self._controller.send_action,
            args=(action,),
            daemon=True,
        ).start()
