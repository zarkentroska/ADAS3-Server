import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests


DEFAULT_COOLDOWNS = {
    "yolo": 30.0,
    "rf": 30.0,
    "audio": 30.0,
}


@dataclass
class TelegramEvent:
    event_type: str
    text: str
    photo_path: Optional[str] = None
    audio_path: Optional[str] = None
    remove_after_send: bool = False


class CooldownGate:
    """Controla la frecuencia de notificaciones por tipo de evento."""

    def __init__(self, cooldowns: Optional[Dict[str, float]] = None):
        self._lock = threading.Lock()
        self._last_sent_at: Dict[str, float] = {}
        self._cooldowns = DEFAULT_COOLDOWNS.copy()
        if cooldowns:
            self.update_cooldowns(cooldowns)

    def update_cooldowns(self, cooldowns: Dict[str, float]):
        with self._lock:
            for key, value in cooldowns.items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if parsed < 0:
                    parsed = 0.0
                self._cooldowns[key] = parsed

    def allow(self, event_type: str, now: Optional[float] = None):
        now = now if now is not None else time.time()
        with self._lock:
            cooldown_seconds = self._cooldowns.get(event_type, 30.0)
            last_sent = self._last_sent_at.get(event_type)
            if last_sent is not None and (now - last_sent) < cooldown_seconds:
                return False
            self._last_sent_at[event_type] = now
            return True


class TelegramClient:
    """Cliente mínimo para Telegram Bot API."""

    def __init__(self, token: str, chat_id: str, timeout_seconds: int = 12):
        self.token = token.strip()
        self.chat_id = str(chat_id).strip()
        self.timeout_seconds = timeout_seconds

    def _url(self, method: str):
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def send_message(self, text: str):
        payload = {"chat_id": self.chat_id, "text": text}
        response = requests.post(
            self._url("sendMessage"),
            data=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok", False):
            raise RuntimeError(body.get("description", "Error desconocido en sendMessage"))

    def send_photo(self, photo_path: str, caption: str):
        with open(photo_path, "rb") as image_file:
            payload = {"chat_id": self.chat_id, "caption": caption}
            files = {"photo": image_file}
            response = requests.post(
                self._url("sendPhoto"),
                data=payload,
                files=files,
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok", False):
            raise RuntimeError(body.get("description", "Error desconocido en sendPhoto"))

    def send_audio(self, audio_path: str, caption: str):
        with open(audio_path, "rb") as audio_file:
            payload = {"chat_id": self.chat_id, "caption": caption}
            files = {"audio": audio_file}
            response = requests.post(
                self._url("sendAudio"),
                data=payload,
                files=files,
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok", False):
            raise RuntimeError(body.get("description", "Error desconocido en sendAudio"))


class TelegramNotifier:
    """Gestiona cola asíncrona + cooldown para notificaciones Telegram."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        token: str = "",
        chat_id: str = "",
        cooldowns: Optional[Dict[str, float]] = None,
    ):
        self._enabled = enabled
        self._token = token.strip()
        self._chat_id = str(chat_id).strip()
        self._cooldown_gate = CooldownGate(cooldowns)
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()

    def start(self):
        with self._state_lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            print("[TELEGRAM] Worker iniciado")

    def stop(self):
        with self._state_lock:
            self._stop_event.set()
            worker = self._worker_thread
        if worker and worker.is_alive():
            worker.join(timeout=2)
        print("[TELEGRAM] Worker detenido")

    def update_settings(self, *, enabled: bool, token: str, chat_id: str, cooldowns: Optional[Dict[str, float]] = None):
        with self._state_lock:
            self._enabled = bool(enabled)
            self._token = (token or "").strip()
            self._chat_id = str(chat_id or "").strip()
        if cooldowns:
            self._cooldown_gate.update_cooldowns(cooldowns)

    def enqueue(self, event: TelegramEvent):
        if not isinstance(event, TelegramEvent):
            return False
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            print("[TELEGRAM] Cola llena, se descarta evento")
            return False

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_event(event)
            finally:
                self._queue.task_done()

    def _process_event(self, event: TelegramEvent):
        with self._state_lock:
            enabled = self._enabled
            token = self._token
            chat_id = self._chat_id

        if not enabled:
            self._cleanup_files(event)
            return

        if not token or not chat_id:
            print("[TELEGRAM] Configuración incompleta: token/chat_id vacíos")
            self._cleanup_files(event)
            return

        if not self._cooldown_gate.allow(event.event_type):
            print(f"[TELEGRAM] Evento omitido por cooldown ({event.event_type})")
            self._cleanup_files(event)
            return

        client = TelegramClient(token=token, chat_id=chat_id)
        try:
            if event.photo_path and os.path.exists(event.photo_path):
                client.send_photo(event.photo_path, event.text)
                if event.audio_path and os.path.exists(event.audio_path):
                    client.send_audio(event.audio_path, f"Audio ({event.event_type})")
            elif event.audio_path and os.path.exists(event.audio_path):
                client.send_audio(event.audio_path, event.text)
            else:
                client.send_message(event.text)
            print(f"[TELEGRAM] Notificación enviada ({event.event_type})")
        except Exception as exc:
            print(f"[TELEGRAM] Error al enviar notificación ({event.event_type}): {exc}")
        finally:
            self._cleanup_files(event)

    def _cleanup_files(self, event: TelegramEvent):
        if not event.remove_after_send:
            return
        for file_path in (event.photo_path, event.audio_path):
            if not file_path:
                continue
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as exc:
                print(f"[TELEGRAM] No se pudo borrar temporal '{file_path}': {exc}")
