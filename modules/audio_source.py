"""
Audio source selector for the Keras drone-detection engine.

ADAS3 has two physical audio paths that can feed the same Keras model:

  * ``phone_mic``     — PCM stream served by the Android client at /audio
                        (today's default; built-in phone microphone).
  * ``esp32_array``   — PCM stream produced by the ESP32 mic array,
                        relayed by the Android client at
                        /adas3/mic-array/pcm. The contract is defined in
                        modules/array_audio_bridge.py; the Android side
                        must implement it (the unified ESP32 firmware
                        already captures I2S but does not yet push raw
                        PCM over BT — that's the next step on the
                        client/firmware side).

This module owns the *selection*, not the streams. The streams are
managed by ``testcam.stream_audio`` (phone) and
``modules.array_audio_bridge.ArrayAudioBridge`` (ESP32 array). Both push
chunks into the same shared ``audio_buffer`` queue that
``run_audio_detection_worker`` consumes. Switching source means: stop
the active stream, start the other one.

The Keras engine doesn't know — and shouldn't know — which physical mic
produced the PCM. The contract for both streams is identical:

    int16 little-endian, mono, sample rate = 44100 Hz
    (same as today's /audio stream)

If the array stream eventually advertises a different rate or channel
count (e.g. 16 kHz to save BT bandwidth), the consumer side
(``stream_audio`` and the bridge) already negotiate it via the same
``Content-Type: audio/pcm; rate=<HZ>; channels=<N>`` header used by the
phone stream — no Keras change required.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Callable, Iterable, Optional


SOURCE_PHONE_MIC = "phone_mic"
SOURCE_ESP32_ARRAY = "esp32_array"

# Order matters: this is the cycle order if the UI just rotates between
# options instead of opening a dropdown.
SOURCES_IN_ORDER = (SOURCE_PHONE_MIC, SOURCE_ESP32_ARRAY)

# Stable mapping to translation keys defined in modules/translations_data.py
# (also added in this change). If a key is missing the i18n layer just
# returns the raw key, which is safe.
SOURCE_TO_LABEL_KEY = {
    SOURCE_PHONE_MIC: "audio_source_phone_mic",
    SOURCE_ESP32_ARRAY: "audio_source_esp32_array",
}


class AudioSourceController:
    """Thread-safe selector between phone-mic and ESP32-array audio.

    Persists the choice to a JSON config file so the user gets back the
    same source on next launch. The class is intentionally minimal: it
    does NOT start or stop streams — it just stores the chosen value
    and fires a callback so the caller (testcam.py) can react.
    """

    def __init__(
        self,
        *,
        config_file: Optional[str] = None,
        default_source: str = SOURCE_PHONE_MIC,
        on_change: Optional[Callable[[str, str], None]] = None,
    ):
        """``on_change(old, new)`` is invoked after a successful change,
        outside the lock. Failures are swallowed (logged)."""
        self._config_file = config_file
        self._on_change = on_change
        self._lock = threading.Lock()
        self._source = self._validate(default_source)
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
                saved = raw.get("audio_source")
                if saved in SOURCES_IN_ORDER:
                    self._source = saved
            except Exception as e:
                print(f"[AUDIO-SRC] No se pudo leer {config_file}: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def options() -> Iterable[str]:
        return SOURCES_IN_ORDER

    @staticmethod
    def is_valid(source: str) -> bool:
        return source in SOURCES_IN_ORDER

    def get(self) -> str:
        with self._lock:
            return self._source

    def is_phone(self) -> bool:
        return self.get() == SOURCE_PHONE_MIC

    def is_array(self) -> bool:
        return self.get() == SOURCE_ESP32_ARRAY

    def label_key(self) -> str:
        """Returns the i18n key for the *current* source. The UI can call
        ``t(controller.label_key())`` to get a localised label."""
        return SOURCE_TO_LABEL_KEY.get(self.get(), self.get())

    def set(self, new_source: str) -> bool:
        """Returns True if the value actually changed."""
        new_source = self._validate(new_source)
        with self._lock:
            old = self._source
            if new_source == old:
                return False
            self._source = new_source
        self._persist()
        if self._on_change is not None:
            try:
                self._on_change(old, new_source)
            except Exception as e:
                print(f"[AUDIO-SRC] callback de cambio fallo: {e}")
        return True

    def cycle(self) -> str:
        """Advance to the next source in ``SOURCES_IN_ORDER``. Returns
        the new value."""
        current = self.get()
        try:
            idx = SOURCES_IN_ORDER.index(current)
        except ValueError:
            idx = -1
        nxt = SOURCES_IN_ORDER[(idx + 1) % len(SOURCES_IN_ORDER)]
        self.set(nxt)
        return nxt

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(source: str) -> str:
        if source not in SOURCES_IN_ORDER:
            raise ValueError(
                f"Audio source no soportada: {source!r}. "
                f"Validas: {SOURCES_IN_ORDER}"
            )
        return source

    def _persist(self) -> None:
        if not self._config_file:
            return
        try:
            existing = {}
            if os.path.exists(self._config_file):
                try:
                    with open(self._config_file, "r", encoding="utf-8") as f:
                        existing = json.load(f) or {}
                except Exception:
                    existing = {}
            existing["audio_source"] = self.get()
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[AUDIO-SRC] No se pudo guardar {self._config_file}: {e}")


__all__ = [
    "AudioSourceController",
    "SOURCE_PHONE_MIC",
    "SOURCE_ESP32_ARRAY",
    "SOURCES_IN_ORDER",
    "SOURCE_TO_LABEL_KEY",
]
