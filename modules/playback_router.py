"""
PlaybackRouter — single shared PyAudio output stream used by every audio
source (phone_mic, esp32_array, future sources).

Why this exists
---------------
Before this module, the server had **two** independent PyAudio output
streams alive at the same time:

    stream_audio()          (phone path)  -> pa.open(..., output=True)
    _ensure_array_playback_stream()       -> pa.open(..., output=True)

PortAudio/ALSA on Linux and CoreAudio on macOS don't tolerate that
well: opening a second exclusive output stream while the first is
alive can silence both, or one of them goes mute permanently until
process restart. The user-visible regression was "neither phone_mic
nor esp32_array reproduces audio after switching once".

The fix is structural: have exactly one writeable PyAudio output
stream in the whole process. Every audio source feeds `write_chunk`;
the router reopens the underlying stream only when the format
changes (different rate or channel count). The mute toggle lives
here so all sources honour it identically.

Testability
-----------
The router takes an injectable `pyaudio_factory` and ``format``
parameter so unit tests can substitute fakes — no real PyAudio
instance is required. ``DummyPyAudio`` in the tests provides a
record-only ``open`` returning a stream that captures `write` calls.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional, Tuple


log = logging.getLogger("adas3.playback_router")


class PlaybackRouter:
    """Single PyAudio output stream shared across audio sources.

    Parameters
    ----------
    pyaudio_factory : callable
        Returns a ``PyAudio``-compatible instance (must expose
        ``open(format, channels, rate, output, frames_per_buffer)``).
        In tests this is replaced by a fake. In production it's
        ``get_pyaudio_instance``.
    audio_format : int
        Sample format passed verbatim to ``pa.open``. In production
        this is ``pyaudio.paInt16`` — injected to keep the module
        importable in test runs that don't have PyAudio installed.
    frames_per_buffer : int
        Buffer size for the underlying stream. Defaults to 1024,
        matching the legacy ``CHUNK`` in testcam.
    on_log : callable, optional
        Diagnostic log sink. Defaults to ``print`` with a tag.
    """

    def __init__(
        self,
        *,
        pyaudio_factory: Callable[[], Any],
        audio_format: int,
        frames_per_buffer: int = 1024,
        on_log: Optional[Callable[[str], None]] = None,
    ):
        self._pyaudio_factory = pyaudio_factory
        self._format = audio_format
        self._frames_per_buffer = max(64, int(frames_per_buffer))
        self._on_log = on_log or (lambda msg: print(f"[PLAYBACK] {msg}"))

        self._lock = threading.Lock()
        self._stream = None
        self._rate = 0
        self._channels = 0
        self._muted = False
        # Bookkeeping for diagnostics / tests.
        self._chunks_written = 0
        self._bytes_written = 0
        self._chunks_dropped_muted = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_muted(self, muted: bool) -> None:
        """Toggle global playback mute. Both sources honour this."""
        with self._lock:
            self._muted = bool(muted)

    def is_muted(self) -> bool:
        with self._lock:
            return self._muted

    def write_chunk(self, chunk: bytes, rate: int, channels: int) -> bool:
        """Submit a PCM chunk for playback.

        Returns True if the chunk was written, False if it was dropped
        (muted, empty, or stream open failed).

        ``stream.write`` se ejecuta bajo el mismo lock que open/close:
        en macOS (CoreAudio) y Linux (ALSA/Pulse) dos hilos escribiendo
        a la vez (phone ``stream_audio`` + ``ArrayAudioBridge``) pueden
        provocar segfault del proceso.
        """
        if not chunk:
            return False
        with self._lock:
            if self._muted:
                self._chunks_dropped_muted += 1
                return False
            stream = self._ensure_stream_unlocked(rate, channels)
            if stream is None:
                return False
            try:
                stream.write(chunk)
            except Exception as e:
                # Most likely the device went away (USB unplug, screen
                # lock, etc). Force a reopen on the next chunk.
                self._on_log(f"write fallo, recreando stream: {e}")
                self._close_locked()
                return False
            self._chunks_written += 1
            self._bytes_written += len(chunk)
            return True

    def close(self) -> None:
        """Close the underlying stream. Safe to call multiple times."""
        with self._lock:
            self._close_locked()

    def get_state(self) -> dict:
        with self._lock:
            return {
                "rate": self._rate,
                "channels": self._channels,
                "muted": self._muted,
                "stream_open": self._stream is not None,
                "chunks_written": self._chunks_written,
                "bytes_written": self._bytes_written,
                "chunks_dropped_muted": self._chunks_dropped_muted,
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Formatos que probamos como fallback cuando el primer `pa.open`
    # falla con el rate/channels que pide el llamador. Algunas tarjetas
    # de salida exigen 48000 Hz aunque el stream entrante venga a otra
    # tasa: PyAudio se queda silencioso en lugar de resamplear, así que
    # mejor abrimos el dispositivo a un rate que sí acepta y dejamos
    # que el SO lo mezcle. Si NINGUNO funciona, devolvemos None y
    # logueamos claro para que el usuario sepa que es PyAudio, no
    # nuestro código de routing, lo que ha fallado.
    _FALLBACK_FORMATS = (
        (44100, 1),
        (48000, 1),
        (44100, 2),
        (48000, 2),
        (16000, 1),
        (8000, 1),
    )

    def _try_open(self, pa, rate, channels):
        return pa.open(
            format=self._format,
            channels=channels,
            rate=rate,
            output=True,
            frames_per_buffer=self._frames_per_buffer,
        )

    def _ensure_stream(self, rate: int, channels: int):
        """API pública: adquiere el lock y delega."""
        with self._lock:
            return self._ensure_stream_unlocked(rate, channels)

    def _ensure_stream_unlocked(self, rate: int, channels: int):
        """Abre o reutiliza el stream. El llamador DEBE tener ``_lock``."""
        rate = max(8000, min(int(rate or 44100), 96000))
        channels = max(1, min(int(channels or 1), 2))
        if (self._stream is not None and
                self._rate == rate and
                self._channels == channels):
            return self._stream
        # Format changed (or first open): close any previous stream
        # and open a fresh one. Doing this under the lock prevents
        # two threads (phone + array transition) from racing to
        # open two streams at the same time.
        self._close_locked()
        try:
            pa = self._pyaudio_factory()
        except Exception as e:
            self._on_log(
                "PyAudio factory falló — no habrá playback hasta "
                f"que se resuelva. Detalle: {e}"
            )
            return None
        # Intento 1: rate/channels exactos pedidos por el llamador.
        try:
            stream = self._try_open(pa, rate, channels)
            self._stream = stream
            self._rate = rate
            self._channels = channels
            self._on_log(
                f"playback abierto OK a {channels}ch @ {rate} Hz"
            )
            return stream
        except Exception as e_primary:
            self._on_log(
                f"playback no pudo abrir {channels}ch @ {rate} Hz "
                f"({e_primary}); probando formatos de fallback…"
            )
        # Intentos de fallback: muchas máquinas Linux con PulseAudio
        # o ALSA sólo abren 44100/48000 mono o estéreo. Si nada
        # funciona, devolvemos None y el bridge sigue alimentando
        # Keras (sin sonido por altavoz) — la detección NO se rompe
        # por un fallo de salida.
        for fb_rate, fb_channels in self._FALLBACK_FORMATS:
            if (fb_rate, fb_channels) == (rate, channels):
                continue
            # No abrir estéreo si el stream entrante es mono: PortAudio
            # interpreta mal buffers mono y suena a ruido blanco.
            if channels == 1 and fb_channels != 1:
                continue
            try:
                stream = self._try_open(pa, fb_rate, fb_channels)
                self._stream = stream
                self._rate = fb_rate
                self._channels = fb_channels
                self._on_log(
                    f"playback abierto en fallback "
                    f"{fb_channels}ch @ {fb_rate} Hz "
                    f"(pedido era {channels}ch @ {rate} Hz)"
                )
                return stream
            except Exception:
                continue
        self._on_log(
            "no se pudo abrir NINGÚN stream PyAudio "
            "(ni primario ni fallbacks). Keras seguirá detectando "
            "pero el altavoz queda mudo hasta que se reasignen "
            "permisos/devices del SO."
        )
        self._stream = None
        self._rate = 0
        self._channels = 0
        return None

    def _close_locked(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._rate = 0
        self._channels = 0


__all__ = ["PlaybackRouter"]
