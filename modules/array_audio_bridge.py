"""
ArrayAudioBridge — pulls raw PCM from the ESP32 mic-array path on the
Android client and pushes it into the shared audio buffer that the Keras
detection worker already consumes.

Wire contract (server consumes; client must implement):

    GET <base_url>/adas3/mic-array/pcm
        Connection: keep-alive
        Content-Type: audio/pcm; rate=<HZ>; channels=<N>
                       [; pair=A|B|sum]
        Body: raw int16 little-endian PCM, mono by default.

This mirrors the existing /audio endpoint exactly (same shape, same
parsing on the server side), so the Keras pipeline doesn't need to know
which physical mic produced the samples. Switching the audio source
amounts to: ``stream_audio`` stops, ``ArrayAudioBridge.start()``
starts. Both push int16 chunks into the same shared
``audio_buffer: queue.Queue``.

If a future client cannot stream pure PCM and instead chooses NDJSON
with base64-encoded chunks (e.g. ``{"type":"pcm","b64":"...","seq":42}``),
the bridge also accepts that as a fallback by sniffing the Content-Type
(``application/x-ndjson`` or ``application/json``).

When ``fallback_endpoint_path`` is set (default ``/audio``) and the
primary ``/adas3/mic-array/pcm`` returns HTTP 404, the bridge retries on
the fallback. This matches the current Android client, which already
upsamples ESP32 array PCM to 44100 Hz and serves it on ``/audio`` while
``esp32_array`` is selected.

The bridge does NOT touch any acoustic event (``acoustic_array``,
``heartbeat``); those still flow through ``acoustic_integration.py``.
This is strictly the **audio for Keras** path.

Diagnostics added in this revision:
  * RMS and peak of the last chunk, exposed via ``get_state()`` so the
    UI can distinguish "bytes flowing" from "bytes carry useful signal".
  * Silence detection (``silence`` flag) when peak < SILENCE_PEAK for
    SILENCE_STREAK_LIMIT consecutive chunks.
  * Optional ``on_pcm_chunk`` callback so testcam can hook a PyAudio
    playback stream — the same speaker the phone-mic source uses.
  * Optional ``software_gain`` (1.0 = passthrough) applied with safe
    int16 saturation, useful when MEMS levels are very low.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import struct
import threading
import time
from typing import Any, Callable, Optional

import requests


log = logging.getLogger("adas3.array_audio_bridge")

# Match the Keras worker's expectations.
_DEFAULT_SAMPLE_RATE = 44100
_DEFAULT_CHANNELS = 1
_DEFAULT_CHUNK = 1024  # bytes per HTTP read; matches phone stream cadence
_DEFAULT_TIMEOUT = (15, 30)  # (connect, read)
_RECONNECT_BACKOFF_S = 3.0

# A peak < 200 (i.e. < -44 dBFS approx) sustained for ~SILENCE_STREAK_LIMIT
# chunks is treated as "the array is sending zeros / floor noise". This is
# the strongest signal we can give the user: "kbps subiendo pero el array
# está mudo, revisa cableado / SEL pin / 3V3".
_SILENCE_PEAK = 200
_SILENCE_STREAK_LIMIT = 20


def _compute_rms_peak_int16(pcm_bytes: bytes) -> tuple:
    """Returns (rms, peak) of an int16 LE PCM payload, both in absolute
    int16 scale (0..32767). Tolerant to odd-length payloads (drops the
    last byte) and to an empty input (returns (0, 0))."""
    if not pcm_bytes:
        return 0, 0
    n = len(pcm_bytes) // 2
    if n == 0:
        return 0, 0
    # struct.unpack is the fastest pure-stdlib path; avoids pulling numpy
    # into the bridge just for a level meter.
    fmt = "<" + ("h" * n)
    try:
        samples = struct.unpack(fmt, pcm_bytes[: n * 2])
    except struct.error:
        return 0, 0
    sum_sq = 0
    peak = 0
    for s in samples:
        a = -s if s < 0 else s
        if a > peak:
            peak = a
        sum_sq += s * s
    rms = int((sum_sq / n) ** 0.5) if n else 0
    return rms, peak


def _apply_int16_gain(pcm_bytes: bytes, gain: float) -> bytes:
    """Apply a multiplicative gain to an int16 LE PCM payload with
    saturation to [-32768, 32767]. gain == 1.0 returns the bytes
    unchanged."""
    if not pcm_bytes or gain == 1.0:
        return pcm_bytes
    n = len(pcm_bytes) // 2
    if n == 0:
        return pcm_bytes
    fmt = "<" + ("h" * n)
    try:
        samples = struct.unpack(fmt, pcm_bytes[: n * 2])
    except struct.error:
        return pcm_bytes
    out = []
    for s in samples:
        v = int(s * gain)
        if v > 32767:
            v = 32767
        elif v < -32768:
            v = -32768
        out.append(v)
    return struct.pack(fmt, *out)


class ArrayAudioBridge:
    """Background HTTP consumer for /adas3/mic-array/pcm.

    Pushes int16 LE PCM chunks (exactly the format the existing audio
    worker expects) into a caller-provided ``queue.Queue``. Caller is
    responsible for:

      * Disabling ``stream_audio`` (the phone path) before calling
        ``start()``, so only one source feeds the queue at a time.
      * Re-enabling ``stream_audio`` after ``stop()`` if it switches
        back to phone mic.

    Content-Type handling:
      * ``audio/pcm`` — body is fed through verbatim.
      * ``application/x-ndjson`` / ``application/json`` — each line is a
        JSON object with at least ``b64`` (base64 of int16 LE PCM)
        and optionally ``rate`` / ``channels``.
    """

    def __init__(
        self,
        *,
        base_url_supplier: Callable[[], str],
        audio_buffer: "queue.Queue[bytes]",
        endpoint_path: str = "/adas3/mic-array/pcm",
        fallback_endpoint_path: Optional[str] = "/audio",
        chunk_size: int = _DEFAULT_CHUNK,
        on_log: Optional[Callable[[str], None]] = None,
        on_stream_meta: Optional[Callable[[int, int], None]] = None,
        on_state: Optional[Callable[[str, str], None]] = None,
        should_push: Optional[Callable[[], bool]] = None,
        on_pcm_chunk: Optional[Callable[[bytes, int, int], None]] = None,
        software_gain: float = 1.0,
        keras_target_bytes_per_s: int = 88200,
        clock_fn: Optional[Callable[[], float]] = None,
    ):
        self._base_url_supplier = base_url_supplier
        self._audio_buffer = audio_buffer
        self._primary_endpoint_path = endpoint_path
        self._fallback_endpoint_path = (
            fallback_endpoint_path if fallback_endpoint_path else None
        )
        self._endpoint_path = endpoint_path
        self._using_fallback = False
        self._chunk_size = max(256, int(chunk_size))
        self._on_log = on_log or (lambda msg: print(f"[ARRAY-PCM] {msg}"))
        self._on_stream_meta = on_stream_meta
        self._on_state = on_state or (lambda new, detail: None)
        # Predicate consulted on every _push: if it returns False (e.g.
        # the user switched audio source to phone_mic), the chunk is
        # discarded instead of pushed to the shared buffer. Defends
        # against dual-stream during source transitions or slow socket
        # closures.
        self._should_push = should_push or (lambda: True)
        # Optional hook for playback / external VU. Called once per
        # chunk after the predicate (so muted/no-source chunks DO NOT
        # reach the speakers). Exceptions inside the hook are swallowed
        # so the bridge keeps running.
        self._on_pcm_chunk = on_pcm_chunk

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = "off"
        self._last_error = ""
        # Number of PCM bytes pushed since start(); useful for the UI
        # ("the array is feeding Keras at ~X kB/s").
        self._bytes_pushed = 0
        self._last_chunk_at = 0.0
        # Counter of bytes dropped by the should_push predicate. Useful
        # for diagnosing dual-stream transients during source changes.
        self._bytes_dropped_predicate = 0
        # Level meter of the last chunk processed. Distinguishes
        # "bytes flowing" from "bytes carry useful signal".
        self._last_rms = 0
        self._last_peak = 0
        # Streak counter of consecutive low-peak chunks. When it exceeds
        # _SILENCE_STREAK_LIMIT, get_state() reports silence=True. This
        # surfaces the "kbps subiendo pero array mudo" case as a clear
        # UI signal instead of leaving the user guessing.
        self._silence_streak = 0
        # Sample rate / channels in use, populated from the response
        # headers. Exposed in get_state() so the UI can confirm the
        # contract being honoured.
        self._stream_rate = 0
        self._stream_channels = 0
        # Software gain applied to every chunk (post-predicate, before
        # both _push to Keras AND on_pcm_chunk). 1.0 = passthrough.
        # Saturated to int16 in _apply_int16_gain to avoid wrap.
        self._software_gain = float(max(0.0, min(software_gain, 32.0)))
        # Rate limiter para Keras: el worker `audio_detection_worker`
        # asume implícitamente que el audio llega a 44.1 kHz int16 mono
        # (88200 bytes/s) porque `required_bytes` está cableado a ese
        # rate. Si el cliente Android empuja PCM "más rápido que el
        # tiempo real" (p.ej. porque ha buffereado varios segundos y
        # los suelta de golpe), Keras procesaría ventanas de
        # ~1 segundo cada pocos cientos de ms — predicciones erráticas
        # y "array updates mucho más rápido que phone". Para evitarlo
        # NO bloqueamos el thread del bridge (eso impediría sacar
        # chunks del socket); en su lugar **dropamos** chunks que
        # excedan el cupo wall-clock. El cupo es laxo (1.5×) para
        # absorber jitter de red sin perder calidad.
        self._keras_target_bytes_per_s = max(8000,
                                             int(keras_target_bytes_per_s))
        self._clock = clock_fn or time.monotonic
        # Estado del rate limiter (token bucket simple).
        self._keras_bucket_bytes = float(self._keras_target_bytes_per_s)
        self._keras_bucket_last_ts = 0.0
        # Para diagnóstico: cuántos bytes se han descartado por estar
        # por encima del cupo de Keras.
        self._bytes_dropped_keras_rate = 0
        # Aviso único si el PCM HTTP llega saturado mientras el ESP32
        # reporta picos bajos en audio_stats (desync phone_mic vs array).
        self._warned_pcm_saturated = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.is_running():
            return True
        self._stop.clear()
        self._endpoint_path = self._primary_endpoint_path
        self._using_fallback = False
        with self._lock:
            self._bytes_pushed = 0
            self._bytes_dropped_predicate = 0
            self._bytes_dropped_keras_rate = 0
            self._last_rms = 0
            self._last_peak = 0
            self._silence_streak = 0
            # Reset del token bucket. Le damos arranque suave: medio
            # segundo de capacidad para absorber el burst inicial sin
            # dropear nada.
            self._keras_bucket_bytes = float(self._keras_target_bytes_per_s) * 0.5
            self._keras_bucket_last_ts = 0.0
            self._warned_pcm_saturated = False
        self._set_state("starting", "")
        t = threading.Thread(target=self._run, name="array-audio-bridge",
                             daemon=True)
        self._thread = t
        t.start()
        return True

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        self._thread = None
        if t is not None:
            t.join(timeout=join_timeout)
        self._set_state("off", "")

    # ------------------------------------------------------------------
    # State / metrics
    # ------------------------------------------------------------------

    def set_software_gain(self, gain: float) -> None:
        """Update the runtime software gain. Clamped to [0, 32]."""
        with self._lock:
            self._software_gain = float(max(0.0, min(gain, 32.0)))

    def get_state(self) -> dict:
        with self._lock:
            silent = self._silence_streak >= _SILENCE_STREAK_LIMIT
            return {
                "state": self._state,
                "last_error": self._last_error,
                "bytes_pushed": int(self._bytes_pushed),
                "bytes_dropped_predicate": int(self._bytes_dropped_predicate),
                "bytes_dropped_keras_rate": int(self._bytes_dropped_keras_rate),
                "last_chunk_at": float(self._last_chunk_at),
                "running": self.is_running(),
                "last_rms": int(self._last_rms),
                "last_peak": int(self._last_peak),
                "silence": bool(silent),
                "silence_streak": int(self._silence_streak),
                "stream_rate": int(self._stream_rate),
                "stream_channels": int(self._stream_channels),
                "software_gain": float(self._software_gain),
                "keras_target_bytes_per_s": int(self._keras_target_bytes_per_s),
                "endpoint_path": self._endpoint_path,
                "using_fallback": bool(self._using_fallback),
            }

    def _set_state(self, state: str, detail: str) -> None:
        with self._lock:
            self._state = state
            if detail:
                self._last_error = detail
        try:
            self._on_state(state, detail)
        except Exception as e:
            log.warning("array bridge state callback failed: %s", e)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        backoff = _RECONNECT_BACKOFF_S
        while not self._stop.is_set():
            base = (self._base_url_supplier() or "").strip().rstrip("/")
            if not base:
                self._set_state("error", "No base_url for array PCM")
                if self._stop.wait(backoff):
                    return
                continue
            url = f"{base}{self._endpoint_path}"
            try:
                self._set_state("connecting", url)
                with requests.get(url, stream=True,
                                  timeout=_DEFAULT_TIMEOUT) as r:
                    if r.status_code == 404:
                        if (
                            not self._using_fallback
                            and self._fallback_endpoint_path
                            and self._endpoint_path != self._fallback_endpoint_path
                        ):
                            self._using_fallback = True
                            self._endpoint_path = self._fallback_endpoint_path
                            self._on_log(
                                f"{url} no disponible (404); "
                                f"reintentando en {self._endpoint_path} "
                                "(PCM del array vía Android /audio)."
                            )
                            if self._stop.wait(0.05):
                                return
                            continue
                        self._set_state(
                            "not_implemented",
                            "Cliente no expone PCM del array "
                            f"({self._endpoint_path})",
                        )
                        self._on_log(
                            "Array PCM no disponible en primary ni fallback; "
                            "deteniendo bridge."
                        )
                        return
                    if r.status_code != 200:
                        self._set_state(
                            "error", f"HTTP {r.status_code} en {url}"
                        )
                        if self._stop.wait(backoff):
                            return
                        continue

                    rate, channels = self._parse_content_type(r.headers)
                    with self._lock:
                        self._stream_rate = rate
                        self._stream_channels = channels
                    if self._on_stream_meta:
                        try:
                            self._on_stream_meta(rate, channels)
                        except Exception as e:
                            log.warning("on_stream_meta callback failed: %s", e)
                    self._set_state("streaming",
                                    f"{channels}ch @ {rate} Hz")
                    self._on_log(f"Stream activo: {channels}ch @ {rate} Hz")

                    content_type = r.headers.get("Content-Type", "").lower()
                    if "application/x-ndjson" in content_type or \
                            content_type.startswith("application/json"):
                        self._consume_ndjson(r)
                    else:
                        # Default to raw PCM (matches /audio behaviour).
                        self._consume_raw_pcm(r)
            except requests.RequestException as e:
                self._set_state("error", f"{type(e).__name__}: {e}")
                if self._stop.wait(backoff):
                    return
                continue
            except Exception as e:
                self._set_state("error", f"Unexpected: {e}")
                if self._stop.wait(backoff):
                    return
                continue

            if self._stop.is_set():
                return
            # Loop ended cleanly; back off briefly before reconnecting.
            if self._stop.wait(backoff):
                return

    # ------------------------------------------------------------------
    # Body consumers
    # ------------------------------------------------------------------

    def _consume_raw_pcm(self, response: "requests.Response") -> None:
        for chunk in response.iter_content(chunk_size=self._chunk_size):
            if self._stop.is_set():
                return
            if not chunk:
                continue
            self._push(chunk)

    def _consume_ndjson(self, response: "requests.Response") -> None:
        for line in response.iter_lines(decode_unicode=False):
            if self._stop.is_set():
                return
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            b64 = obj.get("b64")
            if not isinstance(b64, str) or not b64:
                continue
            try:
                pcm = base64.b64decode(b64, validate=False)
            except Exception:
                continue
            if pcm:
                self._push(pcm)

    def _push(self, pcm_bytes: bytes) -> None:
        # Chunks HTTP pueden cortar en byte impar; desalinean int16 y
        # producen "ruido" a full-scale en Keras aunque el ESP32 esté bien.
        if len(pcm_bytes) % 2:
            pcm_bytes = pcm_bytes[:-1]
        if not pcm_bytes:
            return

        # 1) Predicate gate: if the user has switched to phone_mic, drop.
        try:
            allowed = bool(self._should_push())
        except Exception:
            allowed = True
        if not allowed:
            with self._lock:
                self._bytes_dropped_predicate += len(pcm_bytes)
            return

        # 2) Software gain (saturated). Pass-through when gain == 1.0.
        gain = self._software_gain
        if gain != 1.0:
            pcm_bytes = _apply_int16_gain(pcm_bytes, gain)

        # 3) Level meter on the (possibly gained) bytes. Update before
        # touching the queue so a downstream `queue.Full` doesn't hide
        # the diagnosis.
        rms, peak = _compute_rms_peak_int16(pcm_bytes)
        with self._lock:
            self._last_rms = rms
            self._last_peak = peak
            if peak < _SILENCE_PEAK:
                self._silence_streak += 1
            else:
                self._silence_streak = 0
            if peak > 8000 and not self._warned_pcm_saturated:
                self._warned_pcm_saturated = True
                self._on_log(
                    f"ADVERTENCIA: PCM en /audio con peak={peak} (saturado). "
                    "Si el heartbeat del ESP32 muestra peak_abs ~500-2000, "
                    "revisa en el móvil que la fuente sea ESP32 ARRAY "
                    "(no micrófono del teléfono): "
                    "curl http://<phone>:8080/adas3/audio-source"
                )

        # 4) Fan out to the playback / external VU hook FIRST so the user
        # can hear / see the audio even if the Keras queue is full.
        if self._on_pcm_chunk is not None:
            try:
                with self._lock:
                    rate = self._stream_rate or _DEFAULT_SAMPLE_RATE
                    channels = self._stream_channels or _DEFAULT_CHANNELS
                self._on_pcm_chunk(pcm_bytes, rate, channels)
            except Exception as e:
                log.warning("on_pcm_chunk callback failed: %s", e)

        # 5) Rate-limit hacia Keras. El worker
        # `audio_detection_worker` espera implícitamente un flujo a
        # ~88200 bytes/s (44.1 kHz int16 mono). Si el Android empuja
        # PCM más rápido que tiempo real (buffer compactado, alta
        # decimación interna, o un rate distinto), Keras procesaría
        # ventanas muchas veces por segundo en lugar de cada 0.5 s →
        # predicciones erráticas. Usamos un token bucket simple: el
        # cupo se rellena a `keras_target_bytes_per_s` y se gasta por
        # cada byte enviado. Si el chunk no cabe en el cupo, lo
        # **dropamos a Keras** (pero el playback y el VU SÍ lo
        # recibieron arriba — el usuario sigue oyendo todo).
        now_ts = self._clock()
        with self._lock:
            if self._keras_bucket_last_ts == 0.0:
                self._keras_bucket_last_ts = now_ts
            elapsed = max(0.0, now_ts - self._keras_bucket_last_ts)
            self._keras_bucket_last_ts = now_ts
            # Rellenar el cupo proporcional al tiempo transcurrido,
            # con un techo de 1.5× el target (= 150 ms de holgura).
            cap = self._keras_target_bytes_per_s * 1.5
            self._keras_bucket_bytes = min(
                cap,
                self._keras_bucket_bytes
                + elapsed * self._keras_target_bytes_per_s,
            )
            chunk_len = len(pcm_bytes)
            if self._keras_bucket_bytes < chunk_len:
                self._bytes_dropped_keras_rate += chunk_len
                return
            self._keras_bucket_bytes -= chunk_len

        # 6) Finally, feed Keras. Drop on overflow to match the phone
        # stream behaviour.
        try:
            self._audio_buffer.put_nowait(pcm_bytes)
        except queue.Full:
            return
        with self._lock:
            self._bytes_pushed += len(pcm_bytes)
            self._last_chunk_at = time.time()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_content_type(headers: Any) -> tuple:
        rate = _DEFAULT_SAMPLE_RATE
        channels = _DEFAULT_CHANNELS
        try:
            ct = headers.get("Content-Type", "")
        except Exception:
            ct = ""
        for part in str(ct).split(";"):
            part = part.strip().lower()
            if part.startswith("rate="):
                try:
                    rate = int(part.split("=", 1)[1])
                except (ValueError, IndexError):
                    rate = _DEFAULT_SAMPLE_RATE
            elif part.startswith("channels="):
                try:
                    channels = int(part.split("=", 1)[1])
                except (ValueError, IndexError):
                    channels = _DEFAULT_CHANNELS
        rate = max(8000, min(rate, 96000))
        channels = max(1, min(channels, 2))
        return rate, channels


__all__ = ["ArrayAudioBridge", "_compute_rms_peak_int16", "_apply_int16_gain"]
