"""Tests para `modules.playback_router.PlaybackRouter`.

El router es el punto único de playback PyAudio para todas las fuentes
(phone_mic, esp32_array). Esto verifica que:

  - Una sola apertura del stream cuando varios "escritores" coexisten
    al mismo rate/channels (regresión: dos pa.open simultáneos dejaban
    mudas a ambas fuentes).
  - Cambio de rate/channels reabre exactamente una vez.
  - Mute global suprime writes sin afectar la contabilidad de
    "intentos".
  - Fallo en `stream.write` provoca cierre y reapertura en el siguiente
    chunk (en vez de quedar mudo permanentemente).
"""

from __future__ import annotations

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.playback_router import PlaybackRouter  # noqa: E402


class _FakeStream:
    def __init__(self, rate, channels, fail_on=None):
        self.rate = rate
        self.channels = channels
        self.writes = []
        self.closed = False
        self._fail_on = set(fail_on or ())
        self._counter = 0

    def write(self, chunk):
        self._counter += 1
        if self._counter in self._fail_on:
            raise OSError("simulated device error")
        self.writes.append(bytes(chunk))

    def stop_stream(self):
        pass

    def close(self):
        self.closed = True


class _FakePyAudio:
    def __init__(self, fail_on_writes_for_stream_index=None):
        self.streams = []
        self._fail_on = fail_on_writes_for_stream_index or {}

    def open(self, format, channels, rate, output, frames_per_buffer):
        idx = len(self.streams)
        fail_on = self._fail_on.get(idx)
        stream = _FakeStream(rate, channels, fail_on=fail_on)
        self.streams.append(stream)
        return stream


class _FakePyAudioOpenFail:
    """pa.open raises always — simulates no audio device available."""

    def open(self, *args, **kwargs):
        raise RuntimeError("no audio device")


def _make_router(pa, **kwargs):
    return PlaybackRouter(
        pyaudio_factory=lambda: pa,
        audio_format=8,  # arbitrary placeholder, fake doesn't use it
        frames_per_buffer=512,
        on_log=lambda _msg: None,
        **kwargs,
    )


class TestRouterSingleStream(unittest.TestCase):
    def test_writes_from_two_sources_share_one_stream(self):
        """phone_mic y esp32_array escribiendo al mismo rate/channels
        NO deben provocar dos pa.open. Si esto regresa, PortAudio se
        quedará mudo en una de las dos fuentes."""
        pa = _FakePyAudio()
        router = _make_router(pa)
        # Simula varios chunks alternados de dos fuentes:
        for i in range(5):
            router.write_chunk(b"phone" + bytes([i]), 44100, 1)
            router.write_chunk(b"array" + bytes([i]), 44100, 1)
        self.assertEqual(len(pa.streams), 1)
        self.assertEqual(len(pa.streams[0].writes), 10)

    def test_format_change_reopens_stream_once(self):
        pa = _FakePyAudio()
        router = _make_router(pa)
        router.write_chunk(b"\x01\x02", 44100, 1)
        router.write_chunk(b"\x03\x04", 16000, 1)
        router.write_chunk(b"\x05\x06", 16000, 1)
        self.assertEqual(len(pa.streams), 2)
        self.assertEqual(pa.streams[0].rate, 44100)
        self.assertEqual(pa.streams[1].rate, 16000)
        # El stream antiguo debe quedar cerrado.
        self.assertTrue(pa.streams[0].closed)


class TestRouterMute(unittest.TestCase):
    def test_muted_drops_writes_and_does_not_open_stream(self):
        pa = _FakePyAudio()
        router = _make_router(pa)
        router.set_muted(True)
        ok1 = router.write_chunk(b"x" * 100, 44100, 1)
        ok2 = router.write_chunk(b"y" * 100, 44100, 1)
        self.assertFalse(ok1)
        self.assertFalse(ok2)
        # Si está muteado desde el principio NO debería abrir el stream.
        self.assertEqual(len(pa.streams), 0)
        state = router.get_state()
        self.assertEqual(state["chunks_dropped_muted"], 2)
        self.assertEqual(state["chunks_written"], 0)

    def test_unmute_resumes_playback(self):
        pa = _FakePyAudio()
        router = _make_router(pa)
        router.set_muted(True)
        router.write_chunk(b"\x00" * 16, 44100, 1)
        router.set_muted(False)
        ok = router.write_chunk(b"\x10" * 16, 44100, 1)
        self.assertTrue(ok)
        self.assertEqual(len(pa.streams), 1)
        self.assertEqual(len(pa.streams[0].writes), 1)


class TestRouterRobustness(unittest.TestCase):
    def test_write_failure_triggers_reopen_on_next_chunk(self):
        """Si el dispositivo se cae (write OSError), el router debe
        cerrar el stream defectuoso y abrir uno nuevo en el siguiente
        chunk en vez de quedar mudo para siempre."""
        # Hacer fallar el primer write del primer stream (counter==1).
        pa = _FakePyAudio(fail_on_writes_for_stream_index={0: [1]})
        router = _make_router(pa)
        ok1 = router.write_chunk(b"\xAA" * 8, 44100, 1)
        ok2 = router.write_chunk(b"\xBB" * 8, 44100, 1)
        self.assertFalse(ok1)
        self.assertTrue(ok2)
        # Deben haberse abierto dos streams: el primero (fallido) y el
        # segundo (que ya escribe sin error porque sólo el counter==1
        # del primer stream falla).
        self.assertEqual(len(pa.streams), 2)
        self.assertTrue(pa.streams[0].closed)

    def test_open_failure_returns_false_and_does_not_crash(self):
        router = _make_router(_FakePyAudioOpenFail())
        ok = router.write_chunk(b"\x01" * 8, 44100, 1)
        self.assertFalse(ok)
        state = router.get_state()
        self.assertFalse(state["stream_open"])
        self.assertEqual(state["chunks_written"], 0)

    def test_empty_chunk_is_dropped_silently(self):
        pa = _FakePyAudio()
        router = _make_router(pa)
        self.assertFalse(router.write_chunk(b"", 44100, 1))
        self.assertEqual(len(pa.streams), 0)

    def test_close_idempotent(self):
        pa = _FakePyAudio()
        router = _make_router(pa)
        router.write_chunk(b"\x00" * 8, 44100, 1)
        router.close()
        router.close()
        self.assertTrue(pa.streams[0].closed)


class TestRouterConcurrency(unittest.TestCase):
    def test_concurrent_writers_open_single_stream(self):
        """Phone y array empujando concurrentemente: el lock interno
        debe garantizar que sólo se abra UN stream, no uno por thread."""
        pa = _FakePyAudio()
        router = _make_router(pa)
        errors = []

        def writer(tag):
            try:
                for i in range(50):
                    router.write_chunk(tag + bytes([i & 0xFF]), 44100, 1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(b"P",)),
                   threading.Thread(target=writer, args=(b"A",))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(errors, [])
        self.assertEqual(len(pa.streams), 1)
        self.assertEqual(len(pa.streams[0].writes), 100)


class TestRouterBothSourcesScenario(unittest.TestCase):
    """Simula el caso real: la fuente cambia de phone -> array y
    vuelve. El router debe servir a las dos sin abrir streams
    paralelos. Verifica también que muting global durante array y
    luego unmute durante phone se comporta como espera el usuario."""

    def test_phone_then_array_then_phone(self):
        pa = _FakePyAudio()
        router = _make_router(pa)
        # Phone path emite a 44100/1
        for _ in range(3):
            router.write_chunk(b"P" * 16, 44100, 1)
        # Cambio a array (mismo rate/channels — no debería reabrir).
        for _ in range(3):
            router.write_chunk(b"A" * 16, 44100, 1)
        # Vuelta a phone
        for _ in range(3):
            router.write_chunk(b"P" * 16, 44100, 1)
        self.assertEqual(len(pa.streams), 1)
        self.assertEqual(len(pa.streams[0].writes), 9)

    def test_mute_affects_both_sources(self):
        pa = _FakePyAudio()
        router = _make_router(pa)
        router.write_chunk(b"phone", 44100, 1)  # phone unmuted
        router.set_muted(True)
        router.write_chunk(b"phone", 44100, 1)  # phone muted, dropped
        router.write_chunk(b"array", 44100, 1)  # array muted, dropped
        router.set_muted(False)
        router.write_chunk(b"array", 44100, 1)  # array unmuted, written
        self.assertEqual(len(pa.streams), 1)
        self.assertEqual(len(pa.streams[0].writes), 2)
        state = router.get_state()
        self.assertEqual(state["chunks_dropped_muted"], 2)
        self.assertEqual(state["chunks_written"], 2)


class _FakeStreamFailingFormat(_FakeStream):
    """No falla por sample rate, sólo registra."""
    pass


class _FakePyAudioPickyRate:
    """pa.open lanza si rate != 48000. Simula tarjeta que sólo abre a
    una tasa fija (ALSA dmix con sample rate fijo a menudo se comporta
    así)."""

    def __init__(self, accepted_rate=48000, accepted_channels=1):
        self.streams = []
        self.attempts = []
        self._rate = accepted_rate
        self._channels = accepted_channels

    def open(self, format, channels, rate, output, frames_per_buffer):
        self.attempts.append((rate, channels))
        if rate != self._rate or channels != self._channels:
            raise OSError(
                f"unsupported format rate={rate} channels={channels}"
            )
        s = _FakeStreamFailingFormat(rate, channels)
        self.streams.append(s)
        return s


class TestRouterFormatFallback(unittest.TestCase):
    """Cuando el rate/channels pedidos fallan, el router debe probar
    formatos comunes antes de rendirse. Esto es exactamente lo que
    necesita el caso "Keras detecta pero no se oye nada": el primer
    `pa.open` con 44100/1 puede estar fallando sin log, y antes el
    router devolvía None silenciosamente."""

    def test_falls_back_to_48000_when_44100_unsupported(self):
        pa = _FakePyAudioPickyRate(accepted_rate=48000, accepted_channels=1)
        logs = []
        router = PlaybackRouter(
            pyaudio_factory=lambda: pa,
            audio_format=8,
            frames_per_buffer=512,
            on_log=lambda m: logs.append(m),
        )
        ok = router.write_chunk(b"\x00" * 64, 44100, 1)
        self.assertTrue(ok)
        # Intentó 44100/1 primero, falló, y abrió 48000/1.
        self.assertIn((44100, 1), pa.attempts)
        self.assertIn((48000, 1), pa.attempts)
        self.assertEqual(pa.streams[0].rate, 48000)
        # Logs incluyen tanto el fallo del primario como el éxito del fallback.
        self.assertTrue(
            any("fallback" in m for m in logs),
            f"logs sin mensaje de fallback: {logs}",
        )

    def test_total_open_failure_logs_clearly_and_does_not_crash(self):
        class _AllFail:
            def open(self, **kw):
                raise OSError("device busy")
        logs = []
        router = PlaybackRouter(
            pyaudio_factory=lambda: _AllFail(),
            audio_format=8,
            frames_per_buffer=512,
            on_log=lambda m: logs.append(m),
        )
        ok = router.write_chunk(b"\xAA" * 64, 44100, 1)
        self.assertFalse(ok)
        # Debe existir un mensaje claro tras agotar los fallbacks.
        self.assertTrue(
            any("NINGÚN stream" in m for m in logs),
            f"logs sin mensaje de fallo total: {logs}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
