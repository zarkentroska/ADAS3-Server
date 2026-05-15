"""Unit tests for modules.array_audio_bridge.ArrayAudioBridge.

We avoid real network IO by monkey-patching ``requests.get`` inside the
module. The bridge runs in a daemon thread; tests start/stop it
explicitly and inspect the shared ``audio_buffer`` queue.
"""

import base64
import json
import os
import queue
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import array_audio_bridge as aab  # noqa: E402


class _StubResponse:
    """Minimal stand-in for ``requests.Response`` with a controlled
    body. Closes when exited from a ``with`` block."""

    def __init__(self, status_code=200, headers=None, body=b"",
                 ndjson_lines=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self._ndjson_lines = ndjson_lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def iter_content(self, chunk_size=1024):
        # Yield the body in `chunk_size` slices, then stop.
        b = self._body
        i = 0
        while i < len(b):
            yield b[i:i + chunk_size]
            i += chunk_size

    def iter_lines(self, decode_unicode=False):
        for line in (self._ndjson_lines or []):
            if isinstance(line, str):
                yield line.encode("utf-8")
            else:
                yield line


class TestArrayAudioBridgeRawPcm(unittest.TestCase):
    def _make_bridge(self, audio_buffer):
        return aab.ArrayAudioBridge(
            base_url_supplier=lambda: "http://phone:8080",
            audio_buffer=audio_buffer,
            chunk_size=512,
        )

    def test_404_on_primary_falls_back_to_audio_and_streams(self):
        audio_buffer = queue.Queue(maxsize=8)
        bridge = self._make_bridge(audio_buffer)
        pcm = b"\x10\x20" * 512
        primary_404 = _StubResponse(status_code=404,
                                    headers={"Content-Type": "text/plain"})
        audio_ok = _StubResponse(
            status_code=200,
            headers={"Content-Type": "audio/pcm; rate=44100; channels=1"},
            body=pcm,
        )
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            if "/adas3/mic-array/pcm" in url:
                return primary_404
            if url.endswith("/audio"):
                return audio_ok
            return primary_404

        with patch.object(aab.requests, "get", side_effect=fake_get):
            bridge.start()
            deadline = time.time() + 2.0
            collected = 0
            while time.time() < deadline:
                try:
                    chunk = audio_buffer.get(timeout=0.2)
                    collected += len(chunk)
                    if collected >= len(pcm):
                        break
                except queue.Empty:
                    pass
            bridge.stop()
        self.assertTrue(
            any("/adas3/mic-array/pcm" in u for u in calls),
            "debe intentar el endpoint primary",
        )
        self.assertTrue(
            any(u.endswith("/audio") for u in calls),
            "debe hacer fallback a /audio tras 404",
        )
        state = bridge.get_state()
        self.assertTrue(state["using_fallback"])
        self.assertEqual(state["endpoint_path"], "/audio")
        self.assertGreaterEqual(collected, len(pcm))

    def test_404_on_primary_and_fallback_marks_not_implemented(self):
        audio_buffer = queue.Queue(maxsize=4)
        bridge = self._make_bridge(audio_buffer)
        resp = _StubResponse(status_code=404,
                             headers={"Content-Type": "text/plain"})
        with patch.object(aab.requests, "get", return_value=resp):
            bridge.start()
            deadline = time.time() + 2.0
            while bridge.is_running() and time.time() < deadline:
                time.sleep(0.05)
            self.assertFalse(bridge.is_running())
        state = bridge.get_state()
        self.assertEqual(state["state"], "not_implemented")

    def test_raw_pcm_pushed_to_buffer(self):
        audio_buffer = queue.Queue(maxsize=8)
        bridge = self._make_bridge(audio_buffer)
        pcm = b"\x01\x02\x03\x04" * 256  # 1024 bytes (int16 LE garbage)
        resp = _StubResponse(
            status_code=200,
            headers={"Content-Type": "audio/pcm; rate=44100; channels=1"},
            body=pcm,
        )
        # After the body is exhausted the worker will loop & reconnect.
        # Patch get to return the same body once then a 404 to terminate.
        get_call_count = {"n": 0}

        def fake_get(url, **kwargs):
            get_call_count["n"] += 1
            if get_call_count["n"] == 1:
                return resp
            return _StubResponse(status_code=404,
                                 headers={"Content-Type": "text/plain"})

        with patch.object(aab.requests, "get", side_effect=fake_get):
            bridge.start()
            deadline = time.time() + 2.0
            collected = 0
            while time.time() < deadline:
                try:
                    chunk = audio_buffer.get(timeout=0.2)
                    collected += len(chunk)
                    if collected >= len(pcm):
                        break
                except queue.Empty:
                    pass
            bridge.stop()
        self.assertGreaterEqual(collected, len(pcm))
        state = bridge.get_state()
        self.assertGreaterEqual(state["bytes_pushed"], len(pcm))

    def test_content_type_clamps_rate_and_channels(self):
        rate, channels = aab.ArrayAudioBridge._parse_content_type(
            {"Content-Type": "audio/pcm; rate=999999; channels=8"}
        )
        self.assertEqual(rate, 96000)
        self.assertEqual(channels, 2)

        rate, channels = aab.ArrayAudioBridge._parse_content_type(
            {"Content-Type": "audio/pcm; rate=1000; channels=0"}
        )
        self.assertEqual(rate, 8000)
        self.assertEqual(channels, 1)

    def test_meta_callback_invoked(self):
        audio_buffer = queue.Queue(maxsize=4)
        meta_events = []
        bridge = aab.ArrayAudioBridge(
            base_url_supplier=lambda: "http://phone:8080",
            audio_buffer=audio_buffer,
            on_stream_meta=lambda r, c: meta_events.append((r, c)),
        )
        resp = _StubResponse(
            status_code=200,
            headers={"Content-Type": "audio/pcm; rate=16000; channels=1"},
            body=b"\x00\x01\x00\x02",
        )

        def fake_get(url, **kwargs):
            # First call returns 200 + body, second returns 404 to stop.
            if not meta_events:
                return resp
            return _StubResponse(status_code=404,
                                 headers={"Content-Type": "text/plain"})

        with patch.object(aab.requests, "get", side_effect=fake_get):
            bridge.start()
            deadline = time.time() + 1.5
            while not meta_events and time.time() < deadline:
                time.sleep(0.05)
            bridge.stop()
        self.assertEqual(meta_events[:1], [(16000, 1)])


class TestArrayAudioBridgeNdjson(unittest.TestCase):
    def test_ndjson_base64_chunks_decoded(self):
        audio_buffer = queue.Queue(maxsize=8)
        bridge = aab.ArrayAudioBridge(
            base_url_supplier=lambda: "http://phone:8080",
            audio_buffer=audio_buffer,
        )
        pcm1 = b"\x10\x00\x20\x00"
        pcm2 = b"\x30\x00\x40\x00"
        lines = [
            json.dumps({"b64": base64.b64encode(pcm1).decode()}),
            json.dumps({"b64": base64.b64encode(pcm2).decode()}),
            json.dumps({"no_b64_here": True}),  # ignored
        ]
        resp = _StubResponse(
            status_code=200,
            headers={"Content-Type": "application/x-ndjson; rate=44100; channels=1"},
            ndjson_lines=lines,
        )

        call_count = {"n": 0}

        def fake_get(url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return resp
            return _StubResponse(status_code=404,
                                 headers={"Content-Type": "text/plain"})

        with patch.object(aab.requests, "get", side_effect=fake_get):
            bridge.start()
            deadline = time.time() + 1.5
            collected = b""
            while time.time() < deadline and len(collected) < 8:
                try:
                    collected += audio_buffer.get(timeout=0.2)
                except queue.Empty:
                    pass
            bridge.stop()
        self.assertIn(pcm1, collected)
        self.assertIn(pcm2, collected)


class TestShouldPushGate(unittest.TestCase):
    """Defensa contra dual-stream: si el predicado should_push devuelve
    False, el bridge no debe meter chunks en el audio_buffer compartido
    (los descarta y los cuenta en bytes_dropped_predicate)."""

    def test_should_push_blocks_pcm_chunks(self):
        audio_buffer = queue.Queue(maxsize=4)
        gate = {"open": False}

        pcm = b"\x00\x10" * 200  # 400 bytes raw PCM16
        resp = _StubResponse(
            status_code=200,
            headers={"Content-Type": "audio/pcm; rate=44100; channels=1"},
            body=pcm,
        )

        bridge = aab.ArrayAudioBridge(
            base_url_supplier=lambda: "http://phone:8080",
            audio_buffer=audio_buffer,
            chunk_size=64,
            should_push=lambda: gate["open"],
        )

        # Mismo patrón que test_ndjson: una primera respuesta con datos
        # y luego 404 para que el bridge salga limpiamente.
        call_count = {"n": 0}

        def fake_get(url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return resp
            return _StubResponse(status_code=404,
                                 headers={"Content-Type": "text/plain"})

        with patch.object(aab.requests, "get", side_effect=fake_get):
            bridge.start()
            deadline = time.time() + 1.5
            while bridge.is_running() and time.time() < deadline:
                time.sleep(0.05)
            bridge.stop()

        # Nada en el buffer: el gate estaba cerrado.
        self.assertEqual(audio_buffer.qsize(), 0)
        state = bridge.get_state()
        # Debe haber contado lo que descartó.
        self.assertGreater(state["bytes_dropped_predicate"], 0)
        self.assertEqual(state["bytes_pushed"], 0)

    def test_should_push_open_passes_pcm_chunks(self):
        audio_buffer = queue.Queue(maxsize=16)

        pcm = b"\x01\x02" * 100
        resp = _StubResponse(
            status_code=200,
            headers={"Content-Type": "audio/pcm; rate=44100; channels=1"},
            body=pcm,
        )

        bridge = aab.ArrayAudioBridge(
            base_url_supplier=lambda: "http://phone:8080",
            audio_buffer=audio_buffer,
            chunk_size=64,
            should_push=lambda: True,
        )

        call_count = {"n": 0}

        def fake_get(url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return resp
            return _StubResponse(status_code=404,
                                 headers={"Content-Type": "text/plain"})

        with patch.object(aab.requests, "get", side_effect=fake_get):
            bridge.start()
            deadline = time.time() + 1.5
            collected = b""
            while time.time() < deadline and len(collected) < len(pcm):
                try:
                    collected += audio_buffer.get(timeout=0.2)
                except queue.Empty:
                    pass
            bridge.stop()

        self.assertGreaterEqual(len(collected), 50)
        state = bridge.get_state()
        self.assertGreater(state["bytes_pushed"], 0)
        self.assertEqual(state["bytes_dropped_predicate"], 0)


class TestLevelMeterAndGain(unittest.TestCase):
    """Las nuevas señales de diagnóstico que distinguen
    'bytes_pushed' de 'señal audible'."""

    def test_rms_peak_of_silent_buffer_is_zero(self):
        rms, peak = aab._compute_rms_peak_int16(b"\x00\x00" * 1024)
        self.assertEqual(rms, 0)
        self.assertEqual(peak, 0)

    def test_rms_peak_of_full_scale_buffer(self):
        # int16 LE 32767 = 0xFF 0x7F repetido.
        import struct
        pcm = struct.pack("<" + "h" * 1000, *([32767] * 1000))
        rms, peak = aab._compute_rms_peak_int16(pcm)
        self.assertEqual(peak, 32767)
        self.assertGreater(rms, 30000)

    def test_software_gain_saturates_int16(self):
        import struct
        pcm = struct.pack("<h", 20000)  # un solo sample
        # gain 4 saturaría a 80000 -> clip a 32767
        out = aab._apply_int16_gain(pcm, 4.0)
        v = struct.unpack("<h", out)[0]
        self.assertEqual(v, 32767)

    def test_software_gain_passthrough(self):
        pcm = b"\x10\x20" * 100
        self.assertEqual(aab._apply_int16_gain(pcm, 1.0), pcm)

    def test_bridge_exposes_level_state_after_chunk(self):
        """get_state() debe reportar rms/peak/silence tras procesar un
        chunk no silente."""
        import struct
        audio_buffer = queue.Queue(maxsize=8)
        pcm = struct.pack("<" + "h" * 800, *([5000] * 800))
        resp = _StubResponse(
            status_code=200,
            headers={"Content-Type": "audio/pcm; rate=44100; channels=1"},
            body=pcm,
        )
        bridge = aab.ArrayAudioBridge(
            base_url_supplier=lambda: "http://phone:8080",
            audio_buffer=audio_buffer,
            chunk_size=200,
        )
        call_count = {"n": 0}

        def fake_get(url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return resp
            return _StubResponse(status_code=404,
                                 headers={"Content-Type": "text/plain"})

        with patch.object(aab.requests, "get", side_effect=fake_get):
            bridge.start()
            deadline = time.time() + 1.5
            while bridge.is_running() and time.time() < deadline:
                time.sleep(0.05)
            bridge.stop()
        state = bridge.get_state()
        self.assertEqual(state["last_peak"], 5000)
        self.assertGreater(state["last_rms"], 4000)
        self.assertFalse(state["silence"])
        self.assertEqual(state["stream_rate"], 44100)
        self.assertEqual(state["stream_channels"], 1)

    def test_on_pcm_chunk_callback_is_invoked(self):
        """El nuevo hook on_pcm_chunk se llama por cada chunk, con
        (chunk, rate, channels). Permite playback externo."""
        audio_buffer = queue.Queue(maxsize=8)
        pcm = b"\xAA\x55" * 256
        resp = _StubResponse(
            status_code=200,
            headers={"Content-Type": "audio/pcm; rate=16000; channels=1"},
            body=pcm,
        )
        seen = []
        bridge = aab.ArrayAudioBridge(
            base_url_supplier=lambda: "http://phone:8080",
            audio_buffer=audio_buffer,
            chunk_size=128,
            on_pcm_chunk=lambda chunk, rate, channels: seen.append(
                (len(chunk), rate, channels)
            ),
        )
        call_count = {"n": 0}

        def fake_get(url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return resp
            return _StubResponse(status_code=404,
                                 headers={"Content-Type": "text/plain"})

        with patch.object(aab.requests, "get", side_effect=fake_get):
            bridge.start()
            deadline = time.time() + 1.5
            while bridge.is_running() and time.time() < deadline:
                time.sleep(0.05)
            bridge.stop()
        self.assertGreater(len(seen), 0)
        # Cada entrada debe llevar rate y channels coherentes con la cab.
        for _len, rate, channels in seen:
            self.assertEqual(rate, 16000)
            self.assertEqual(channels, 1)


class TestEsp32ArrayKerasPath(unittest.TestCase):
    """Checklist Parte C (automático): PCM del array debe superar los gates
    de silencio del bridge y del worker Keras cuando hay señal útil."""

    def test_default_fallback_endpoint_is_audio(self):
        buf = queue.Queue()
        bridge = aab.ArrayAudioBridge(
            base_url_supplier=lambda: "http://x",
            audio_buffer=buf,
        )
        self.assertEqual(bridge._fallback_endpoint_path, "/audio")
        self.assertEqual(bridge._primary_endpoint_path, "/adas3/mic-array/pcm")

    def test_drone_like_pcm_passes_bridge_and_keras_silence_gates(self):
        # 440 Hz tone @ 44100 Hz, amplitude ~8000 — similar to palmada/speech.
        import struct
        import math

        n = 44100
        samples = [
            int(8000 * math.sin(2 * math.pi * 440 * i / 44100))
            for i in range(n)
        ]
        pcm = struct.pack("<" + "h" * n, *samples)
        rms, peak = aab._compute_rms_peak_int16(pcm)
        self.assertGreater(peak, aab._SILENCE_PEAK,
                           "bridge silence gate (pk>=200)")
        mean_abs = sum(abs(s) for s in samples) / n
        keras_gate = float(os.environ.get("ADAS3_AUDIO_SILENCE_GATE", "30"))
        self.assertGreaterEqual(mean_abs, keras_gate,
                                "Keras silence gate (mean_abs)")


class TestKerasRateLimit(unittest.TestCase):
    """Token bucket en `ArrayAudioBridge._push`.

    El motivo de existir es que `audio_detection_worker` asume audio
    a ~88200 bytes/s (44.1 kHz int16). Si el Android empuja más
    rápido que tiempo real, el bridge dropa el exceso hacia Keras
    PERO conserva el camino de playback (on_pcm_chunk) intacto."""

    def _make_bridge(self, on_pcm_chunk, clock_seq, target=8000):
        # `clock_seq` se itera; cada llamada a `_clock()` consume uno.
        # Si se agota, devolvemos el último valor (clock detenido).
        state = {"i": 0}

        def fake_clock():
            i = state["i"]
            if i >= len(clock_seq):
                return clock_seq[-1]
            v = clock_seq[i]
            state["i"] = i + 1
            return v

        return aab.ArrayAudioBridge(
            base_url_supplier=lambda: "http://phone:8080",
            audio_buffer=queue.Queue(maxsize=64),
            chunk_size=200,
            on_pcm_chunk=on_pcm_chunk,
            keras_target_bytes_per_s=target,
            clock_fn=fake_clock,
        )

    def test_chunks_within_budget_pass(self):
        """En tiempo real (1 byte/s con cupo 8000 b/s) todos pasan."""
        played = []
        # Cada chunk son 100 bytes; 100 ms entre chunks → 1000 b/s,
        # muy por debajo del cupo de 8000.
        b = self._make_bridge(
            on_pcm_chunk=lambda c, r, ch: played.append(len(c)),
            clock_seq=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            target=8000,
        )
        # Simular bucket inicial razonable.
        with b._lock:
            b._keras_bucket_bytes = 4000.0
        for _ in range(5):
            b._push(b"\x10" * 100)
        # Todos los chunks fueron a Keras Y a playback.
        self.assertEqual(b._audio_buffer.qsize(), 5)
        self.assertEqual(played, [100] * 5)
        self.assertEqual(b.get_state()["bytes_dropped_keras_rate"], 0)

    def test_burst_above_budget_drops_excess_to_keras_not_playback(self):
        """Burst: 10 chunks de 2000 bytes en 0 ms entre cada uno con
        target 8000 b/s. Sólo deberían pasar a Keras los que caben en
        el cupo (~bucket inicial); el resto se dropa. PERO los 10
        deben llegar a playback."""
        played = []
        b = self._make_bridge(
            on_pcm_chunk=lambda c, r, ch: played.append(len(c)),
            # Todos los _push ocurren en el mismo instante (clock no
            # avanza), así que el bucket no se rellena.
            clock_seq=[0.0] * 30,
            target=8000,
        )
        with b._lock:
            # Bucket inicial = target * 0.5 según `start()`.
            b._keras_bucket_bytes = 4000.0
        for _ in range(10):
            b._push(b"\xAB" * 2000)
        # Playback recibió los 10.
        self.assertEqual(len(played), 10)
        # Keras recibió como mucho 2 (4000 / 2000), el resto dropeado.
        state = b.get_state()
        self.assertLessEqual(b._audio_buffer.qsize(), 2)
        self.assertGreater(state["bytes_dropped_keras_rate"], 0)

    def test_get_state_exposes_target_and_drops(self):
        b = self._make_bridge(
            on_pcm_chunk=None,
            clock_seq=[0.0, 0.0],
            target=88200,
        )
        state = b.get_state()
        self.assertEqual(state["keras_target_bytes_per_s"], 88200)
        self.assertIn("bytes_dropped_keras_rate", state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
