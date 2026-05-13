"""Unit tests for modules.esp32_acoustic_array.

These tests do not require pyserial or any real hardware. They cover:

  - parse_message: JSON happy path, JSON malformed, CSV legacy, empty lines
  - AcousticArrayClient with simulation transport: starts, dispatches events,
    state snapshot, debounce, clean shutdown
"""

import os
import sys
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.esp32_acoustic_array import (  # noqa: E402
    AcousticArrayClient,
    AcousticArrayConfig,
    parse_message,
)


class TestParser(unittest.TestCase):
    def test_json_heartbeat(self):
        msg = parse_message(
            '{"type":"heartbeat","mic_count":4,"firmware":"adas-array-0.1"}'
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg["type"], "heartbeat")
        self.assertEqual(msg["mic_count"], 4)
        self.assertEqual(msg["firmware"], "adas-array-0.1")

    def test_json_acoustic(self):
        line = (
            '{"type":"acoustic","detected":true,"doa_deg":35.0,'
            '"energy":0.72,"confidence":0.81,"mic_count":4}'
        )
        msg = parse_message(line)
        self.assertIsNotNone(msg)
        self.assertEqual(msg["type"], "acoustic")
        self.assertTrue(msg["detected"])
        self.assertAlmostEqual(msg["doa_deg"], 35.0)
        self.assertAlmostEqual(msg["energy"], 0.72)
        self.assertAlmostEqual(msg["confidence"], 0.81)
        self.assertEqual(msg["mic_count"], 4)

    def test_json_inferred_type(self):
        msg = parse_message('{"doa_deg":10.0,"energy":0.5,"confidence":0.6,"detected":false}')
        self.assertIsNotNone(msg)
        self.assertEqual(msg["type"], "acoustic")

    def test_json_malformed(self):
        # Truncated JSON: hard parse failure
        self.assertIsNone(parse_message('{"type":"acoustic"'))
        # Non-dict top level: rejected
        self.assertIsNone(parse_message('[1,2,3]'))
        # Garbled CSV-like content: fails cleanly
        self.assertIsNone(parse_message('abcdef'))
        # Dict with unrecognised type and no acoustic fields: parsed as "unknown"
        msg = parse_message('{"type":42}')
        self.assertIsNotNone(msg)
        self.assertEqual(msg["type"], "unknown")

    def test_csv_legacy(self):
        msg = parse_message('35.0,0.72,0.81,1')
        self.assertIsNotNone(msg)
        self.assertEqual(msg["type"], "acoustic")
        self.assertAlmostEqual(msg["doa_deg"], 35.0)
        self.assertAlmostEqual(msg["energy"], 0.72)
        self.assertAlmostEqual(msg["confidence"], 0.81)
        self.assertTrue(msg["detected"])

    def test_csv_no_detected_flag(self):
        msg = parse_message('-10.5,0.20,0.40')
        self.assertIsNotNone(msg)
        self.assertEqual(msg["type"], "acoustic")
        self.assertFalse(msg["detected"])

    def test_empty_and_none(self):
        self.assertIsNone(parse_message(""))
        self.assertIsNone(parse_message("   \n"))
        self.assertIsNone(parse_message(None))

    def test_csv_bad_numbers(self):
        self.assertIsNone(parse_message('abc,def,ghi,1'))


class TestSimulatedClient(unittest.TestCase):
    def _make_cfg(self):
        return AcousticArrayConfig(
            enabled=True,
            transport="simulation",
            sim_heartbeat_period_s=0.1,
            sim_detection_period_s=0.3,
            detection_debounce_s=0.1,
            energy_threshold=0.0,
            confidence_threshold=0.0,
            smoothing_alpha=0.0,
            reconnect_delay_s=0.1,
        )

    def test_lifecycle_and_state(self):
        client = AcousticArrayClient(self._make_cfg())
        events = []
        ev_lock = threading.Lock()

        def cb(event_type, payload, snapshot):
            with ev_lock:
                events.append((event_type, dict(payload), snapshot))

        client.on_event(cb)
        self.assertTrue(client.start())
        try:
            # Wait until we accumulate at least a connected + a heartbeat + a detection
            deadline = time.monotonic() + 5.0
            saw = {"connected": False, "heartbeat": False, "detection": False}
            while time.monotonic() < deadline:
                with ev_lock:
                    for name, _, _ in events:
                        if name in saw:
                            saw[name] = True
                if all(saw.values()):
                    break
                time.sleep(0.05)
            self.assertTrue(saw["connected"], "no connected event")
            self.assertTrue(saw["heartbeat"], "no heartbeat event")
            self.assertTrue(saw["detection"], "no detection event")

            st = client.get_state()
            self.assertTrue(st.connected)
            self.assertGreaterEqual(st.messages_received, 1)
            self.assertEqual(st.mic_count, 4)
        finally:
            client.stop(join_timeout=2.0)
        self.assertFalse(client.is_running())

    def test_detection_debounce(self):
        cfg = self._make_cfg()
        cfg.sim_detection_period_s = 0.05  # rapid detections
        cfg.detection_debounce_s = 0.5
        client = AcousticArrayClient(cfg)
        det_count = {"n": 0}
        lock = threading.Lock()

        def cb(event_type, payload, snapshot):
            if event_type == "detection":
                with lock:
                    det_count["n"] += 1

        client.on_event(cb)
        client.start()
        try:
            time.sleep(1.2)  # ~24 acoustic events expected, but at most ~3 detections
        finally:
            client.stop()
        # With debounce 0.5s over ~1.2s we should see at most 3-4 detection callbacks
        with lock:
            n = det_count["n"]
        self.assertGreaterEqual(n, 1, "no detection at all")
        self.assertLessEqual(n, 4, f"debounce did not work, got {n} detections")

    def test_disabled_config_does_not_start(self):
        cfg = self._make_cfg()
        cfg.enabled = False
        client = AcousticArrayClient(cfg)
        self.assertFalse(client.start())
        self.assertFalse(client.is_running())

    def test_short_sim_detection_period_is_honoured(self):
        """Regression: simulator must not silently clamp short periods to 1s.

        The smoke test of the README runs ~1.0s with det_period=0.2s and
        expects multiple alerts. The earlier implementation floored
        det_period at 1.0s, so the first detection landed exactly at the
        end of the sleep window and the user observed alerts=0.
        """
        cfg = AcousticArrayConfig(
            enabled=True,
            transport="simulation",
            sim_heartbeat_period_s=0.1,
            sim_detection_period_s=0.2,
            detection_debounce_s=0.05,
            energy_threshold=0.0,
            confidence_threshold=0.0,
            smoothing_alpha=0.0,
            reconnect_delay_s=0.1,
        )
        client = AcousticArrayClient(cfg)
        det_n = {"n": 0}
        lock = threading.Lock()

        def cb(event_type, payload, snapshot):
            if event_type == "detection":
                with lock:
                    det_n["n"] += 1

        client.on_event(cb)
        client.start()
        try:
            time.sleep(1.0)
        finally:
            client.stop()
        with lock:
            n = det_n["n"]
        self.assertGreaterEqual(
            n, 2,
            f"expected >=2 detections in 1.0s with det_period=0.2s, got {n}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
