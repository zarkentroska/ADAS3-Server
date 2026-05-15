"""Unit tests for modules.esp32_acoustic_array.

These tests do not require pyserial or any real hardware. They cover:

  - parse_message: JSON happy path, JSON malformed, CSV legacy, empty lines
  - AcousticArrayClient with simulation transport: starts, dispatches events,
    state snapshot, debounce, clean shutdown
"""

import json
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
    DEFAULT_WIRING,
    default_wiring_dict,
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


class TestDefaultWiring(unittest.TestCase):
    """Pinout regression tests. These lock in the definitive 4-mic wiring;
    if anyone re-numbers a GPIO without thinking, the suite breaks loudly."""

    def test_default_wiring_summary(self):
        self.assertEqual(DEFAULT_WIRING.mic_count, 4)
        self.assertEqual(DEFAULT_WIRING.power_rail, "3V3")
        self.assertEqual(DEFAULT_WIRING.common_ground, "GND")
        self.assertEqual(len(DEFAULT_WIRING.mics), 4)
        self.assertEqual(len(DEFAULT_WIRING.buses), 2)

    def test_default_wiring_mics(self):
        mics = {m.index: m for m in DEFAULT_WIRING.mics}
        self.assertEqual(mics[1].pair, "A")
        self.assertEqual(mics[1].side, "LEFT")
        self.assertEqual(mics[1].sel_to, "GND")
        self.assertEqual(mics[2].pair, "A")
        self.assertEqual(mics[2].side, "RIGHT")
        self.assertEqual(mics[2].sel_to, "3V3")
        self.assertEqual(mics[3].pair, "B")
        self.assertEqual(mics[3].side, "LEFT")
        self.assertEqual(mics[3].sel_to, "GND")
        self.assertEqual(mics[4].pair, "B")
        self.assertEqual(mics[4].side, "RIGHT")
        self.assertEqual(mics[4].sel_to, "3V3")

    def test_default_wiring_buses(self):
        buses = {b.pair: b for b in DEFAULT_WIRING.buses}
        self.assertEqual(buses["A"].bclk_gpio, 14)
        self.assertEqual(buses["A"].lrcl_gpio, 13)
        self.assertEqual(buses["A"].dout_gpio, 34)
        self.assertEqual(buses["A"].left_mic, 1)
        self.assertEqual(buses["A"].right_mic, 2)
        self.assertEqual(buses["B"].bclk_gpio, 22)
        self.assertEqual(buses["B"].lrcl_gpio, 21)
        self.assertEqual(buses["B"].dout_gpio, 35)
        self.assertEqual(buses["B"].left_mic, 3)
        self.assertEqual(buses["B"].right_mic, 4)

    def test_default_wiring_remote_control(self):
        rc = DEFAULT_WIRING.remote_control
        self.assertEqual(rc.up_gpio, 26)
        self.assertEqual(rc.down_gpio, 27)
        self.assertEqual(rc.left_gpio, 32)
        self.assertEqual(rc.right_gpio, 33)

    def test_default_wiring_dict_round_trip(self):
        d = default_wiring_dict()
        self.assertEqual(d["mic_count"], 4)
        self.assertEqual(len(d["mics"]), 4)
        self.assertEqual(len(d["buses"]), 2)
        self.assertEqual(
            [b["dout_gpio"] for b in d["buses"]],
            [34, 35],
        )
        self.assertEqual(
            d["remote_control"],
            {"up_gpio": 26, "down_gpio": 27, "left_gpio": 32, "right_gpio": 33},
        )


class TestParserExtendedMetadata(unittest.TestCase):
    """Parser must preserve the new optional metadata keys verbatim."""

    def test_parse_acoustic_with_pair_and_bus(self):
        msg = parse_message(
            '{"type":"acoustic","detected":true,"doa_deg":12.0,'
            '"energy":0.5,"confidence":0.7,"mic_count":4,'
            '"pair":"B","bus":"i2s1"}'
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg["pair"], "B")
        self.assertEqual(msg["bus"], "i2s1")

    def test_parse_heartbeat_with_wiring(self):
        wiring = default_wiring_dict()
        line = json.dumps({
            "type": "heartbeat",
            "mic_count": 4,
            "firmware": "fw-1",
            "wiring": wiring,
        })
        msg = parse_message(line)
        self.assertIsNotNone(msg)
        self.assertEqual(msg["type"], "heartbeat")
        self.assertEqual(msg["wiring"], wiring)

    def test_parse_acoustic_with_config_synonym(self):
        msg = parse_message(
            '{"type":"acoustic","detected":false,"doa_deg":0,'
            '"energy":0,"confidence":0,"config":{"mic_count":4}}'
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg["type"], "acoustic")
        self.assertEqual(msg["config"], {"mic_count": 4})


class TestClientWiringState(unittest.TestCase):
    """The client must seed default wiring and absorb payload wiring."""

    def _direct_client(self):
        # Disabled config so we can poke _handle_message directly without
        # spinning up the worker thread (no simulation noise to deal with).
        cfg = AcousticArrayConfig(enabled=False, transport="simulation")
        return AcousticArrayClient(cfg)

    def test_initial_state_has_default_wiring(self):
        client = self._direct_client()
        st = client.get_state()
        self.assertEqual(st.mic_count, 4)
        self.assertEqual(st.wiring_source, "default")
        self.assertEqual(st.wiring, default_wiring_dict())

    def test_minimal_payload_keeps_default_wiring(self):
        client = self._direct_client()
        # Minimal heartbeat without wiring metadata.
        client._handle_message({"type": "heartbeat", "mic_count": 4,
                                "firmware": "min-fw"})
        st = client.get_state()
        self.assertEqual(st.firmware, "min-fw")
        self.assertEqual(st.wiring_source, "default")
        # Default wiring is still in place.
        self.assertEqual(st.wiring, default_wiring_dict())

    def test_enriched_payload_overrides_wiring(self):
        client = self._direct_client()
        custom = default_wiring_dict()
        # Tweak the dict to prove it is being preserved verbatim.
        custom["buses"][0]["dout_gpio"] = 99
        client._handle_message({
            "type": "heartbeat",
            "mic_count": 4,
            "firmware": "rich-fw",
            "wiring": custom,
        })
        st = client.get_state()
        self.assertEqual(st.wiring_source, "payload")
        self.assertEqual(st.wiring["buses"][0]["dout_gpio"], 99)

    def test_pair_and_bus_are_stored(self):
        client = self._direct_client()
        client._handle_message({
            "type": "acoustic",
            "detected": True,
            "doa_deg": 10.0,
            "energy": 0.9,
            "confidence": 0.9,
            "mic_count": 4,
            "pair": "B",
            "bus": "i2s1",
        })
        st = client.get_state()
        self.assertEqual(st.pair, "B")
        self.assertEqual(st.bus, "i2s1")

    def test_config_synonym_absorbed_as_wiring(self):
        client = self._direct_client()
        client._handle_message({
            "type": "heartbeat",
            "mic_count": 4,
            "config": {"mic_count": 4, "buses": [], "mics": [],
                       "remote_control": {}, "power_rail": "3V3",
                       "common_ground": "GND"},
        })
        st = client.get_state()
        self.assertEqual(st.wiring_source, "payload")
        self.assertEqual(st.wiring["power_rail"], "3V3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
