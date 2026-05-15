"""Unit tests for modules.audio_source.AudioSourceController."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.audio_source import (  # noqa: E402
    AudioSourceController,
    SOURCE_ESP32_ARRAY,
    SOURCE_PHONE_MIC,
    SOURCES_IN_ORDER,
)


class TestAudioSourceController(unittest.TestCase):
    def test_default_is_phone_mic(self):
        c = AudioSourceController(default_source=SOURCE_PHONE_MIC)
        self.assertEqual(c.get(), SOURCE_PHONE_MIC)
        self.assertTrue(c.is_phone())
        self.assertFalse(c.is_array())

    def test_set_to_array(self):
        c = AudioSourceController()
        self.assertTrue(c.set(SOURCE_ESP32_ARRAY))
        self.assertTrue(c.is_array())
        # No-op when value is unchanged.
        self.assertFalse(c.set(SOURCE_ESP32_ARRAY))

    def test_set_rejects_invalid(self):
        c = AudioSourceController()
        with self.assertRaises(ValueError):
            c.set("usb_dongle")

    def test_cycle_rotates(self):
        c = AudioSourceController(default_source=SOURCE_PHONE_MIC)
        self.assertEqual(c.cycle(), SOURCE_ESP32_ARRAY)
        self.assertEqual(c.cycle(), SOURCE_PHONE_MIC)

    def test_on_change_invoked(self):
        events = []
        c = AudioSourceController(on_change=lambda old, new: events.append((old, new)))
        c.set(SOURCE_ESP32_ARRAY)
        c.set(SOURCE_PHONE_MIC)
        self.assertEqual(events,
                         [(SOURCE_PHONE_MIC, SOURCE_ESP32_ARRAY),
                          (SOURCE_ESP32_ARRAY, SOURCE_PHONE_MIC)])

    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "audio_source.json")
            c1 = AudioSourceController(config_file=cfg)
            c1.set(SOURCE_ESP32_ARRAY)
            with open(cfg, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("audio_source"), SOURCE_ESP32_ARRAY)
            # Reload — should respect the persisted value.
            c2 = AudioSourceController(config_file=cfg)
            self.assertEqual(c2.get(), SOURCE_ESP32_ARRAY)

    def test_persist_does_not_clobber_other_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "audio_source.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump({"unrelated": 42}, f)
            c = AudioSourceController(config_file=cfg)
            c.set(SOURCE_ESP32_ARRAY)
            with open(cfg, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload.get("audio_source"), SOURCE_ESP32_ARRAY)
            self.assertEqual(payload.get("unrelated"), 42)

    def test_options_order(self):
        self.assertEqual(tuple(AudioSourceController.options()), SOURCES_IN_ORDER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
