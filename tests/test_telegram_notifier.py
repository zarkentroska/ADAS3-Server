import unittest

from modules.telegram_notifier import CooldownGate


class TestCooldownGate(unittest.TestCase):
    def test_allow_respects_cooldown_window(self):
        gate = CooldownGate({"yolo": 30})
        self.assertTrue(gate.allow("yolo", now=100.0))
        self.assertFalse(gate.allow("yolo", now=110.0))
        self.assertTrue(gate.allow("yolo", now=131.0))

    def test_update_cooldown_clamps_negative_values(self):
        gate = CooldownGate({"rf": 30})
        gate.update_cooldowns({"rf": -5})
        self.assertTrue(gate.allow("rf", now=1.0))
        self.assertTrue(gate.allow("rf", now=1.0))


if __name__ == "__main__":
    unittest.main()
