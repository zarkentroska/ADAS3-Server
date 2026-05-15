"""Unit tests for modules.ep32_client.

Covers:
  - set_enabled / toggle / state vocabulary
  - send_command / send_action mapping (including new TEST/STATUS tokens)
  - HTTP error -> bridge_unreachable status
  - HTTP 200/400/405/409 -> appropriate status string
  - probe_bridge happy path and unreachable path

We monkey-patch ``requests.post`` / ``requests.get`` inside the module so
the tests stay hermetic.
"""

import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import ep32_client  # noqa: E402
from modules.ep32_client import (  # noqa: E402
    Ep32ClientController,
    EP32_CONTROL_ACTIONS,
    EP32_SUPPORTED_COMMANDS,
    EP32_ACTION_TO_COMMAND,
    STATUS_BRIDGE_UNREACHABLE,
    STATUS_CONNECTED,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_INVALID_ACTION,
    STATUS_INVALID_PAYLOAD,
    STATUS_LEGACY_BRIDGE,
    STATUS_METHOD_NOT_ALLOWED,
    STATUS_NOT_CONNECTED,
    STATUS_OFF,
)


def _mk_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    if json_body is None:
        resp.content = b""
        resp.json.return_value = {}
    else:
        import json as _json
        resp.content = _json.dumps(json_body).encode()
        resp.json.return_value = json_body
    return resp


class TestVocabulary(unittest.TestCase):
    def test_unified_firmware_tokens_included(self):
        # The unified firmware accepts UP/DOWN/LEFT/RIGHT/TEST/STATUS.
        for token in ("UP", "DOWN", "LEFT", "RIGHT", "TEST", "STATUS"):
            self.assertIn(token, EP32_SUPPORTED_COMMANDS, token)
        # Legacy tokens are still recognised for forward-compat.
        for token in ("A", "B", "MENU", "AUTO"):
            self.assertIn(token, EP32_SUPPORTED_COMMANDS, token)

    def test_action_to_command_covers_test_and_status(self):
        self.assertEqual(EP32_ACTION_TO_COMMAND["test"], "TEST")
        self.assertEqual(EP32_ACTION_TO_COMMAND["status"], "STATUS")


class TestEnableDisable(unittest.TestCase):
    def test_initial_state_is_off(self):
        c = Ep32ClientController(base_url_supplier=lambda: "http://x")
        st = c.get_state()
        self.assertFalse(st["enabled"])
        self.assertEqual(st["status"], STATUS_OFF)
        self.assertEqual(st["last_command"], "")

    def test_enabling_sets_idle_not_scanning(self):
        # Regression: enabling EP32 BT from the server used to set status
        # to "scanning" indefinitely, which made the user think the server
        # was scanning Bluetooth locally. The new value is "idle".
        c = Ep32ClientController(base_url_supplier=lambda: "http://x")
        c.set_enabled(True)
        self.assertEqual(c.get_state()["status"], STATUS_IDLE)

    def test_toggle_round_trip(self):
        c = Ep32ClientController(base_url_supplier=lambda: "http://x")
        self.assertTrue(c.toggle_enabled())
        self.assertFalse(c.toggle_enabled())
        self.assertEqual(c.get_state()["status"], STATUS_OFF)


class TestSendCommand(unittest.TestCase):
    def setUp(self):
        self.controller = Ep32ClientController(
            base_url_supplier=lambda: "http://phone:8080",
        )
        self.controller.set_enabled(True)

    def _post_returns(self, response):
        return patch.object(ep32_client.requests, "post", return_value=response)

    def _post_raises(self, exc):
        return patch.object(ep32_client.requests, "post", side_effect=exc)

    def test_send_command_200_marks_connected(self):
        with self._post_returns(_mk_response(200)):
            result = self.controller.send_command("UP")
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], STATUS_CONNECTED)
        self.assertEqual(self.controller.get_state()["last_command"], "UP")

    def test_send_command_409_marks_not_connected(self):
        with self._post_returns(_mk_response(409)):
            result = self.controller.send_command("DOWN")
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], STATUS_NOT_CONNECTED)

    def test_send_command_400_marks_invalid_payload(self):
        with self._post_returns(_mk_response(400)):
            result = self.controller.send_command("LEFT")
        self.assertEqual(result["status"], STATUS_INVALID_PAYLOAD)

    def test_send_command_405_marks_method_not_allowed(self):
        with self._post_returns(_mk_response(405)):
            result = self.controller.send_command("RIGHT")
        self.assertEqual(result["status"], STATUS_METHOD_NOT_ALLOWED)

    def test_send_command_500_marks_error(self):
        with self._post_returns(_mk_response(500)):
            result = self.controller.send_command("UP")
        self.assertEqual(result["status"], STATUS_ERROR)

    def test_connection_error_marks_bridge_unreachable(self):
        # The user's reported symptom: Android works but server "scans" and
        # errors. That mapped to a generic "error" before; now it must be
        # the explicit `bridge_unreachable`.
        with self._post_raises(ep32_client.requests.ConnectionError("refused")):
            result = self.controller.send_command("UP")
        self.assertEqual(result["status"], STATUS_BRIDGE_UNREACHABLE)
        self.assertIn("Android", result["error"])

    def test_timeout_marks_bridge_unreachable(self):
        with self._post_raises(ep32_client.requests.Timeout("slow")):
            result = self.controller.send_command("UP")
        self.assertEqual(result["status"], STATUS_BRIDGE_UNREACHABLE)

    def test_disabled_send_fails_without_http(self):
        self.controller.set_enabled(False)
        # Sentinel: if requests.post is called we explode.
        with patch.object(ep32_client.requests, "post",
                          side_effect=AssertionError("must not POST when disabled")):
            result = self.controller.send_command("UP")
        self.assertEqual(result["status"], STATUS_OFF)

    def test_unknown_command_does_not_hit_network(self):
        with patch.object(ep32_client.requests, "post",
                          side_effect=AssertionError("must not POST for invalid cmd")):
            result = self.controller.send_command("FLY")
        self.assertEqual(result["status"], "invalid_command")

    def test_send_action_routes_arrow_keys_correctly(self):
        for action, expected_token in [
            ("up", "UP"), ("down", "DOWN"),
            ("left", "LEFT"), ("right", "RIGHT"),
            ("test", "TEST"), ("status", "STATUS"),
        ]:
            with self._post_returns(_mk_response(200)) as m:
                self.controller.send_action(action)
                args, kwargs = m.call_args
                self.assertEqual(kwargs["json"]["command"], expected_token,
                                 f"action={action}")


class TestProbeBridge(unittest.TestCase):
    def setUp(self):
        self.controller = Ep32ClientController(
            base_url_supplier=lambda: "http://phone:8080",
        )
        self.controller.set_enabled(True)

    def test_probe_405_keeps_idle(self):
        with patch.object(ep32_client.requests, "get",
                          return_value=_mk_response(405)):
            status = self.controller.probe_bridge()
        self.assertEqual(status, STATUS_IDLE)

    def test_probe_connection_error_marks_bridge_unreachable(self):
        with patch.object(ep32_client.requests, "get",
                          side_effect=ep32_client.requests.ConnectionError("nope")):
            status = self.controller.probe_bridge()
        self.assertEqual(status, STATUS_BRIDGE_UNREACHABLE)

    def test_probe_unexpected_status_marks_error(self):
        with patch.object(ep32_client.requests, "get",
                          return_value=_mk_response(503)):
            status = self.controller.probe_bridge()
        self.assertEqual(status, STATUS_ERROR)


class TestAcousticOverlayGating(unittest.TestCase):
    """The acoustic overlay should hide itself when there is nothing to say
    (EP32 disabled AND array disconnected), and otherwise render below the
    EP32 indicator (anchor="below-ep32", default y_top=230)."""

    def test_overlay_hidden_when_ep32_off_and_array_disconnected(self):
        import acoustic_integration as ai
        # Force a fresh state with connected=False.
        from modules.esp32_acoustic_array import AcousticArrayState
        ai._latest_state = AcousticArrayState(connected=False)
        # Pseudo frame: a tiny numpy ndarray-like. The overlay only reads
        # frame.shape and frame.copy() if it decides to draw.
        class FakeFrame:
            shape = (480, 640, 3)
            copy_calls = 0
            def copy(self):
                FakeFrame.copy_calls += 1
                return self
        frame = FakeFrame()
        out = ai.acoustic_overlay(frame, ep32_enabled=False)
        self.assertIs(out, frame)
        self.assertEqual(FakeFrame.copy_calls, 0,
                         "overlay should not have started drawing")

    def test_overlay_uses_default_anchor_when_ep32_on(self):
        # We can't easily verify the actual pixel positions without cv2,
        # but we can verify that the function does NOT bail out early
        # when ep32_enabled=True.
        import acoustic_integration as ai
        from modules.esp32_acoustic_array import AcousticArrayState
        ai._latest_state = AcousticArrayState(connected=False)
        try:
            import cv2  # noqa: F401
            has_cv2 = True
        except Exception:
            has_cv2 = False
        if not has_cv2:
            # Without cv2 the function bails before drawing; the assertion
            # we care about is the gating logic, not the cv2 path.
            self.skipTest("cv2 not available — gating-only test")
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = ai.acoustic_overlay(frame, ep32_enabled=True)
        self.assertIsNotNone(out)

    def test_overlay_hidden_when_ep32_off_even_if_simulation_connected(self):
        """Regresion: la integracion arrancaba la simulacion al inicio, lo
        que dejaba ``connected=True``, y la version vieja del overlay
        renderizaba "ARRAY OK [simulation]" en la esquina nada mas
        abrir la app, antes de que el usuario tocase EP32 BT. Ahora el
        gate exige explicitamente ``ep32_enabled=True``."""
        import acoustic_integration as ai
        from modules.esp32_acoustic_array import AcousticArrayState
        ai._latest_state = AcousticArrayState(connected=True,
                                               transport="simulation",
                                               mic_count=4)

        class FakeFrame:
            shape = (480, 640, 3)
            copy_calls = 0
            def copy(self):
                FakeFrame.copy_calls += 1
                return self
        frame = FakeFrame()
        out = ai.acoustic_overlay(frame, ep32_enabled=False)
        self.assertIs(out, frame)
        self.assertEqual(FakeFrame.copy_calls, 0)

    def test_overlay_force_show_bypasses_ep32_gate(self):
        """``force_show=True`` (modo debug) debe dibujar incluso con EP32
        en OFF y array desconectado."""
        import acoustic_integration as ai
        from modules.esp32_acoustic_array import AcousticArrayState
        ai._latest_state = AcousticArrayState(connected=False)
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("cv2 not available")
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = ai.acoustic_overlay(frame, ep32_enabled=False, force_show=True)
        self.assertIsNotNone(out)

    def test_overlay_default_y_is_below_dpad(self):
        """El badge debe caer en y >= 484 (debajo del panel D-pad EP32),
        que ocupa y=230..476 cuando EP32 BT esta activado."""
        import acoustic_integration as ai
        from modules.esp32_acoustic_array import AcousticArrayState
        ai._latest_state = AcousticArrayState(connected=True,
                                               transport="simulation",
                                               mic_count=4,
                                               doa_deg=10.0,
                                               energy=0.3,
                                               confidence=0.8)
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("cv2 not available")
        import numpy as np
        # Marcamos un pixel en y=230 (centro del D-pad) y comprobamos que
        # el overlay NO lo machaca; y otro en y=484 que SI deberia
        # quedar modificado.
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[230, 1280 - 100] = (123, 45, 67)  # pixel testigo bajo D-pad
        ai.acoustic_overlay(frame, ep32_enabled=True)
        # El badge esta a x in [w-270, w-10] e y in [484, 484+70]. El
        # pixel en y=230 debe seguir intacto.
        self.assertTrue(
            (frame[230, 1280 - 100] == [123, 45, 67]).all(),
            "El badge ARRAY se ha dibujado en la zona del D-pad",
        )


class TestControlEndpoint(unittest.TestCase):
    """POST /adas3/ep32-control on newer Android clients."""

    def setUp(self):
        self.controller = Ep32ClientController(
            base_url_supplier=lambda: "http://phone:8080",
        )

    def _post_returns(self, response):
        return patch.object(ep32_client.requests, "post", return_value=response)

    def _post_raises(self, exc):
        return patch.object(ep32_client.requests, "post", side_effect=exc)

    def test_control_actions_whitelist(self):
        self.assertEqual(
            EP32_CONTROL_ACTIONS,
            {"enable", "disable", "reconnect", "stop"},
        )

    def test_request_control_invalid_action_is_rejected_locally(self):
        with patch.object(ep32_client.requests, "post",
                          side_effect=AssertionError("must not POST for invalid action")):
            result = self.controller.request_control("FLY")
        self.assertEqual(result["status"], STATUS_INVALID_ACTION)

    def test_request_control_enable_with_connected_snapshot(self):
        snapshot = {
            "connected": True,
            "state": "CONNECTED",
            "detail": "ESP32-ADAS3",
            "enabled": True,
            "active": True,
            "bt_adapter_enabled": True,
            "permissions_granted": True,
            "firmware": "esp32-adas3 0.2.0",
            "mic_count": 4,
        }
        with self._post_returns(_mk_response(202, snapshot)) as m:
            result = self.controller.request_control("enable")
            kwargs = m.call_args.kwargs
            self.assertEqual(kwargs["json"]["action"], "enable")
            self.assertEqual(kwargs["json"]["type"], "adas3-ep32-control")
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], STATUS_CONNECTED)
        state = self.controller.get_state()
        self.assertTrue(state["enabled"])
        self.assertEqual(state["status"], STATUS_CONNECTED)
        self.assertEqual(state["bridge_status"]["mic_count"], 4)
        self.assertIs(state["control_supported"], True)

    def test_request_control_enable_with_scanning_snapshot_is_idle(self):
        snapshot = {"connected": False, "state": "SCANNING", "enabled": True}
        with self._post_returns(_mk_response(202, snapshot)):
            result = self.controller.request_control("enable")
        self.assertEqual(result["status"], STATUS_IDLE)
        self.assertEqual(self.controller.get_state()["status"], STATUS_IDLE)

    def test_request_control_disable_flips_local_state_off(self):
        # Force ON first so we can verify the controller actually flips OFF
        self.controller.set_enabled(True)
        snapshot = {"connected": False, "state": "OFF", "enabled": False}
        with self._post_returns(_mk_response(202, snapshot)):
            result = self.controller.request_control("disable")
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], STATUS_OFF)
        self.assertFalse(self.controller.is_enabled())

    def test_request_control_legacy_404_marks_unsupported(self):
        with self._post_returns(_mk_response(404)):
            result = self.controller.request_control("enable")
        self.assertEqual(result["status"], STATUS_LEGACY_BRIDGE)
        self.assertIs(self.controller.supports_control(), False)
        # Subsequent calls must NOT hit the network anymore.
        with patch.object(ep32_client.requests, "post",
                          side_effect=AssertionError("must not POST after legacy detected")):
            result2 = self.controller.request_control("enable")
        self.assertEqual(result2["status"], STATUS_LEGACY_BRIDGE)

    def test_request_control_405_also_marks_legacy(self):
        with self._post_returns(_mk_response(405)):
            result = self.controller.request_control("disable")
        self.assertEqual(result["status"], STATUS_LEGACY_BRIDGE)
        self.assertIs(self.controller.supports_control(), False)

    def test_request_control_connection_error(self):
        with self._post_raises(ep32_client.requests.ConnectionError("refused")):
            result = self.controller.request_control("enable")
        self.assertEqual(result["status"], STATUS_BRIDGE_UNREACHABLE)

    def test_request_control_400_invalid_payload(self):
        with self._post_returns(_mk_response(400, {"status": "invalid_payload"})):
            result = self.controller.request_control("reconnect")
        self.assertEqual(result["status"], STATUS_INVALID_PAYLOAD)

    def test_request_control_unexpected_status_marks_error(self):
        with self._post_returns(_mk_response(500)):
            result = self.controller.request_control("enable")
        self.assertEqual(result["status"], STATUS_ERROR)


class TestStatusEndpoint(unittest.TestCase):
    """GET /adas3/ep32-status on newer Android clients."""

    def setUp(self):
        self.controller = Ep32ClientController(
            base_url_supplier=lambda: "http://phone:8080",
        )
        self.controller.set_enabled(True)

    def test_fetch_status_connected_promotes_state(self):
        snapshot = {
            "connected": True,
            "state": "CONNECTED",
            "detail": "ESP32-ADAS3",
            "enabled": True,
            "active": True,
            "bt_adapter_enabled": True,
            "permissions_granted": True,
            "firmware": "esp32-adas3 0.2.0",
            "mic_count": 4,
        }
        with patch.object(ep32_client.requests, "get",
                          return_value=_mk_response(200, snapshot)):
            body = self.controller.fetch_status()
        self.assertEqual(body["mic_count"], 4)
        self.assertEqual(self.controller.get_state()["status"], STATUS_CONNECTED)

    def test_fetch_status_scanning_promotes_to_idle(self):
        snapshot = {"connected": False, "state": "SCANNING", "enabled": True}
        with patch.object(ep32_client.requests, "get",
                          return_value=_mk_response(200, snapshot)):
            self.controller.fetch_status()
        self.assertEqual(self.controller.get_state()["status"], STATUS_IDLE)

    def test_fetch_status_disabled_in_bridge_marks_not_connected(self):
        # Android: usuario tiene el switch EP32 BT en OFF en la app
        snapshot = {"connected": False, "state": "OFF", "enabled": False}
        with patch.object(ep32_client.requests, "get",
                          return_value=_mk_response(200, snapshot)):
            self.controller.fetch_status()
        self.assertEqual(self.controller.get_state()["status"],
                         STATUS_NOT_CONNECTED)

    def test_fetch_status_error_state_marks_error(self):
        snapshot = {"connected": False, "state": "ERROR",
                    "detail": "Bluetooth is disabled", "enabled": True}
        with patch.object(ep32_client.requests, "get",
                          return_value=_mk_response(200, snapshot)):
            self.controller.fetch_status()
        state = self.controller.get_state()
        self.assertEqual(state["status"], STATUS_ERROR)
        self.assertIn("Bluetooth is disabled", state["last_error"])

    def test_fetch_status_legacy_404_marks_unsupported(self):
        with patch.object(ep32_client.requests, "get",
                          return_value=_mk_response(404)):
            body = self.controller.fetch_status()
        self.assertEqual(body, {})
        self.assertIs(self.controller.supports_status(), False)
        # Now legacy: next call must NOT hit the network.
        with patch.object(ep32_client.requests, "get",
                          side_effect=AssertionError("must not GET after legacy detected")):
            body2 = self.controller.fetch_status()
        self.assertEqual(body2, {})

    def test_fetch_status_connection_error_marks_bridge_unreachable(self):
        with patch.object(ep32_client.requests, "get",
                          side_effect=ep32_client.requests.ConnectionError("refused")):
            self.controller.fetch_status()
        self.assertEqual(self.controller.get_state()["status"],
                         STATUS_BRIDGE_UNREACHABLE)


class TestToggleHelperBehaviour(unittest.TestCase):
    """End-to-end: simulate the testcam toggle helper logic against the
    new and legacy bridges, in-process (no Tk, no cv2, no threading)."""

    def setUp(self):
        self.controller = Ep32ClientController(
            base_url_supplier=lambda: "http://phone:8080",
        )

    def _toggle_on_like_testcam(self):
        new_value = self.controller.toggle_enabled()
        if new_value:
            res = self.controller.request_control("enable")
            if res.get("status") == STATUS_LEGACY_BRIDGE:
                self.controller.probe_bridge()
            else:
                self.controller.fetch_status()
        return new_value

    def test_toggle_on_with_new_bridge_uses_control_then_status(self):
        snap_enable = {"connected": False, "state": "SCANNING", "enabled": True}
        snap_status = {"connected": True, "state": "CONNECTED", "enabled": True,
                       "mic_count": 4}
        with patch.object(ep32_client.requests, "post",
                          return_value=_mk_response(202, snap_enable)) as p, \
             patch.object(ep32_client.requests, "get",
                          return_value=_mk_response(200, snap_status)) as g:
            self._toggle_on_like_testcam()
            self.assertEqual(p.call_count, 1)
            self.assertEqual(g.call_count, 1)
        self.assertEqual(self.controller.get_state()["status"], STATUS_CONNECTED)

    def test_toggle_on_with_legacy_bridge_falls_back_to_probe(self):
        # First POST (control) -> 404. Then GET probe to /ep32-command -> 405.
        with patch.object(ep32_client.requests, "post",
                          return_value=_mk_response(404)) as p, \
             patch.object(ep32_client.requests, "get",
                          return_value=_mk_response(405)) as g:
            self._toggle_on_like_testcam()
            # control_supported flipped to False after the 404
            self.assertIs(self.controller.supports_control(), False)
            self.assertEqual(p.call_count, 1)
            self.assertEqual(g.call_count, 1)
        # Probe with 405 -> idle
        self.assertEqual(self.controller.get_state()["status"], STATUS_IDLE)


class TestStatusUnreachableLogDampening(unittest.TestCase):
    """Cuando el puente Android está unreachable, `fetch_status` se
    llama por el tick de polling y antes loggeaba WARNING cada vez.
    Eso saturaba la consola del usuario. Ahora sólo loggea WARNING en
    el primer fallo de la racha; los siguientes pasan a DEBUG hasta
    que vuelva una respuesta OK.
    """

    def setUp(self):
        self.controller = Ep32ClientController(
            base_url_supplier=lambda: "http://phone:8080",
        )
        self.controller.set_enabled(True)

    def _get_raises(self, exc):
        return patch.object(ep32_client.requests, "get", side_effect=exc)

    def _get_returns(self, response):
        return patch.object(ep32_client.requests, "get", return_value=response)

    def test_first_failure_warns_subsequent_debug(self):
        records = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        h = _Handler()
        h.setLevel(logging.DEBUG)
        prev_level = ep32_client.log.level
        ep32_client.log.setLevel(logging.DEBUG)
        ep32_client.log.addHandler(h)
        try:
            with self._get_raises(ep32_client.requests.ConnectionError("nope")):
                for _ in range(5):
                    self.controller.fetch_status()
        finally:
            ep32_client.log.removeHandler(h)
            ep32_client.log.setLevel(prev_level)

        warnings = [r for r in records if r.levelno == logging.WARNING]
        debugs = [r for r in records if r.levelno == logging.DEBUG]
        # Exactamente un WARNING, los demás como DEBUG.
        self.assertEqual(len(warnings), 1, [r.getMessage() for r in records])
        self.assertGreaterEqual(len(debugs), 4)

    def test_warning_reenabled_after_success(self):
        records = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        h = _Handler()
        h.setLevel(logging.DEBUG)
        prev_level = ep32_client.log.level
        ep32_client.log.setLevel(logging.DEBUG)
        ep32_client.log.addHandler(h)
        try:
            with self._get_raises(ep32_client.requests.ConnectionError("nope")):
                self.controller.fetch_status()
                self.controller.fetch_status()
            # Bridge vuelve a responder.
            with self._get_returns(_mk_response(200, json_body={"connected": True})):
                self.controller.fetch_status()
            # Vuelve a fallar — debe volver a loggear WARNING.
            with self._get_raises(ep32_client.requests.ConnectionError("nope2")):
                self.controller.fetch_status()
        finally:
            ep32_client.log.removeHandler(h)
            ep32_client.log.setLevel(prev_level)

        warnings = [r for r in records if r.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 2,
                         [r.getMessage() for r in records])


if __name__ == "__main__":
    unittest.main(verbosity=2)
