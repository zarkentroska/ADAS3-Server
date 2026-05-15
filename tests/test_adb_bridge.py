"""Tests para resolución de ``adb`` en `modules.adb_bridge`."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import adb_bridge  # noqa: E402


class GetAdbPathTests(unittest.TestCase):
    def test_uses_which_when_available(self):
        shutil_mock = mock.Mock()
        shutil_mock.which.return_value = "/opt/homebrew/bin/adb"
        self.assertEqual(adb_bridge.get_adb_path(shutil_mock), "/opt/homebrew/bin/adb")

    @mock.patch.object(adb_bridge.sys, "platform", "darwin")
    def test_macos_homebrew_fallback_when_which_misses(self):
        shutil_mock = mock.Mock()
        shutil_mock.which.return_value = None
        homebrew_adb = "/opt/homebrew/bin/adb"

        def _isfile(path):
            return path == homebrew_adb

        with mock.patch.object(adb_bridge.os.path, "isfile", side_effect=_isfile):
            with mock.patch.object(adb_bridge.os, "access", return_value=True):
                self.assertEqual(adb_bridge.get_adb_path(shutil_mock), homebrew_adb)

    @mock.patch.object(adb_bridge.sys, "platform", "darwin")
    def test_macos_android_sdk_fallback(self):
        shutil_mock = mock.Mock()
        shutil_mock.which.return_value = None
        sdk_adb = os.path.join("/Users/test", "Library", "Android", "sdk", "platform-tools", "adb")

        def _isfile(path):
            return path == sdk_adb

        with mock.patch.dict(os.environ, {}, clear=False):
            with mock.patch.object(adb_bridge.os.path, "expanduser", return_value="/Users/test"):
                with mock.patch.object(adb_bridge.os.path, "isfile", side_effect=_isfile):
                    with mock.patch.object(adb_bridge.os, "access", return_value=True):
                        self.assertEqual(adb_bridge.get_adb_path(shutil_mock), sdk_adb)

    def test_returns_none_when_not_found(self):
        shutil_mock = mock.Mock()
        shutil_mock.which.return_value = None
        with mock.patch.object(adb_bridge.os.path, "isfile", return_value=False):
            self.assertIsNone(adb_bridge.get_adb_path(shutil_mock))


class PollAdbConnectionTests(unittest.TestCase):
    def test_skips_poll_when_adb_missing(self):
        shutil_mock = mock.Mock()
        shutil_mock.which.return_value = None
        with mock.patch.object(adb_bridge.os.path, "isfile", return_value=False):
            state = adb_bridge.poll_adb_connection(
                last_adb_check=0,
                adb_check_interval=5.0,
                adb_connected=False,
                pending_ip_change=None,
                last_wifi_ip=None,
                current_ip="192.168.1.10:8080",
                adb_target_ip="127.0.0.1:8080",
                subprocess_module=mock.Mock(),
                shutil_module=shutil_mock,
                time_module=mock.Mock(time=mock.Mock(return_value=100.0)),
            )
        self.assertFalse(state["adb_connected"])
        self.assertEqual(state["messages"], [])


if __name__ == "__main__":
    unittest.main()
