import ipaddress
import json
import queue
import re
import socket
import threading
import time

import requests


def _is_valid_ipv4(ip):
    if not ip:
        return False
    if not re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", str(ip)):
        return False
    try:
        return all(0 <= int(part) <= 255 for part in str(ip).split("."))
    except ValueError:
        return False


class LanDiscoveryManager:
    def __init__(
        self,
        *,
        adb_target_ip="127.0.0.1:8080",
        beacon_port=39000,
        beacon_type="adas3-client-discovery",
        max_age_seconds=25.0,
    ):
        self.adb_target_ip = str(adb_target_ip)
        self.beacon_port = int(beacon_port)
        self.beacon_type = str(beacon_type)
        self.max_age_seconds = float(max_age_seconds)
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._seen = {}

    def _register_candidate(self, ip_text, port_value):
        if not _is_valid_ipv4(ip_text):
            return
        if str(ip_text).startswith("127.") or str(ip_text).endswith(".1"):
            return
        try:
            ip_obj = ipaddress.ip_address(ip_text)
        except ValueError:
            return
        if ip_obj.is_loopback:
            return
        if not ip_obj.is_private and not str(ip_text).startswith("100."):
            return
        try:
            port_int = int(port_value)
        except (TypeError, ValueError):
            return
        if port_int <= 0 or port_int > 65535:
            return
        value = f"{ip_text}:{port_int}"
        with self._lock:
            self._seen[value] = time.time()

    def _listener_worker(self):
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", self.beacon_port))
            sock.settimeout(1.0)
            print(f"[LAN DISCOVERY] Escuchando beacons UDP en puerto {self.beacon_port}")
        except Exception as e:
            print(f"[LAN DISCOVERY] No se pudo iniciar listener UDP: {e}")
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
            return

        while not self._stop_event.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception:
                continue

            try:
                payload = json.loads(data.decode("utf-8", errors="replace").strip())
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("type", "")).strip() != self.beacon_type:
                continue

            beacon_ip = str(payload.get("ip", "")).strip()
            source_ip = str(addr[0]).strip() if addr and len(addr) >= 1 else ""
            target_ip = beacon_ip if _is_valid_ipv4(beacon_ip) else source_ip
            beacon_port = payload.get("port", 8080)
            self._register_candidate(target_ip, beacon_port)

        try:
            sock.close()
        except Exception:
            pass

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listener_worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

    def get_recent_values(self):
        now = time.time()
        valid_values = []
        stale = []
        with self._lock:
            for value, ts in self._seen.items():
                if now - ts <= self.max_age_seconds:
                    valid_values.append(value)
                else:
                    stale.append(value)
            for value in stale:
                self._seen.pop(value, None)
        return sorted(valid_values)

    def get_recent_targets(self):
        targets = []
        for ip_with_port in self.get_recent_values():
            parts = str(ip_with_port).rsplit(":", 1)
            if len(parts) != 2:
                continue
            ip_text, port_text = parts
            if not _is_valid_ipv4(ip_text) or not str(port_text).isdigit():
                continue
            port_int = int(port_text)
            if port_int <= 0 or port_int > 65535:
                continue
            targets.append((ip_text, port_int))
        return targets

    def build_ip_selector_options(
        self,
        *,
        current_ip_with_port,
        t_func,
        get_tailscale_ip_fn,
        get_tailscale_devices_fn,
    ):
        self.start()
        candidates = []
        seen = set()

        adb_value = self.adb_target_ip
        candidates.append({"label": f"{t_func('ip_selector_adb')} {adb_value}", "value": adb_value})
        seen.add(adb_value)

        def _add_option(ip_with_port, label_prefix, source="generic"):
            if not isinstance(ip_with_port, str):
                return
            parts = ip_with_port.rsplit(":", 1)
            if len(parts) != 2:
                return
            raw_ip, raw_port = parts
            if not _is_valid_ipv4(raw_ip) or not str(raw_port).isdigit():
                return
            if str(raw_ip).startswith("127."):
                return
            if source == "lan" and str(raw_ip).endswith(".1"):
                return
            try:
                ip_obj = ipaddress.ip_address(raw_ip)
            except ValueError:
                return
            if ip_obj.is_loopback:
                return
            if not ip_obj.is_private and not str(raw_ip).startswith("100."):
                return
            if ip_with_port in seen:
                return
            seen.add(ip_with_port)
            candidates.append({"label": f"{label_prefix} {ip_with_port}", "value": ip_with_port})

        _add_option(str(current_ip_with_port), t_func("ip_selector_current"), source="generic")

        own_tailscale_ips = set()
        tailscale_ip = get_tailscale_ip_fn()
        if tailscale_ip:
            for ip_line in str(tailscale_ip).splitlines():
                ip_line = ip_line.strip()
                if _is_valid_ipv4(ip_line):
                    own_tailscale_ips.add(ip_line)

        for device in get_tailscale_devices_fn():
            device_ip = str(device.get("ip", "")).strip()
            device_name = str(device.get("name", "")).strip()
            if device_ip in own_tailscale_ips:
                continue
            if _is_valid_ipv4(device_ip):
                label = t_func("ip_selector_tailscale_device", device_name) if device_name else t_func("ip_selector_tailscale")
                _add_option(f"{device_ip}:{str(current_ip_with_port).rsplit(':', 1)[-1]}", label, source="tailscale")

        for ip_with_port in self.get_recent_values():
            _add_option(ip_with_port, t_func("ip_selector_lan"), source="lan")

        return candidates


class ClientDetectionEventWorker:
    def __init__(
        self,
        *,
        targets_supplier,
        endpoint_path="/adas3/detection-event",
        timeout_seconds=1.6,
        event_type="adas3-server-detection",
        cooldowns=None,
        queue_size=64,
    ):
        self._targets_supplier = targets_supplier
        self._endpoint_path = str(endpoint_path)
        self._timeout = float(timeout_seconds)
        self._event_type = str(event_type)
        self._cooldowns = dict(cooldowns or {"yolo": 10.0, "rf": 10.0, "tensorflow": 10.0})
        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._thread = None
        self._running = False
        self._cooldown_lock = threading.Lock()
        self._last_sent = {}

    def _allowed(self, event_name, now_ts):
        cooldown = float(self._cooldowns.get(event_name, 8.0))
        if cooldown <= 0:
            return True
        with self._cooldown_lock:
            prev = float(self._last_sent.get(event_name, 0.0))
            if (now_ts - prev) < cooldown:
                return False
            self._last_sent[event_name] = now_ts
        return True

    def _post_payload(self, payload):
        targets = self._targets_supplier() or []
        for ip_text, port_int in targets:
            url = f"http://{ip_text}:{port_int}{self._endpoint_path}"
            try:
                requests.post(url, json=payload, timeout=self._timeout)
            except Exception:
                continue

    def _worker_loop(self):
        while self._running:
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                continue
            try:
                self._post_payload(payload)
            except Exception as exc:
                print(f"[CLIENT EVENT] Error enviando evento: {exc}")
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.2)
        self._thread = None

    def enqueue(self, event_name, *, timestamp, confidence=None, frequency_hz=None):
        ts = float(timestamp or time.time())
        if not self._allowed(event_name, ts):
            return

        confidence_value = max(0.0, min(1.0, float(confidence or 0.0)))
        payload = {
            "type": self._event_type,
            "event": str(event_name),
            "timestamp": ts,
            "time": time.strftime("%H:%M:%S", time.localtime(ts)),
            "confidence": confidence_value,
            "confidence_percent": int(confidence_value * 100),
        }
        if frequency_hz is not None:
            try:
                payload["frequency_hz"] = float(frequency_hz)
            except (TypeError, ValueError):
                pass

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except Exception:
                pass
            try:
                self._queue.put_nowait(payload)
            except Exception:
                pass
