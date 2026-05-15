"""
ESP32 Acoustic Array integration for ADAS3-Server.

Self-contained module that ingests lightweight events (DOA, energy, confidence,
detection flag, heartbeat) from a remote ESP32 reading an I2S MEMS microphone
array (SPH0645 / INMP441). Raw audio is NOT carried over the link — only
preprocessed events. The audio clip in Telegram alerts keeps coming from the
phone client as today; the array is a direction/confirmation sensor.

Supports four transports:

  - "serial"      : pyserial (USB CDC, Bluetooth SPP)
  - "tcp"         : TCP client to ESP32 server, line-delimited
  - "udp"         : UDP listener (binds local port), line-delimited
  - "simulation"  : in-process generator for UI testing without hardware

Message wire format (primary, JSON lines):

    {"type":"heartbeat","mic_count":4,"firmware":"adas-array-0.1"}
    {"type":"acoustic","detected":true,"doa_deg":35.0,
     "energy":0.72,"confidence":0.81,"mic_count":4}

Also accepts a CSV legacy form on a single line:

    DOA,energy,confidence,detected
    35.0,0.72,0.81,1

The module is intentionally decoupled from testcam.py: no DSP, no Tk, no cv2.
Integration happens through a thread-safe callback and the public State.

Author: ADAS3 team
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import socket
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional


log = logging.getLogger("adas3.acoustic_array")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default values are intentionally conservative so the module is safe to load
# even when no ESP32 is connected.
DEFAULT_BAUDRATE = 115200
DEFAULT_TCP_PORT = 5005
DEFAULT_UDP_PORT = 5005
DEFAULT_HEARTBEAT_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Definitive 4-microphone wiring (two I2S pairs + PC817 remote control).
#
# This is the physical layout the firmware/cliente actually use. Every mic
# shares power (3V3) and ground from the ESP32; SEL is hardwired locally
# (no GPIO) to GND for the LEFT channel and to 3V3 for the RIGHT channel of
# each pair. Two independent I2S buses carry pair A and pair B.
#
#     ESP32 3V3 -> Mic1/Mic2/Mic3/Mic4  (single power rail in parallel)
#     ESP32 GND -> Mic1..Mic4 + PC817   (common ground)
#
#     Pair A (I2S bus 0):
#         BCLK=GPIO14, LRCL=GPIO13, DOUT=GPIO34
#         Mic1 SEL->GND  -> LEFT
#         Mic2 SEL->3V3  -> RIGHT
#
#     Pair B (I2S bus 1):
#         BCLK=GPIO22, LRCL=GPIO21, DOUT=GPIO35
#         Mic3 SEL->GND  -> LEFT
#         Mic4 SEL->3V3  -> RIGHT
#
#     YT2000 / PC817 remote control:
#         UP=GPIO26, DOWN=GPIO27, LEFT=GPIO32, RIGHT=GPIO33
#
# The server keeps this as the canonical default. If the client sends a
# richer payload carrying its own `wiring` / `config` / `pair` / `bus`
# metadata, we preserve it verbatim in the state; otherwise we fall back to
# DEFAULT_WIRING so downstream consumers (overlay, internal events, status
# endpoints) always see a coherent description.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicChannel:
    """One physical microphone in the array."""

    index: int          # 1..4, matches Mic1..Mic4 in the schematic
    pair: str           # "A" or "B"
    side: str           # "LEFT" or "RIGHT" (driven by SEL pin)
    sel_to: str         # "GND" or "3V3"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "pair": self.pair,
            "side": self.side,
            "sel_to": self.sel_to,
        }


@dataclass(frozen=True)
class I2sBus:
    """One I2S bus carrying a stereo pair of MEMS mics."""

    pair: str           # "A" or "B"
    bclk_gpio: int      # ESP32 GPIO number
    lrcl_gpio: int
    dout_gpio: int
    left_mic: int       # index of the mic on LEFT channel
    right_mic: int      # index of the mic on RIGHT channel

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "bclk_gpio": self.bclk_gpio,
            "lrcl_gpio": self.lrcl_gpio,
            "dout_gpio": self.dout_gpio,
            "left_mic": self.left_mic,
            "right_mic": self.right_mic,
        }


@dataclass(frozen=True)
class RemoteControlPinout:
    """YT2000 / PC817 opto-isolated remote control pinout."""

    up_gpio: int
    down_gpio: int
    left_gpio: int
    right_gpio: int

    def to_dict(self) -> dict:
        return {
            "up_gpio": self.up_gpio,
            "down_gpio": self.down_gpio,
            "left_gpio": self.left_gpio,
            "right_gpio": self.right_gpio,
        }


@dataclass(frozen=True)
class ArrayWiring:
    """Definitive physical wiring for the ESP32 + 4-mic acoustic array."""

    power_rail: str = "3V3"
    common_ground: str = "GND"
    mic_count: int = 4
    mics: tuple = ()                # tuple[MicChannel, ...]
    buses: tuple = ()               # tuple[I2sBus, ...]
    remote_control: RemoteControlPinout = field(
        default_factory=lambda: RemoteControlPinout(26, 27, 32, 33)
    )

    def to_dict(self) -> dict:
        return {
            "power_rail": self.power_rail,
            "common_ground": self.common_ground,
            "mic_count": self.mic_count,
            "mics": [m.to_dict() for m in self.mics],
            "buses": [b.to_dict() for b in self.buses],
            "remote_control": self.remote_control.to_dict(),
        }


DEFAULT_WIRING = ArrayWiring(
    power_rail="3V3",
    common_ground="GND",
    mic_count=4,
    mics=(
        MicChannel(index=1, pair="A", side="LEFT",  sel_to="GND"),
        MicChannel(index=2, pair="A", side="RIGHT", sel_to="3V3"),
        MicChannel(index=3, pair="B", side="LEFT",  sel_to="GND"),
        MicChannel(index=4, pair="B", side="RIGHT", sel_to="3V3"),
    ),
    buses=(
        I2sBus(pair="A", bclk_gpio=14, lrcl_gpio=13, dout_gpio=34,
               left_mic=1, right_mic=2),
        I2sBus(pair="B", bclk_gpio=22, lrcl_gpio=21, dout_gpio=35,
               left_mic=3, right_mic=4),
    ),
    remote_control=RemoteControlPinout(
        up_gpio=26, down_gpio=27, left_gpio=32, right_gpio=33,
    ),
)


def default_wiring_dict() -> dict:
    """Return the canonical default wiring as a plain dict (deep copy safe)."""
    return DEFAULT_WIRING.to_dict()


@dataclass
class AcousticArrayConfig:
    """Tunable parameters for the acoustic array client."""

    # Master switch. If false, start_acoustic_array() returns a no-op client.
    enabled: bool = True

    # Transport: "serial" | "tcp" | "udp" | "simulation"
    transport: str = "serial"

    # Serial transport
    port: Optional[str] = None  # e.g. "/dev/ttyUSB0", "COM7". None = autodetect.
    baudrate: int = DEFAULT_BAUDRATE

    # TCP/UDP transport
    host: str = "0.0.0.0"  # TCP: ESP32 address; UDP: local bind address
    tcp_port: int = DEFAULT_TCP_PORT
    udp_port: int = DEFAULT_UDP_PORT

    # Liveness
    heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S
    reconnect_delay_s: float = 2.0
    socket_timeout_s: float = 1.0

    # Detection logic
    # Energy / confidence below these are ignored even if device says detected.
    energy_threshold: float = 0.15
    confidence_threshold: float = 0.55
    # Smoothing factor (EMA) for doa_deg and energy. 0 disables.
    smoothing_alpha: float = 0.3
    # Minimum seconds between two detection callbacks (debounce).
    detection_debounce_s: float = 1.5

    # Fallbacks and behaviour
    # If True and the real transport fails to start, fall back to simulation.
    allow_simulation_fallback: bool = True
    # Simulation cadence
    sim_heartbeat_period_s: float = 1.0
    sim_detection_period_s: float = 7.0

    @classmethod
    def from_env(cls, prefix: str = "ADAS3_ACOUSTIC_") -> "AcousticArrayConfig":
        """Build a config from environment variables for ops convenience."""

        def _get(name: str, default: Any) -> Any:
            return os.environ.get(prefix + name, default)

        def _b(name: str, default: bool) -> bool:
            v = _get(name, None)
            if v is None:
                return default
            return str(v).strip().lower() in ("1", "true", "yes", "on")

        def _f(name: str, default: float) -> float:
            try:
                return float(_get(name, default))
            except (TypeError, ValueError):
                return default

        def _i(name: str, default: int) -> int:
            try:
                return int(_get(name, default))
            except (TypeError, ValueError):
                return default

        cfg = cls()
        cfg.enabled = _b("ENABLE", cfg.enabled)
        cfg.transport = str(_get("TRANSPORT", cfg.transport)).lower()
        cfg.port = _get("PORT", cfg.port) or None
        cfg.baudrate = _i("BAUDRATE", cfg.baudrate)
        cfg.host = _get("HOST", cfg.host)
        cfg.tcp_port = _i("TCP_PORT", cfg.tcp_port)
        cfg.udp_port = _i("UDP_PORT", cfg.udp_port)
        cfg.heartbeat_timeout_s = _f("HEARTBEAT_TIMEOUT_S", cfg.heartbeat_timeout_s)
        cfg.reconnect_delay_s = _f("RECONNECT_DELAY_S", cfg.reconnect_delay_s)
        cfg.energy_threshold = _f("ENERGY_THRESHOLD", cfg.energy_threshold)
        cfg.confidence_threshold = _f("CONFIDENCE_THRESHOLD", cfg.confidence_threshold)
        cfg.smoothing_alpha = _f("SMOOTHING_ALPHA", cfg.smoothing_alpha)
        cfg.detection_debounce_s = _f("DETECTION_DEBOUNCE_S", cfg.detection_debounce_s)
        cfg.allow_simulation_fallback = _b(
            "ALLOW_SIMULATION_FALLBACK", cfg.allow_simulation_fallback
        )
        return cfg


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class AcousticArrayState:
    """Snapshot of the array state. Updated atomically by the worker."""

    connected: bool = False
    transport: str = ""
    last_seen: float = 0.0  # monotonic seconds
    last_message_at: float = 0.0  # wall-clock unix seconds

    mic_count: int = 0
    firmware: str = ""

    doa_deg: Optional[float] = None
    energy: float = 0.0
    confidence: float = 0.0
    detected: bool = False

    # Last reported pair/bus this acoustic event came from, if the client
    # bothers to send it (e.g. "A" or "B"). Optional metadata, never blocks
    # processing.
    pair: Optional[str] = None
    bus: Optional[str] = None

    # Physical wiring snapshot. Starts as the canonical default (set in the
    # client constructor) and is overwritten verbatim if the client sends
    # its own `wiring` / `config` metadata in a heartbeat or acoustic event.
    wiring: dict = field(default_factory=dict)
    wiring_source: str = "default"  # "default" | "payload"

    messages_received: int = 0
    parse_errors: int = 0
    last_error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def is_alive(self, timeout_s: float, now: Optional[float] = None) -> bool:
        if not self.connected or self.last_seen == 0.0:
            return False
        cur = now if now is not None else time.monotonic()
        return (cur - self.last_seen) <= timeout_s


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Field names accepted in JSON messages. Extras are ignored.
#
# New optional metadata keys (since the 4-mic / 2-pair wiring became
# definitive): `pair`, `bus`, `wiring`, `config`. They flow through the
# parser untouched; the worker decides what to store in the state and which
# of them are authoritative against DEFAULT_WIRING.
_ACOUSTIC_FIELDS = {
    "doa_deg",
    "energy",
    "confidence",
    "detected",
    "mic_count",
    "firmware",
    "label",
    "pair",
    "bus",
    "wiring",
    "config",
}


def parse_message(line: str) -> Optional[dict]:
    """Parse one wire line into a normalised dict, or None on hard failure.

    Accepts either JSON (preferred) or CSV legacy form. Returns dict with
    at least the field ``type`` ("heartbeat" | "acoustic" | "unknown").
    """
    if line is None:
        return None
    s = line.strip()
    if not s:
        return None

    # JSON path
    if s[0] in "{[":
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None
        msg_type = str(obj.get("type", "")).lower()
        if msg_type not in ("heartbeat", "acoustic"):
            msg_type = "acoustic" if "doa_deg" in obj or "detected" in obj else "unknown"
        out: dict = {"type": msg_type}
        for k in _ACOUSTIC_FIELDS:
            if k in obj:
                out[k] = obj[k]
        return out

    # CSV legacy: DOA,energy,confidence,detected
    parts = [p.strip() for p in s.split(",")]
    if len(parts) >= 3:
        try:
            doa = float(parts[0])
            energy = float(parts[1])
            confidence = float(parts[2])
            detected = False
            if len(parts) >= 4:
                detected = parts[3].lower() in ("1", "true", "yes", "t")
            return {
                "type": "acoustic",
                "doa_deg": doa,
                "energy": energy,
                "confidence": confidence,
                "detected": detected,
            }
        except ValueError:
            return None

    return None


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class _Transport:
    """Tiny interface: open(), readline() -> Optional[str], close()."""

    def open(self) -> None:
        raise NotImplementedError

    def readline(self) -> Optional[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def name(self) -> str:
        return type(self).__name__


class _SerialTransport(_Transport):
    def __init__(self, port: Optional[str], baudrate: int, timeout: float) -> None:
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._serial: Any = None
        self._buf = b""

    def _autodetect(self) -> Optional[str]:
        try:
            from serial.tools import list_ports  # type: ignore
        except Exception:
            return None
        for p in list_ports.comports():
            desc = (p.description or "") + " " + (p.manufacturer or "")
            if any(tag in desc.lower() for tag in ("esp32", "cp210", "ch340", "ftdi", "silicon")):
                return p.device
        # As a last resort return the first available port
        ports = list(list_ports.comports())
        return ports[0].device if ports else None

    def open(self) -> None:
        import serial  # type: ignore  # raises if pyserial missing
        port = self._port or self._autodetect()
        if not port:
            raise RuntimeError("no serial port available")
        self._serial = serial.Serial(port, self._baudrate, timeout=self._timeout)
        log.info("acoustic array serial opened: %s @ %d", port, self._baudrate)

    def readline(self) -> Optional[str]:
        if self._serial is None:
            return None
        try:
            data = self._serial.readline()
        except Exception as e:
            raise RuntimeError(f"serial read error: {e}") from e
        if not data:
            return None
        try:
            return data.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

    def close(self) -> None:
        try:
            if self._serial is not None:
                self._serial.close()
        except Exception:
            pass
        self._serial = None


class _TcpTransport(_Transport):
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buf = b""

    def open(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self._timeout)
        s.connect((self._host, self._port))
        self._sock = s
        log.info("acoustic array tcp connected: %s:%d", self._host, self._port)

    def readline(self) -> Optional[str]:
        if self._sock is None:
            return None
        while b"\n" not in self._buf:
            try:
                chunk = self._sock.recv(1024)
            except socket.timeout:
                return None
            except Exception as e:
                raise RuntimeError(f"tcp read error: {e}") from e
            if not chunk:
                raise RuntimeError("tcp peer closed")
            self._buf += chunk
        line, _, rest = self._buf.partition(b"\n")
        self._buf = rest
        return line.decode("utf-8", errors="replace").strip()

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._buf = b""


class _UdpTransport(_Transport):
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None

    def open(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self._host, self._port))
        s.settimeout(self._timeout)
        self._sock = s
        log.info("acoustic array udp listening: %s:%d", self._host, self._port)

    def readline(self) -> Optional[str]:
        if self._sock is None:
            return None
        try:
            data, _ = self._sock.recvfrom(2048)
        except socket.timeout:
            return None
        except Exception as e:
            raise RuntimeError(f"udp read error: {e}") from e
        if not data:
            return None
        return data.decode("utf-8", errors="replace").strip()

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None


class _SimulationTransport(_Transport):
    """Emits realistic-looking heartbeats and occasional detections."""

    # Hard lower bound — keeps the worker from busy-spinning if the caller
    # passes 0 or a negative value. The worker also sleeps 0.1s per idle
    # tick, so the effective minimum is in any case ~0.1s.
    _MIN_PERIOD_S = 0.05

    def __init__(self, hb_period: float, det_period: float) -> None:
        self._hb_period = max(self._MIN_PERIOD_S, float(hb_period))
        self._det_period = max(self._MIN_PERIOD_S, float(det_period))
        self._next_hb = 0.0
        self._next_det = 0.0
        self._open = False

    def open(self) -> None:
        now = time.monotonic()
        # Fire the first heartbeat almost immediately and the first detection
        # one period later so callers that sleep ~det_period still observe at
        # least one detection event.
        self._next_hb = now
        self._next_det = now + self._det_period
        self._open = True
        log.info("acoustic array simulation transport active")

    def readline(self) -> Optional[str]:
        if not self._open:
            return None
        # Sleep a small slice so the worker does not busy-loop.
        time.sleep(0.1)
        now = time.monotonic()
        if now >= self._next_det:
            self._next_det = now + self._det_period
            doa = round(random.uniform(-180.0, 180.0), 1)
            energy = round(random.uniform(0.3, 0.95), 2)
            conf = round(random.uniform(0.6, 0.95), 2)
            # Alternate the originating pair so the overlay/status code is
            # exercised against both buses.
            pair = "A" if int(now) % 2 == 0 else "B"
            return json.dumps(
                {
                    "type": "acoustic",
                    "detected": True,
                    "doa_deg": doa,
                    "energy": energy,
                    "confidence": conf,
                    "mic_count": 4,
                    "pair": pair,
                }
            )
        if now >= self._next_hb:
            self._next_hb = now + self._hb_period
            # Heartbeat carries the definitive wiring so an end-to-end
            # simulation looks exactly like the real device.
            return json.dumps(
                {
                    "type": "heartbeat",
                    "mic_count": 4,
                    "firmware": "adas-array-sim-0.1",
                    "wiring": DEFAULT_WIRING.to_dict(),
                }
            )
        return None

    def close(self) -> None:
        self._open = False


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


EventCallback = Callable[[str, dict, "AcousticArrayState"], None]


class AcousticArrayClient:
    """Background client that owns the transport, parses events and exposes state.

    Callers register a thread-safe callback via ``on_event``. The callback is
    invoked with ``(event_type, payload, state_snapshot)``. ``event_type`` is
    one of ``"heartbeat"``, ``"acoustic"``, ``"detection"``, ``"connected"``,
    ``"disconnected"`` or ``"error"``. The callback runs on the worker thread
    and SHOULD NOT block; UI code should marshal back to the Tk/main thread.
    """

    def __init__(self, config: Optional[AcousticArrayConfig] = None) -> None:
        self._cfg = config or AcousticArrayConfig()
        self._state = AcousticArrayState()
        self._state.transport = self._cfg.transport
        # Seed the state with the canonical 4-mic / 2-pair wiring. The client
        # will overwrite this verbatim if the device sends its own wiring
        # block in any payload (heartbeat or acoustic). mic_count defaults to
        # 4 so the overlay/status text shows mic=4 even before the first
        # heartbeat lands.
        self._state.wiring = default_wiring_dict()
        self._state.wiring_source = "default"
        self._state.mic_count = DEFAULT_WIRING.mic_count
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[EventCallback] = []
        self._cb_lock = threading.Lock()
        self._last_detection_at = 0.0
        self._using_simulation = False

    # --- public API -------------------------------------------------------

    @property
    def config(self) -> AcousticArrayConfig:
        return self._cfg

    def get_state(self) -> AcousticArrayState:
        """Return a copy of the current state."""
        with self._lock:
            return AcousticArrayState(**self._state.to_dict())

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def on_event(self, cb: EventCallback) -> None:
        with self._cb_lock:
            self._callbacks.append(cb)

    def start(self) -> bool:
        """Start the worker. Returns True if a thread was started."""
        if not self._cfg.enabled:
            log.info("acoustic array disabled by config")
            return False
        if self.is_running():
            return True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="acoustic-array-worker", daemon=True
        )
        self._thread.start()
        return True

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=join_timeout)
        self._thread = None

    # --- worker -----------------------------------------------------------

    def _build_transport(self) -> _Transport:
        cfg = self._cfg
        mode = (cfg.transport or "").lower()
        if mode == "simulation":
            self._using_simulation = True
            return _SimulationTransport(cfg.sim_heartbeat_period_s, cfg.sim_detection_period_s)
        if mode == "tcp":
            return _TcpTransport(cfg.host, cfg.tcp_port, cfg.socket_timeout_s)
        if mode == "udp":
            return _UdpTransport(cfg.host, cfg.udp_port, cfg.socket_timeout_s)
        if mode == "serial":
            return _SerialTransport(cfg.port, cfg.baudrate, cfg.socket_timeout_s)
        raise ValueError(f"unknown transport: {cfg.transport!r}")

    def _run(self) -> None:
        log.info("acoustic array worker starting (transport=%s)", self._cfg.transport)
        while not self._stop.is_set():
            transport: Optional[_Transport] = None
            try:
                transport = self._build_transport()
                transport.open()
            except Exception as e:
                msg = f"open failed: {e}"
                log.warning("acoustic array %s", msg)
                self._set_error(msg)
                if self._cfg.allow_simulation_fallback and not self._using_simulation:
                    log.info("acoustic array falling back to simulation")
                    self._using_simulation = True
                    with self._lock:
                        self._state.transport = "simulation"
                    transport = _SimulationTransport(
                        self._cfg.sim_heartbeat_period_s, self._cfg.sim_detection_period_s
                    )
                    try:
                        transport.open()
                    except Exception as e2:
                        self._set_error(f"sim open failed: {e2}")
                        if self._stop.wait(self._cfg.reconnect_delay_s):
                            break
                        continue
                else:
                    if self._stop.wait(self._cfg.reconnect_delay_s):
                        break
                    continue

            self._mark_connected(True)
            self._notify("connected", {"transport": self._state.transport})
            try:
                self._read_loop(transport)
            except Exception as e:
                self._set_error(str(e))
                log.warning("acoustic array read loop ended: %s", e)
            finally:
                try:
                    transport.close()
                except Exception:
                    pass
                self._mark_connected(False)
                self._notify("disconnected", {})

            if self._stop.is_set():
                break
            if self._stop.wait(self._cfg.reconnect_delay_s):
                break

        log.info("acoustic array worker stopped")

    def _read_loop(self, transport: _Transport) -> None:
        cfg = self._cfg
        while not self._stop.is_set():
            line = transport.readline()
            if line is None:
                # idle tick: just check heartbeat timeout
                self._check_alive()
                continue
            msg = parse_message(line)
            if msg is None:
                with self._lock:
                    self._state.parse_errors += 1
                continue
            self._handle_message(msg)

    def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("type", "unknown")
        now_mono = time.monotonic()
        now_wall = time.time()
        with self._lock:
            self._state.messages_received += 1
            self._state.last_seen = now_mono
            self._state.last_message_at = now_wall
            if "mic_count" in msg:
                try:
                    self._state.mic_count = int(msg["mic_count"])
                except (TypeError, ValueError):
                    pass
            if "firmware" in msg and isinstance(msg["firmware"], str):
                self._state.firmware = msg["firmware"]

            # Optional enriched metadata. Accept whichever the firmware sends;
            # if both `wiring` and `config` are present we treat `wiring` as
            # the authoritative one (config is allowed as a synonym for older
            # clients). Anything we receive is preserved verbatim and marked
            # as `payload`-sourced. If nothing comes, we keep the default
            # wiring seeded in __init__.
            wiring_in = msg.get("wiring")
            if wiring_in is None:
                wiring_in = msg.get("config")
            if isinstance(wiring_in, dict) and wiring_in:
                self._state.wiring = dict(wiring_in)
                self._state.wiring_source = "payload"
                try:
                    if "mic_count" in wiring_in:
                        self._state.mic_count = int(wiring_in["mic_count"])
                except (TypeError, ValueError):
                    pass

            pair_in = msg.get("pair")
            if isinstance(pair_in, str) and pair_in:
                self._state.pair = pair_in.upper()
            bus_in = msg.get("bus")
            if isinstance(bus_in, str) and bus_in:
                self._state.bus = bus_in

            if msg_type == "acoustic":
                doa = msg.get("doa_deg")
                energy = msg.get("energy")
                confidence = msg.get("confidence")
                detected = bool(msg.get("detected", False))
                alpha = self._cfg.smoothing_alpha

                if isinstance(doa, (int, float)) and math.isfinite(float(doa)):
                    d = float(doa)
                    if self._state.doa_deg is None or alpha <= 0:
                        self._state.doa_deg = d
                    else:
                        prev = self._state.doa_deg
                        # angular smoothing: shortest-arc EMA
                        diff = (d - prev + 540.0) % 360.0 - 180.0
                        self._state.doa_deg = (prev + alpha * diff + 540.0) % 360.0 - 180.0
                if isinstance(energy, (int, float)):
                    e = float(energy)
                    self._state.energy = (
                        e if alpha <= 0 else (1 - alpha) * self._state.energy + alpha * e
                    )
                if isinstance(confidence, (int, float)):
                    self._state.confidence = float(confidence)

                gated_detection = (
                    detected
                    and self._state.energy >= self._cfg.energy_threshold
                    and self._state.confidence >= self._cfg.confidence_threshold
                )
                self._state.detected = gated_detection
                snapshot = AcousticArrayState(**self._state.to_dict())
            else:
                snapshot = AcousticArrayState(**self._state.to_dict())

        # Dispatch outside the lock
        self._notify(msg_type, msg, snapshot=snapshot)
        if msg_type == "acoustic" and snapshot.detected:
            if (now_mono - self._last_detection_at) >= self._cfg.detection_debounce_s:
                self._last_detection_at = now_mono
                self._notify("detection", msg, snapshot=snapshot)

    def _check_alive(self) -> None:
        now = time.monotonic()
        with self._lock:
            was_connected = self._state.connected
            alive = self._state.is_alive(self._cfg.heartbeat_timeout_s, now)
            if was_connected and not alive and self._state.last_seen > 0:
                self._state.connected = False
                self._state.last_error = "heartbeat timeout"
                snapshot = AcousticArrayState(**self._state.to_dict())
                fire = True
            else:
                fire = False
                snapshot = None
        if fire and snapshot is not None:
            self._notify("disconnected", {"reason": "heartbeat timeout"}, snapshot=snapshot)

    def _mark_connected(self, connected: bool) -> None:
        with self._lock:
            self._state.connected = connected
            if connected:
                self._state.last_error = ""

    def _set_error(self, err: str) -> None:
        with self._lock:
            self._state.last_error = err

    def _notify(
        self,
        event_type: str,
        payload: dict,
        snapshot: Optional[AcousticArrayState] = None,
    ) -> None:
        if snapshot is None:
            snapshot = self.get_state()
        with self._cb_lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(event_type, payload, snapshot)
            except Exception as e:
                log.exception("acoustic array callback failed: %s", e)


# ---------------------------------------------------------------------------
# Module-level singleton helpers (used by testcam.py)
# ---------------------------------------------------------------------------

_singleton: Optional[AcousticArrayClient] = None
_singleton_lock = threading.Lock()


def start_acoustic_array(
    config: Optional[AcousticArrayConfig] = None,
    on_event: Optional[EventCallback] = None,
) -> Optional[AcousticArrayClient]:
    """Start (or return existing) singleton client. Returns None if disabled."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None and _singleton.is_running():
            if on_event is not None:
                _singleton.on_event(on_event)
            return _singleton
        cfg = config or AcousticArrayConfig.from_env()
        if not cfg.enabled:
            log.info("acoustic array: not started (disabled)")
            return None
        client = AcousticArrayClient(cfg)
        if on_event is not None:
            client.on_event(on_event)
        if not client.start():
            return None
        _singleton = client
        return client


def stop_acoustic_array() -> None:
    global _singleton
    with _singleton_lock:
        c = _singleton
        _singleton = None
    if c is not None:
        c.stop()


def get_client() -> Optional[AcousticArrayClient]:
    return _singleton
