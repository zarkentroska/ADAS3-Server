# ESP32 Acoustic Array — Server-side integration

This document covers the ADAS3-Server side of the new acoustic array. The
firmware on the ESP32 is out of scope here; only the data contract and how the
server consumes it.

## 1. Why an external module

`testcam.py` already mixes Tk, OpenCV, YOLO, audio ML, TinySA scanraw and
Telegram alerts. Adding I2S DSP / protocol parsing inside it would make it
even harder to read. The acoustic array is therefore split into two files:

```
modules/esp32_acoustic_array.py   <- transport, parsing, threading, state
acoustic_integration.py           <- 4-function bridge consumed by testcam.py
```

`testcam.py` gains exactly 4 lines (import, init, per-frame overlay,
shutdown). No protocol or DSP detail leaks into it.

## 2. Hardware wiring (definitive)

The array uses **four** I2S MEMS microphones (SPH0645 or INMP441) wired as
two stereo pairs, each pair on its own I2S bus. Every mic is powered from
the same ESP32 3V3 rail in parallel; ground is common to all mics and the
PC817 opto-isolator. The channel-select pin (SEL) is **hardwired locally**
on each mic — no GPIO is spent on it — to GND for the LEFT side and to 3V3
for the RIGHT side of each pair.

This is the canonical layout the server uses as `DEFAULT_WIRING`. The
firmware/client may also send the wiring inside `heartbeat` or `acoustic`
payloads (under a `wiring` or `config` key); when present it is preserved
verbatim in the server state. When absent, the server falls back to this
default.

### 2.1 Power and ground

| Net      | ESP32 pin | Connects to              |
|----------|-----------|--------------------------|
| 3V3      | 3V3       | Mic1, Mic2, Mic3, Mic4   |
| GND      | GND       | Mic1..Mic4, PC817 common |

### 2.2 Channel select (local, no GPIO)

| Mic  | SEL strapped to | Resulting channel | I2S pair |
|------|-----------------|-------------------|----------|
| Mic1 | GND             | LEFT              | A        |
| Mic2 | 3V3             | RIGHT             | A        |
| Mic3 | GND             | LEFT              | B        |
| Mic4 | 3V3             | RIGHT             | B        |

### 2.3 I2S pair A — Mic1 (L) + Mic2 (R)

| Signal | ESP32 GPIO |
|--------|-----------:|
| BCLK   | 14         |
| LRCL   | 13         |
| DOUT   | 34         |

### 2.4 I2S pair B — Mic3 (L) + Mic4 (R)

| Signal | ESP32 GPIO |
|--------|-----------:|
| BCLK   | 22         |
| LRCL   | 21         |
| DOUT   | 35         |

### 2.5 YT2000 / PC817 opto-coupled remote control

| Direction | ESP32 GPIO |
|-----------|-----------:|
| UP        | 26         |
| DOWN      | 27         |
| LEFT      | 32         |
| RIGHT     | 33         |

### 2.6 Layout sketch

```
  Pair A (I2S bus 0, GPIO 14/13/34)            Pair B (I2S bus 1, GPIO 22/21/35)
  +-----+              +-----+                 +-----+              +-----+
  | M1  |-- 5 cm -----| M2   |                 | M3  |-- 5 cm -----| M4   |
  | L   |             | R    |                 | L   |             | R    |
  | SEL=GND           | SEL=3V3                 | SEL=GND           | SEL=3V3
  +-----+              +-----+                 +-----+              +-----+

                            common 3V3 + GND from ESP32
```

The pairs do not need to share a clock domain because they live on
independent I2S peripherals; the cross-pair delay estimate is recomputed in
firmware after timestamp alignment.

## 3. Wire protocol (ESP32 → Server)

The link can be USB CDC serial, Bluetooth SPP, TCP or UDP. Raw audio is
explicitly NOT sent — only lightweight events. One line per message,
UTF-8 encoded, terminated by `\n`.

### 3.1 Heartbeat (recommended every ~1 s)

Minimal form (still accepted):

```json
{"type":"heartbeat","mic_count":4,"firmware":"adas-array-0.1"}
```

Enriched form (recommended for the definitive 4-mic wiring — the server
echoes it back in status endpoints and in the internal `acoustic_array`
event):

```json
{
  "type": "heartbeat",
  "mic_count": 4,
  "firmware": "adas-array-0.2",
  "wiring": {
    "power_rail": "3V3",
    "common_ground": "GND",
    "mic_count": 4,
    "mics": [
      {"index": 1, "pair": "A", "side": "LEFT",  "sel_to": "GND"},
      {"index": 2, "pair": "A", "side": "RIGHT", "sel_to": "3V3"},
      {"index": 3, "pair": "B", "side": "LEFT",  "sel_to": "GND"},
      {"index": 4, "pair": "B", "side": "RIGHT", "sel_to": "3V3"}
    ],
    "buses": [
      {"pair": "A", "bclk_gpio": 14, "lrcl_gpio": 13, "dout_gpio": 34,
       "left_mic": 1, "right_mic": 2},
      {"pair": "B", "bclk_gpio": 22, "lrcl_gpio": 21, "dout_gpio": 35,
       "left_mic": 3, "right_mic": 4}
    ],
    "remote_control": {
      "up_gpio": 26, "down_gpio": 27, "left_gpio": 32, "right_gpio": 33
    }
  }
}
```

`config` is accepted as a synonym of `wiring` for backwards compatibility.
The server uses heartbeats to confirm the link is alive. Missing heartbeats
for `heartbeat_timeout_s` seconds (default 5) flag the array as
disconnected.

### 3.2 Acoustic event

Minimal form (still accepted):

```json
{"type":"acoustic","detected":true,"doa_deg":35.0,
 "energy":0.72,"confidence":0.81,"mic_count":4}
```

Enriched form (carries the originating I2S pair and, optionally, the
wiring used to capture this event):

```json
{
  "type": "acoustic",
  "detected": true,
  "doa_deg": 35.0,
  "energy": 0.72,
  "confidence": 0.81,
  "mic_count": 4,
  "pair": "A",
  "bus": "i2s0"
}
```

Fields:

| Field        | Type    | Range / unit            | Notes |
|--------------|---------|-------------------------|-------|
| `type`       | string  | `"acoustic"`            | Required |
| `detected`   | bool    | true/false              | Local detector decision |
| `doa_deg`    | float   | -180..180 (or 0..360)   | Azimuth in degrees, 0 = front |
| `energy`     | float   | 0..1                    | RMS-normalised or similar |
| `confidence` | float   | 0..1                    | Detector score |
| `mic_count`  | int     | 1..8                    | Optional, helps debug |
| `pair`       | string  | `"A"` / `"B"`           | Optional, originating I2S pair |
| `bus`        | string  | free-form               | Optional, originating I2S bus id |
| `wiring`     | object  | see 3.1                 | Optional, overrides default wiring |
| `config`     | object  | see 3.1                 | Optional, synonym of `wiring` |

If `wiring` (or `config`) is absent the server applies `DEFAULT_WIRING`
from `modules.esp32_acoustic_array`, which encodes the definitive 4-mic /
2-pair layout described in §2.

The server applies additional gating: a detection only "counts" if it passes
`energy_threshold` AND `confidence_threshold` (both configurable). DOA is
smoothed with a shortest-arc EMA before being shown in the UI.

### 3.3 Legacy CSV (fallback for very small firmwares)

If the firmware author prefers a smaller message, the server also accepts:

```
35.0,0.72,0.81,1
```

i.e. `doa_deg,energy,confidence,detected` (last field is 0/1). The server
parses this as an `acoustic` event with `type` filled in automatically.

## 4. Server configuration

Configured via `AcousticArrayConfig` or environment variables. The latter is
convenient for ops/PyInstaller builds because no code change is needed.

| Env var                                  | Default       | Meaning |
|------------------------------------------|---------------|---------|
| `ADAS3_ACOUSTIC_ENABLE`                  | `1`           | Master on/off |
| `ADAS3_ACOUSTIC_TRANSPORT`               | `serial`      | `serial`/`tcp`/`udp`/`simulation` |
| `ADAS3_ACOUSTIC_PORT`                    | autodetect    | Serial port, e.g. `/dev/ttyUSB0`, `COM7` |
| `ADAS3_ACOUSTIC_BAUDRATE`                | `115200`      | Serial baudrate |
| `ADAS3_ACOUSTIC_HOST`                    | `0.0.0.0`     | TCP target / UDP bind |
| `ADAS3_ACOUSTIC_TCP_PORT`                | `5005`        | TCP |
| `ADAS3_ACOUSTIC_UDP_PORT`                | `5005`        | UDP |
| `ADAS3_ACOUSTIC_HEARTBEAT_TIMEOUT_S`     | `5.0`         | Link liveness |
| `ADAS3_ACOUSTIC_ENERGY_THRESHOLD`        | `0.15`        | Server-side gate |
| `ADAS3_ACOUSTIC_CONFIDENCE_THRESHOLD`    | `0.55`        | Server-side gate |
| `ADAS3_ACOUSTIC_SMOOTHING_ALPHA`         | `0.3`         | DOA/energy EMA |
| `ADAS3_ACOUSTIC_DETECTION_DEBOUNCE_S`    | `1.5`         | Min seconds between alerts |
| `ADAS3_ACOUSTIC_ALLOW_SIMULATION_FALLBACK` | `1`         | If serial fails, run in sim |

## 5. Simulation mode

To validate the UI without any ESP32 attached:

```bash
ADAS3_ACOUSTIC_ENABLE=1 \
ADAS3_ACOUSTIC_TRANSPORT=simulation \
python testcam.py
```

You should see:

* A green `ARRAY OK [simulation]` badge on the video.
* The DOA arrow rotating to a random heading every ~7 seconds.
* A `[ARRAY] detection ...` line in stdout each time the gate fires.
* `[ARRAY] alert hook error` lines if you forget to register the hook.

If the real link is configured but the device is not present, the worker
emits a single warning and (when `allow_simulation_fallback=1`) flips
itself into simulation transport automatically so the UI keeps working.

## 6. Smoke tests

```bash
cd <repo>
python -m unittest tests.test_esp32_acoustic_array -v
```

The unit tests exercise the parser (JSON, JSON malformed, CSV legacy, empty
lines) and the simulated client (state, debounce, callback dispatch). They
do not require pyserial or hardware.

## 7. Fusion notes

The acoustic array is a confirmation sensor, NOT a primary alerter:

* Telegram alerts continue to be driven by the phone-microphone audio ML,
  YOLO and TinySA pipelines as today. The acoustic-array detection
  callback **must not** post to Telegram directly — `acoustic_integration`
  enforces this by only firing an internal `acoustic_array` event towards
  the Android client (see `handle_acoustic_array_detection` in
  `testcam.py`).
* The DOA can be fed into the autotracking module to bias the turret toward
  the acoustic bearing when YOLO loses the target.
* Inside the fusion table (if/when implemented), assign acoustic-array a
  weight similar to the phone microphone but with higher trust when DOA is
  consistent across several seconds (i.e. low variance after EMA).

## 8. Internal `acoustic_array` event payload

For each debounced detection the integration layer hands the registered
hook a dict like:

```python
{
    "source": "acoustic_array",
    "doa_deg": 35.0,
    "energy": 0.72,
    "confidence": 0.81,
    "mic_count": 4,
    "pair": "A",           # last reported pair, or None
    "bus": "i2s0",         # last reported bus id, or None
    "wiring": { ... },     # last seen wiring or DEFAULT_WIRING
    "wiring_source": "default" | "payload",
    "timestamp": 1715600000.0,
}
```

`testcam.py` then enqueues this as an internal `acoustic_array` event
towards the Android client. No Telegram path is triggered.
