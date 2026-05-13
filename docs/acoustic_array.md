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

## 2. Hardware wiring

The array uses I2S MEMS microphones (SPH0645 or INMP441). They are NOT glued
to the ESP32 — they sit on short twisted-pair leads at a physical separation
of ~4–6.5 cm. Two mics for left/right is the minimum useful layout; four mics
in a bar, cross, square or circle give a proper DOA estimate.

### 2.1 ESP32 ↔ MEMS pins (I2S, common bus)

| Mic pin | Function          | ESP32 pin (typical) | Notes |
|---------|-------------------|---------------------|-------|
| 3V3     | Power             | 3V3                 | Same rail for every mic |
| GND     | Ground            | GND                 | Single star-ground recommended |
| BCLK    | I2S bit clock     | GPIO 26             | Shared by all mics |
| LRCL/WS | Word select       | GPIO 25             | Shared by all mics |
| DOUT    | Data out          | GPIO 22             | One line per pair (L/R via SEL pin) |
| SEL     | Channel select    | GND (L) or 3V3 (R)  | INMP441 / SPH0645 channel pin |

For four microphones, two I2S data lines and the channel-select strap
(SEL=GND on one mic, SEL=VDD on the other in each pair) is enough. The ESP32
S3 can also drive a second I2S peripheral on a separate set of pins for the
second pair.

### 2.2 Physical layout

```
   2-mic bar (left/right)
   +-----+               +-----+
   | M1  |---- 5 cm -----| M2  |
   +-----+               +-----+

   4-mic cross
            +-----+
            | M3  |
            +-----+
              |  5 cm
   +-----+    |    +-----+
   | M1  |---5cm---| M2  |
   +-----+    |    +-----+
              |  5 cm
            +-----+
            | M4  |
            +-----+
```

The exact spacing depends on the frequencies you want to keep below the
spatial-aliasing limit; 4–6.5 cm is the band typically used for drone-rotor
fundamentals (200–2000 Hz). Symmetry matters more than the absolute number.

## 3. Wire protocol (ESP32 → Server)

The link can be USB CDC serial, Bluetooth SPP, TCP or UDP. Raw audio is
explicitly NOT sent — only lightweight events. One line per message,
UTF-8 encoded, terminated by `\n`.

### 3.1 Heartbeat (recommended every ~1 s)

```json
{"type":"heartbeat","mic_count":4,"firmware":"adas-array-0.1"}
```

The server uses these to confirm the link is alive. Missing heartbeats for
`heartbeat_timeout_s` seconds (default 5) flag the array as disconnected.

### 3.2 Acoustic event

```json
{"type":"acoustic","detected":true,"doa_deg":35.0,
 "energy":0.72,"confidence":0.81,"mic_count":4}
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
  YOLO and TinySA pipelines as today.
* The DOA can be fed into the autotracking module to bias the turret toward
  the acoustic bearing when YOLO loses the target.
* Inside the fusion table (if/when implemented), assign acoustic-array a
  weight similar to the phone microphone but with higher trust when DOA is
  consistent across several seconds (i.e. low variance after EMA).
