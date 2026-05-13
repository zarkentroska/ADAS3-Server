"""
Drop-in patch fragments for testcam.py.

This file is documentation, not runnable. Copy the blocks into your existing
testcam.py at the indicated anchors. The integration is intentionally tiny:
testcam.py never touches the ESP32 protocol directly.

Total invasive lines in testcam.py: 4.
"""

# ---------------------------------------------------------------------------
# 1) Imports — add near the top of testcam.py, after the other imports.
# ---------------------------------------------------------------------------
from acoustic_integration import (
    acoustic_init,
    acoustic_overlay,
    acoustic_shutdown,
    acoustic_status_text,
)

# Optional: build an explicit config from environment vars or hard-code one.
# from modules.esp32_acoustic_array import AcousticArrayConfig
# _ACOUSTIC_CFG = AcousticArrayConfig(
#     enabled=True,
#     transport="serial",        # "serial" | "tcp" | "udp" | "simulation"
#     port=None,                 # autodetect ESP32 USB CDC
#     baudrate=115200,
#     heartbeat_timeout_s=5.0,
#     detection_debounce_s=1.5,
#     energy_threshold=0.15,
#     confidence_threshold=0.55,
#     allow_simulation_fallback=True,
# )


# ---------------------------------------------------------------------------
# 2) Startup — call once during program init, near where the other workers
#    (TinySA, audio, YOLO) are wired up. Pass your existing telegram-alert
#    function (or any callable). Order does not matter; the worker is lazy.
# ---------------------------------------------------------------------------

def _on_acoustic_alert(payload):
    """Fired on debounced acoustic detections. Wire into the existing alert
    pipeline. Keep this small — it runs on a background thread."""
    try:
        # Example: piggy-back the global alert pipeline if available.
        # The acoustic array is a confirmation sensor: do not send a separate
        # Telegram clip — let the existing audio/YOLO/RF pipeline gate that.
        # If you have a fusion table, increase its acoustic-array score here.
        # If you don't, log it and surface in the UI status bar.
        print(
            f"[ARRAY] detection doa={payload.get('doa_deg')} "
            f"energy={payload.get('energy'):.2f} conf={payload.get('confidence'):.2f}"
        )
        # Hook into an existing fusion module if present (no-op if not).
        try:
            import fusion  # type: ignore
            fusion.add_score(
                "acoustic_array",
                payload.get("confidence", 0.0),
                meta=payload,
            )
        except Exception:
            pass
    except Exception as e:
        print(f"[ARRAY] alert hook error: {e}")


acoustic_init(alert_callback=_on_acoustic_alert)


# ---------------------------------------------------------------------------
# 3) Per-frame overlay — inside your main render loop, alongside the existing
#    overlay_audio_spectrogram(frame) / overlay_tinysa_graph(frame) calls.
# ---------------------------------------------------------------------------

# frame = overlay_audio_spectrogram(frame)
# frame = overlay_tinysa_graph(frame)
frame = acoustic_overlay(frame)  # NEW: ESP32 acoustic array badge + DOA arrow

# Optional: also log the status periodically (every N frames or every K seconds).
# print(acoustic_status_text())


# ---------------------------------------------------------------------------
# 4) Shutdown — call once at the very end, just before the program exits or
#    closes the OpenCV windows. Joins the worker thread cleanly.
# ---------------------------------------------------------------------------
acoustic_shutdown()
