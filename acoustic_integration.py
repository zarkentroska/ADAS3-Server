"""
Bridge between testcam.py and modules.esp32_acoustic_array.

Goal: keep testcam.py free of protocol/DSP details. testcam.py only calls four
things from this module:

    from acoustic_integration import (
        acoustic_init, acoustic_overlay, acoustic_shutdown, acoustic_status_text,
    )

    acoustic_init(alert_callback=enviar_alerta_telegram)   # at startup
    ...
    frame = acoustic_overlay(frame)                        # inside render loop
    ...
    acoustic_shutdown()                                    # on exit

Everything else (transport, parsing, threading, smoothing, debounce) lives
inside modules/esp32_acoustic_array.py.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from modules.esp32_acoustic_array import (
    AcousticArrayClient,
    AcousticArrayConfig,
    AcousticArrayState,
    start_acoustic_array,
    stop_acoustic_array,
    get_client,
)


log = logging.getLogger("adas3.acoustic_integration")


# Latest snapshot, updated atomically by the worker callback. The render thread
# in testcam.py reads it without locking — a stale frame is harmless.
_latest_state: Optional[AcousticArrayState] = None
_state_lock = threading.Lock()

# External alert hook (e.g. enviar_alerta_telegram). Called only on debounced
# detections by the acoustic module, with a small descriptor dict.
_alert_callback: Optional[Callable[[dict], None]] = None


def acoustic_init(
    alert_callback: Optional[Callable[[dict], None]] = None,
    config: Optional[AcousticArrayConfig] = None,
) -> Optional[AcousticArrayClient]:
    """Start the acoustic array client.

    ``alert_callback`` receives a small dict on every debounced detection:

        {
            "source": "acoustic_array",
            "doa_deg": 35.0,
            "energy": 0.72,
            "confidence": 0.81,
            "mic_count": 4,
            "timestamp": 1715600000.0,
        }

    Safe to call even if pyserial is missing or no hardware is plugged in —
    the underlying module falls back to simulation if configured.
    """
    global _alert_callback
    _alert_callback = alert_callback

    try:
        client = start_acoustic_array(config=config, on_event=_on_event)
    except Exception as e:
        log.exception("acoustic_init failed: %s", e)
        return None
    if client is None:
        log.info("acoustic array not started (disabled)")
        return None
    log.info(
        "acoustic array started: transport=%s",
        client.config.transport,
    )
    return client


def acoustic_shutdown() -> None:
    try:
        stop_acoustic_array()
    except Exception as e:
        log.warning("acoustic_shutdown error: %s", e)


def acoustic_state() -> Optional[AcousticArrayState]:
    with _state_lock:
        return _latest_state


def acoustic_status_text() -> str:
    """Short human-readable status, suitable for a status bar or log line."""
    st = acoustic_state()
    if st is None:
        return "ACOUSTIC ARRAY: off"
    if not st.connected:
        return "ACOUSTIC ARRAY: disconnected"
    doa = "--" if st.doa_deg is None else f"{st.doa_deg:+.0f}deg"
    flag = "DET" if st.detected else "idle"
    return (
        f"ACOUSTIC ARRAY [{st.transport}] mic={st.mic_count} "
        f"{flag} {doa} E={st.energy:.2f} C={st.confidence:.2f}"
    )


def _on_event(event_type: str, payload: dict, snapshot: AcousticArrayState) -> None:
    """Worker-thread callback. Keep this short and non-blocking."""
    global _latest_state
    with _state_lock:
        _latest_state = snapshot

    if event_type == "connected":
        log.info("acoustic array connected (transport=%s)", snapshot.transport)
    elif event_type == "disconnected":
        reason = payload.get("reason", "")
        log.warning("acoustic array disconnected %s", reason or "")
    elif event_type == "detection":
        log.info(
            "acoustic detection: doa=%s energy=%.2f conf=%.2f",
            snapshot.doa_deg,
            snapshot.energy,
            snapshot.confidence,
        )
        cb = _alert_callback
        if cb is not None:
            try:
                cb(
                    {
                        "source": "acoustic_array",
                        "doa_deg": snapshot.doa_deg,
                        "energy": snapshot.energy,
                        "confidence": snapshot.confidence,
                        "mic_count": snapshot.mic_count,
                        "timestamp": snapshot.last_message_at or time.time(),
                    }
                )
            except Exception as e:
                log.exception("acoustic alert callback failed: %s", e)


# ---------------------------------------------------------------------------
# OpenCV overlay (lazy import — testcam.py already pulls cv2 in)
# ---------------------------------------------------------------------------


def acoustic_overlay(frame: Any) -> Any:
    """Draw a small status badge on the frame. Safe no-op if cv2 missing or
    frame is None. Returns the (possibly modified) frame."""
    if frame is None:
        return frame
    st = acoustic_state()
    if st is None:
        return frame
    try:
        import cv2  # type: ignore
    except Exception:
        return frame

    h, w = frame.shape[:2]
    badge_w, badge_h = 260, 70
    x0 = w - badge_w - 10
    y0 = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + badge_w, y0 + badge_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if not st.connected:
        title_color = (60, 60, 200)  # red-ish
        title = "ARRAY OFF"
    elif st.detected:
        title_color = (60, 200, 255)  # amber on detection
        title = "ARRAY DETECT"
    else:
        title_color = (60, 200, 60)  # green on idle
        title = f"ARRAY OK [{st.transport}]"

    cv2.putText(
        frame, title, (x0 + 8, y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, title_color, 1, cv2.LINE_AA,
    )
    doa_txt = "DOA: --" if st.doa_deg is None else f"DOA: {st.doa_deg:+6.1f}deg"
    cv2.putText(
        frame, doa_txt, (x0 + 8, y0 + 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"E:{st.energy:.2f}  C:{st.confidence:.2f}  mic:{st.mic_count}",
        (x0 + 8, y0 + 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # Small DOA arrow if we have a value
    if st.doa_deg is not None:
        import math as _m
        cx = x0 + badge_w - 26
        cy = y0 + badge_h - 22
        r = 14
        cv2.circle(frame, (cx, cy), r, (180, 180, 180), 1, cv2.LINE_AA)
        rad = _m.radians(-st.doa_deg + 90.0)  # 0deg = up
        ex = int(cx + r * _m.cos(rad))
        ey = int(cy - r * _m.sin(rad))
        cv2.line(frame, (cx, cy), (ex, ey), title_color, 2, cv2.LINE_AA)

    return frame


__all__ = [
    "acoustic_init",
    "acoustic_shutdown",
    "acoustic_state",
    "acoustic_status_text",
    "acoustic_overlay",
    "AcousticArrayConfig",
]
