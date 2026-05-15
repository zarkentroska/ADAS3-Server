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
    DEFAULT_WIRING,
    default_wiring_dict,
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


def acoustic_wiring() -> dict:
    """Return the wiring currently advertised by the array, or the canonical
    default if no payload-sourced wiring has been seen yet. Suitable for a
    status endpoint or a /healthz-style summary."""
    st = acoustic_state()
    if st is not None and isinstance(st.wiring, dict) and st.wiring:
        return dict(st.wiring)
    return default_wiring_dict()


def acoustic_status_text() -> str:
    """Short human-readable status, suitable for a status bar or log line."""
    st = acoustic_state()
    if st is None:
        return "ACOUSTIC ARRAY: off"
    if not st.connected:
        return "ACOUSTIC ARRAY: disconnected"
    doa = "--" if st.doa_deg is None else f"{st.doa_deg:+.0f}deg"
    flag = "DET" if st.detected else "idle"
    pair = f" pair={st.pair}" if st.pair else ""
    return (
        f"ACOUSTIC ARRAY [{st.transport}] mic={st.mic_count}{pair} "
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
            "acoustic detection: doa=%s energy=%.2f conf=%.2f pair=%s",
            snapshot.doa_deg,
            snapshot.energy,
            snapshot.confidence,
            snapshot.pair,
        )
        cb = _alert_callback
        if cb is not None:
            try:
                # NOTE: this callback feeds the internal `acoustic_array`
                # event queued towards the client. It MUST NOT trigger a
                # Telegram alert directly — the phone-mic audio ML is the
                # only Telegram source, to avoid duplicates.
                cb(
                    {
                        "source": "acoustic_array",
                        "doa_deg": snapshot.doa_deg,
                        "energy": snapshot.energy,
                        "confidence": snapshot.confidence,
                        "mic_count": snapshot.mic_count,
                        "pair": snapshot.pair,
                        "bus": snapshot.bus,
                        "wiring": snapshot.wiring,
                        "wiring_source": snapshot.wiring_source,
                        "timestamp": snapshot.last_message_at or time.time(),
                    }
                )
            except Exception as e:
                log.exception("acoustic alert callback failed: %s", e)


# ---------------------------------------------------------------------------
# OpenCV overlay (lazy import — testcam.py already pulls cv2 in)
# ---------------------------------------------------------------------------


def acoustic_overlay(
    frame: Any,
    *,
    ep32_enabled: bool = False,
    anchor: str = "below-dpad",
    y_top: Optional[int] = None,
    badge_w: int = 260,
    badge_h: int = 70,
    show_when_disconnected: bool = False,
    force_show: bool = False,
) -> Any:
    """Draw a small status badge on the frame.

    Visibility policy (changed twice since v0.7):

    - Old behaviour pinned the badge at top-right (x=w-270, y=10) and was
      shown whenever ``acoustic_state()`` reported ``connected=True`` —
      which, with simulation fallback, is *always at startup*. User
      reported ``ARRAY OK (SERIAL) DOA ...`` showing without touching the
      EP32 BT button, and overlapping the EP32 D-pad when the button was
      pressed.
    - New default policy: the badge is **only** drawn when the user has
      EP32 BT enabled (``ep32_enabled=True``). The simulation transport
      no longer leaks "ARRAY OK" into the UI at startup. Pass
      ``show_when_disconnected=True`` to force the badge when the array
      is reporting OFF, or ``force_show=True`` to bypass the EP32 gate
      entirely (debug mode).

    Position policy:

    - ``anchor="below-dpad"`` (default): the badge sits *under* the EP32
      D-pad floating panel — y is far enough down that the panel never
      overlaps the badge, regardless of whether the user has the D-pad
      open. Specifically, ``y_top`` defaults to **484**, which is below
      the 230..476 region taken by ``draw_ep32_floating_controls``.
    - ``anchor="below-ep32"`` (legacy): y=230, below the EP32 BT
      *indicator* but on top of the D-pad. Kept for backwards
      compatibility.
    - ``anchor="top-right"`` (legacy): y=10, the original placement.
    - ``y_top=<int>``: explicit override (testcam can compute the exact
      y after measuring the D-pad panel and pass it here).

    ``ep32_enabled`` should reflect ``Ep32ClientController.is_enabled()``.
    """
    if frame is None:
        return frame
    st = acoustic_state()
    if st is None:
        return frame

    # Visibility gate. The user explicitly does NOT want the badge to be
    # visible while EP32 BT is off, even if the array is "connected" via
    # the simulation transport. ``force_show=True`` lets a future debug
    # toggle bypass this.
    if not force_show and not ep32_enabled:
        return frame
    if not show_when_disconnected and not st.connected and not force_show:
        # If the user turned EP32 on but the array hasn't reported yet,
        # we still draw the badge in OFF state so they know the
        # subsystem exists; that's what the previous gate already did.
        # Only suppress when we *really* have nothing to show AND the
        # caller said so explicitly.
        if not ep32_enabled:
            return frame

    try:
        import cv2  # type: ignore
    except Exception:
        return frame

    h, w = frame.shape[:2]
    x0 = w - badge_w - 10

    if y_top is not None:
        y0 = int(y_top)
    elif anchor == "top-right":
        y0 = 10
    elif anchor == "below-ep32":
        # Legacy: directly under EP32 BT indicator at y=140. Overlaps the
        # D-pad panel (y=230..476). Kept for compatibility only.
        y0 = 230
    else:
        # Default "below-dpad": below the D-pad floating panel.
        # draw_ep32_floating_controls uses panel_y=230 and panel_h~=246.
        # Land at 484 (= 230 + 246 + 8 margin) and clamp below.
        y0 = 484

    # Clamp to frame so the badge never goes off-screen on small windows.
    if y0 + badge_h > h - 8:
        y0 = max(8, h - badge_h - 8)
    if x0 < 8:
        x0 = 8

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
    pair_txt = f" pair:{st.pair}" if st.pair else ""
    cv2.putText(
        frame,
        f"E:{st.energy:.2f}  C:{st.confidence:.2f}  mic:{st.mic_count}{pair_txt}",
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
    "acoustic_wiring",
    "AcousticArrayConfig",
    "DEFAULT_WIRING",
    "default_wiring_dict",
]
