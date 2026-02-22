"""
OpenCV slider controls for YOLO and RF detection parameter tuning.

Draws semi-transparent slider panels on the video frame and handles
mouse interaction (click/drag) to adjust values in real time.
"""

import cv2


def draw_slider_control(frame, label, value, min_val, max_val,
                         origin, size, mouse_pos, click_pos,
                         slider_key, mouse_is_down,
                         active_slider, set_active_slider_fn):
    """Draws a slider and returns (frame, new_value_or_None)."""
    x, y = origin
    width, height = size
    overlay = frame.copy()

    panel_y1 = y
    panel_y2 = y + height
    cv2.rectangle(overlay, (x, panel_y1), (x + width, panel_y2), (20, 20, 20), -1)

    text_y = panel_y1 + 18
    cv2.putText(overlay, label, (x + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)
    cv2.putText(overlay, f"{value:.2f}", (x + width - 55, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)

    slider_offset = 10
    panel_y1 += slider_offset

    track_x1 = x + 20
    track_x2 = x + width - 20
    track_y = panel_y1 + height - 25
    cv2.line(overlay, (track_x1, track_y), (track_x2, track_y), (210, 210, 210), 6, cv2.LINE_AA)

    ratio = (value - min_val) / (max_val - min_val)
    ratio = max(0.0, min(1.0, ratio))
    handle_x = int(track_x1 + ratio * (track_x2 - track_x1))
    cv2.circle(overlay, (handle_x, track_y), 10, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(overlay, (handle_x, track_y), 10, (0, 102, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    new_value = None

    if click_pos:
        cx, cy = click_pos
        if track_y - 18 <= cy <= track_y + 18 and track_x1 <= cx <= track_x2:
            ratio = (cx - track_x1) / (track_x2 - track_x1)
            ratio = max(0.0, min(1.0, ratio))
            new_value = min_val + ratio * (max_val - min_val)
            set_active_slider_fn(slider_key)
    elif active_slider == slider_key and mouse_is_down:
        mx, my = mouse_pos
        ratio = (mx - track_x1) / (track_x2 - track_x1)
        ratio = max(0.0, min(1.0, ratio))
        new_value = min_val + ratio * (max_val - min_val)
    elif active_slider == slider_key and not mouse_is_down:
        set_active_slider_fn(None)

    return frame, new_value


def draw_yolo_sliders(frame, mouse_pos, click_pos, *,
                       yolo_enabled, yolo_conf_threshold, yolo_iou_threshold,
                       yolo_threshold_lock, mouse_is_down,
                       yolo_slider_active, set_yolo_slider_active_fn):
    """Draws YOLO threshold sliders. Returns (frame, remaining_click, new_conf, new_iou)."""
    if not yolo_enabled:
        return frame, click_pos, yolo_conf_threshold, yolo_iou_threshold

    slider_width = int(frame.shape[1] * 0.16)
    slider_height = 50
    x = 50
    y_start = 105
    spacing = 6

    specs = [
        ("Confidence threshold", yolo_conf_threshold, 0.05, 0.99, "conf"),
        ("IoU threshold", yolo_iou_threshold, 0.05, 0.99, "iou"),
    ]

    remaining_click = click_pos
    new_conf = yolo_conf_threshold
    new_iou = yolo_iou_threshold

    for idx, (label, value, v_min, v_max, key) in enumerate(specs):
        y = y_start + idx * (slider_height + spacing)
        frame, new_val = draw_slider_control(
            frame, label, value, v_min, v_max,
            (x, y), (slider_width, slider_height),
            mouse_pos, remaining_click, key,
            mouse_is_down, yolo_slider_active, set_yolo_slider_active_fn,
        )

        if new_val is not None:
            with yolo_threshold_lock:
                if key == "conf":
                    new_conf = new_val
                else:
                    new_iou = new_val
            remaining_click = None

    return frame, remaining_click, new_conf, new_iou


def draw_rf_drone_sliders(frame, mouse_pos, click_pos, *,
                           rf_sliders_visible, tinysa_running,
                           rf_peak_threshold, rf_min_peak_height_db,
                           rf_min_peak_width_mhz, rf_max_peak_width_mhz,
                           rf_detection_params_lock, mouse_is_down,
                           rf_slider_active, set_rf_slider_active_fn):
    """Draws RF detection sliders. Returns (frame, remaining_click, params_dict)."""
    params = {
        "rf_peak_threshold": rf_peak_threshold,
        "rf_min_peak_height_db": rf_min_peak_height_db,
        "rf_min_peak_width_mhz": rf_min_peak_width_mhz,
        "rf_max_peak_width_mhz": rf_max_peak_width_mhz,
    }

    if not rf_sliders_visible or not tinysa_running:
        return frame, click_pos, params

    slider_width = int(frame.shape[1] * 0.20)
    slider_height = 50
    x = 50
    y_start = 105
    spacing = 6

    specs = [
        ("Umbral Potencia (dBm)", rf_peak_threshold, -100.0, -50.0, "rf_peak_thresh"),
        ("Altura Min Ruido (dB)", rf_min_peak_height_db, 1.0, 40.0, "rf_min_height"),
        ("Ancho Min (MHz)", rf_min_peak_width_mhz, 1.0, 30.0, "rf_min_width"),
        ("Ancho Max (MHz)", rf_max_peak_width_mhz, 20.0, 80.0, "rf_max_width"),
    ]

    remaining_click = click_pos
    key_to_param = {
        "rf_peak_thresh": "rf_peak_threshold",
        "rf_min_height": "rf_min_peak_height_db",
        "rf_min_width": "rf_min_peak_width_mhz",
        "rf_max_width": "rf_max_peak_width_mhz",
    }

    for idx, (label, value, v_min, v_max, key) in enumerate(specs):
        y = y_start + idx * (slider_height + spacing)
        frame, new_val = draw_slider_control(
            frame, label, value, v_min, v_max,
            (x, y), (slider_width, slider_height),
            mouse_pos, remaining_click, key,
            mouse_is_down, rf_slider_active, set_rf_slider_active_fn,
        )

        if new_val is not None:
            param_name = key_to_param[key]
            with rf_detection_params_lock:
                params[param_name] = new_val
            print(f"[RF SLIDER] {label}: {new_val:.2f}")
            remaining_click = None

    return frame, remaining_click, params
