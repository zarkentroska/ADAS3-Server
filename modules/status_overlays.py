import cv2


def draw_adb_message(frame, t_func, adb_connected):
    if not adb_connected:
        return frame
    text = t_func("adb_connected")
    cv2.putText(frame, text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def draw_tinysa_message(
    frame,
    t_func,
    tinysa_detected,
    tinysa_use_http,
    rf_drone_detection_lock,
    rf_drone_detection_result,
    rf_drone_detection_enabled,
):
    if not tinysa_detected:
        return frame

    text = t_func("tinysa_connected_android") if tinysa_use_http else t_func("tinysa_connected")
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 10
    y = frame.shape[0] - 15
    cv2.putText(frame, text, (x, y), font, 0.55, (0, 255, 255), 2)

    with rf_drone_detection_lock:
        rf_result = rf_drone_detection_result.copy()

    if rf_result.get("is_drone", False) and rf_drone_detection_enabled:
        confidence = rf_result.get("confidence", 0.0)
        frequency = rf_result.get("frequency")

        if frequency:
            freq_mhz = frequency / 1e6
            alert_text = t_func("rf_drone_detected", freq_mhz, int(confidence * 100))
        else:
            alert_text = t_func("rf_drone_detected_no_freq", int(confidence * 100))

        text_size, _ = cv2.getTextSize(alert_text, font, 0.7, 2)
        text_w, text_h = text_size
        alert_x = 10
        alert_y = y - text_h - 25

        cv2.rectangle(
            frame,
            (alert_x - 5, alert_y - 5),
            (alert_x + text_w + 5, alert_y + text_h + 5),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            frame,
            alert_text,
            (alert_x, alert_y + text_h),
            font,
            0.7,
            (255, 255, 255),
            2,
        )

    return frame


def draw_fps_indicator(frame, fps, t_func):
    x = 10
    y = 50
    text = t_func("fps_label", fps)

    overlay = frame.copy()
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(overlay, (x - 5, y - 18), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame
