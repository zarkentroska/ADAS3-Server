import time

import cv2


def _draw_icon_overlay_button(frame, mouse_pos, click_pos, icon, x1, y1):
    """Dibuja un icono con alpha/hover/click en coordenadas absolutas."""
    if icon is None:
        return frame, False

    h, w = icon.shape[:2]
    x2 = x1 + w
    y2 = y1 + h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return frame, False

    roi = frame[y1:y2, x1:x2]
    icon_resized = icon[: y2 - y1, : x2 - x1]

    if icon_resized.shape[2] == 4:
        alpha = icon_resized[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * icon_resized[:, :, c]
    else:
        roi[:] = icon_resized

    mx, my = mouse_pos
    is_hover = x1 <= mx <= x2 and y1 <= my <= y2
    is_clicked = False
    if click_pos:
        cx_click, cy_click = click_pos
        if x1 <= cx_click <= x2 and y1 <= cy_click <= y2:
            is_clicked = True

    if is_hover:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    return frame, is_clicked


def draw_interactive_button(
    frame,
    text,
    x_start,
    y_center,
    w,
    h,
    text_color,
    mouse_pos,
    click_pos,
    align_right=False,
):
    """Dibuja un botón redondeado transparente con hover y detección de clic."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    padding_x = 20
    padding_y = 12

    width = text_size[0] + padding_x
    height = text_size[1] + padding_y
    if w > 0:
        width = max(width, w)
    if h > 0:
        height = max(height, h)

    y1 = int(y_center - text_size[1] - padding_y / 2)
    y2 = y1 + int(height)

    if align_right:
        x2 = int(x_start)
        x1 = x2 - int(width)
        text_x = x1 + padding_x // 2
    else:
        x1 = int(x_start)
        x2 = x1 + int(width)
        text_x = x1 + padding_x // 2

    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2) and (y1 <= my <= y2)

    is_clicked = False
    if click_pos:
        cx, cy = click_pos
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            is_clicked = True

    overlay = frame.copy()
    bg_color = (0, 0, 0)
    alpha = 0.6 if is_hover else 0.4
    radius = 10

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), bg_color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), bg_color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, bg_color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, bg_color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, bg_color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, bg_color, -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_y = y2 - padding_y // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    return frame, is_clicked


def draw_yolo_indicator(frame, mouse_pos, click_pos, yolo_enabled, detecciones, t_func):
    x = frame.shape[1] - 40
    y = 20
    if yolo_enabled:
        color = (0, 255, 0)
        text = t_func("yolo_on", detecciones)
    else:
        color = (0, 0, 255)
        text = t_func("yolo_off")
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def draw_tinysa_indicator(frame, mouse_pos, click_pos, tinysa_running, t_func):
    x = frame.shape[1] - 40
    y = 50
    if tinysa_running:
        color = (0, 255, 0)
        text = t_func("tinysa_on")
    else:
        color = (0, 0, 255)
        text = t_func("tinysa_off")
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def draw_audio_detection_toggle(frame, mouse_pos, click_pos, audio_detection_enabled, t_func):
    x_text = frame.shape[1] - 40
    y = 80
    if audio_detection_enabled:
        color = (0, 255, 0)
        text = t_func("det_audio_on")
    else:
        color = (0, 0, 255)
        text = t_func("det_audio_off")
    return draw_interactive_button(frame, text, x_text, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def draw_tailscale_indicator(frame, mouse_pos, click_pos, tailscale_running, t_func):
    x = frame.shape[1] - 40
    y = 110
    if tailscale_running:
        color = (0, 255, 0)
        text = t_func("tailscale_on")
    else:
        color = (0, 0, 255)
        text = t_func("tailscale_off")
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def draw_telegram_indicator(frame, mouse_pos, click_pos, t_func):
    x = frame.shape[1] - 40
    y = 140
    text = t_func("telegram_notif_app")
    color = (255, 255, 255)
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def draw_language_indicator(frame, mouse_pos, click_pos, t_func):
    x = frame.shape[1] - 40
    y = 170
    text = t_func("language_app")
    color = (255, 255, 255)
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def draw_tailscale_settings_icon(frame, mouse_pos, click_pos, icon):
    """Dibuja el icono de ajustes para Tailscale."""
    if icon is None:
        return frame, False
    h, w = icon.shape[:2]
    x2 = frame.shape[1] - 10
    x1 = x2 - w
    y1 = 110 - h // 2 - 5
    return _draw_icon_overlay_button(frame, mouse_pos, click_pos, icon, x1, y1)


def draw_yolo_settings_icon(frame, mouse_pos, click_pos, icon):
    if icon is None:
        return frame, False
    h, w = icon.shape[:2]
    x2 = frame.shape[1] - 10
    x1 = x2 - w
    y1 = 15 - h // 2
    return _draw_icon_overlay_button(frame, mouse_pos, click_pos, icon, x1, y1)


def draw_tinysa_settings_icon(frame, mouse_pos, click_pos, icon):
    if icon is None:
        return frame, False
    h, w = icon.shape[:2]
    x2 = frame.shape[1] - 10
    x1 = x2 - w
    y1 = 45 - h // 2
    return _draw_icon_overlay_button(frame, mouse_pos, click_pos, icon, x1, y1)


def draw_ip_settings_icon(frame, mouse_pos, click_pos, icon, ip_text):
    if icon is None:
        return frame, False
    h, w = icon.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(ip_text, font, 0.5, 2)
    x2 = 10 + text_size[0] + 20 + w
    x1 = x2 - w
    y_center = 15
    y1 = y_center - h // 2
    return _draw_icon_overlay_button(frame, mouse_pos, click_pos, icon, x1, y1)


def draw_audio_volume_icon(frame, mouse_pos, click_pos, icon):
    if icon is None:
        return frame, False
    h, _ = icon.shape[:2]
    x_text = frame.shape[1] - 40
    y_text = 80
    icon_x = x_text - 175
    icon_y = y_text - h // 2 - 6
    return _draw_icon_overlay_button(frame, mouse_pos, click_pos, icon, icon_x, icon_y)


def draw_ip_indicator(frame, ip_y_puerto, t_func):
    x = 10
    y = 20
    text = t_func("ip_label", ip_y_puerto)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    padding_x = 14
    padding_y = 10
    x1 = x - 6
    y1 = y - text_size[1] - padding_y // 2
    x2 = x1 + text_size[0] + padding_x
    y2 = y + padding_y // 2

    x1 = max(0, x1)
    y1 = max(0, y1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return frame, False


def draw_audio_detection_indicator(
    frame,
    audio_detection_enabled,
    audio_detection_lock,
    audio_detection_result,
    audio_detection_alert_time,
    audio_detection_max_confidence,
    audio_visual_multiplier,
    t_func,
):
    if not audio_detection_enabled:
        return frame

    with audio_detection_lock:
        is_drone = audio_detection_result["is_drone"]
        confidence = audio_detection_result["confidence"]

    y = frame.shape[0] - 30

    if is_drone:
        blink = int(time.time() * 2) % 2 == 0
        color = (0, 255, 255) if blink else (0, 128, 128)
        if audio_detection_alert_time is not None:
            alert_time_str = time.strftime("%H:%M:%S", time.localtime(audio_detection_alert_time))
            max_visual = min(100, int(audio_detection_max_confidence * audio_visual_multiplier * 100))
            text = f"AUDIO DRON DETECTADO A LAS {alert_time_str} - {max_visual}%"
        else:
            text = t_func("audio_drone_detected", int(confidence * 100))
    else:
        color = (0, 0, 255)
        text = t_func("no_audio_dron", int(confidence * 100))

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    x = frame.shape[1] - text_size[0] - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 23), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
