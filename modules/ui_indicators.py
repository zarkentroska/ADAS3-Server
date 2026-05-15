import time

import cv2
import numpy as np

_hovering_interactive = False


def begin_hover_tracking():
    global _hovering_interactive
    _hovering_interactive = False


def is_hovering_interactive():
    return _hovering_interactive


def _mark_hover(is_hover):
    global _hovering_interactive
    if is_hover:
        _hovering_interactive = True


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
    _mark_hover(is_hover)
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
    _mark_hover(is_hover)

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


def draw_yolo_indicator(frame, mouse_pos, click_pos, yolo_enabled, detecciones, power_level, t_func):
    """YOLO + selector DET.POWER (L/M/H/VH). power_level: 'low'|'medium'|'high'|'very_high'."""
    y = 20
    margin_right = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    padding_x, padding_y = 20, 12
    radius = 10

    if yolo_enabled:
        yolo_color = (0, 255, 0)
        yolo_text = t_func("yolo_on", detecciones)
    else:
        yolo_color = (0, 0, 255)
        yolo_text = t_func("yolo_off")

    (tw, th), _ = cv2.getTextSize(yolo_text, font, font_scale, thickness)
    yolo_w = tw + padding_x
    yolo_h = th + padding_y
    yolo_right = frame.shape[1] - margin_right
    yolo_x1 = int(yolo_right - yolo_w)
    yolo_y1 = int(y - th - padding_y / 2)
    yolo_x2, yolo_y2 = int(yolo_right), int(yolo_y1 + yolo_h)

    seg_specs = [
        ("low", t_func("det_power_short_low")),
        ("medium", t_func("det_power_short_med")),
        ("high", t_func("det_power_short_high")),
        ("very_high", t_func("det_power_short_vhigh")),
    ]
    seg_font_scale = 0.4
    seg_pad_x = 8
    seg_gap = 3
    seg_w = {}
    for key, st in seg_specs:
        (sw, _), _ = cv2.getTextSize(st, font, seg_font_scale, 2)
        seg_w[key] = sw + seg_pad_x * 2

    lbl = t_func("det_power_label")
    lbl_scale = 0.42
    (lbl_tw, lbl_th), _ = cv2.getTextSize(lbl, font, lbl_scale, 1)
    lbl_gap = 8

    power_sel = None
    cursor_r = yolo_x1 - 10

    for key, seg_txt in reversed(seg_specs):
        wseg = seg_w[key]
        x2 = cursor_r
        x1 = x2 - wseg
        y1, y2 = int(yolo_y1), int(yolo_y2)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)
        if x2 > x1 and y2 > y1:
            active = power_level == key
            mx, my = mouse_pos
            is_hover = (x1 <= mx <= x2) and (y1 <= my <= y2)
            _mark_hover(is_hover)
            if click_pos:
                cx, cy = click_pos
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    power_sel = key
            overlay = frame.copy()
            bg = (20, 90, 40) if active else (0, 0, 0)
            alpha = 0.72 if active else (0.62 if is_hover else 0.45)
            r = min(radius, (y2 - y1) // 3, (x2 - x1) // 3)
            cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), bg, -1)
            cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), bg, -1)
            cv2.circle(overlay, (x1 + r, y1 + r), r, bg, -1)
            cv2.circle(overlay, (x2 - r, y1 + r), r, bg, -1)
            cv2.circle(overlay, (x1 + r, y2 - r), r, bg, -1)
            cv2.circle(overlay, (x2 - r, y2 - r), r, bg, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            border = (0, 255, 120) if active else (180, 180, 180)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
            (twg, thg), _ = cv2.getTextSize(seg_txt, font, seg_font_scale, 2)
            tx = x1 + (wseg - twg) // 2
            ty = y2 - (y2 - y1 - thg) // 2 - 2
            cv2.putText(frame, seg_txt, (tx, ty), font, seg_font_scale, (255, 255, 255), 2)
        cursor_r = x1 - seg_gap

    lbl_x2 = cursor_r - lbl_gap
    lbl_x1 = lbl_x2 - lbl_tw
    lbl_y = yolo_y2 - 4
    if lbl_x1 >= 0:
        mx, my = mouse_pos
        if lbl_x1 <= mx <= lbl_x2 and yolo_y1 <= my <= yolo_y2:
            _mark_hover(True)
        cv2.putText(frame, lbl, (lbl_x1, lbl_y), font, lbl_scale, (230, 230, 230), 1)

    mx, my = mouse_pos
    yolo_hover = (yolo_x1 <= mx <= yolo_x2) and (yolo_y1 <= my <= yolo_y2)
    _mark_hover(yolo_hover)
    yolo_clicked = False
    if click_pos:
        cx, cy = click_pos
        if yolo_x1 <= cx <= yolo_x2 and yolo_y1 <= cy <= yolo_y2:
            yolo_clicked = True

    overlay = frame.copy()
    bg_color = (0, 0, 0)
    alpha = 0.6 if yolo_hover else 0.4
    r = radius
    cv2.rectangle(overlay, (yolo_x1 + r, yolo_y1), (yolo_x2 - r, yolo_y2), bg_color, -1)
    cv2.rectangle(overlay, (yolo_x1, yolo_y1 + r), (yolo_x2, yolo_y2 - r), bg_color, -1)
    cv2.circle(overlay, (yolo_x1 + r, yolo_y1 + r), r, bg_color, -1)
    cv2.circle(overlay, (yolo_x2 - r, yolo_y1 + r), r, bg_color, -1)
    cv2.circle(overlay, (yolo_x1 + r, yolo_y2 - r), r, bg_color, -1)
    cv2.circle(overlay, (yolo_x2 - r, yolo_y2 - r), r, bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    text_y = yolo_y2 - padding_y // 2
    text_x = yolo_x1 + padding_x // 2
    cv2.putText(frame, yolo_text, (text_x, text_y), font, font_scale, yolo_color, 2)
    if yolo_hover:
        cv2.rectangle(frame, (yolo_x1, yolo_y1), (yolo_x2, yolo_y2), (255, 255, 255), 1, cv2.LINE_AA)

    return frame, yolo_clicked, power_sel


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


def draw_ep32_indicator(frame, mouse_pos, click_pos, ep32_enabled, ep32_status_text, t_func):
    x = frame.shape[1] - 40
    y = 140
    if ep32_enabled:
        color = (0, 255, 0)
        text = t_func("ep32_bt_on")
        if ep32_status_text:
            text = f"{text} ({ep32_status_text})"
    else:
        color = (0, 0, 255)
        text = t_func("ep32_bt_off")
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def draw_telegram_indicator(frame, mouse_pos, click_pos, t_func):
    x = frame.shape[1] - 40
    y = 170
    text = t_func("telegram_notif_app")
    color = (255, 255, 255)
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def draw_language_indicator(frame, mouse_pos, click_pos, t_func):
    x = frame.shape[1] - 40
    y = 200
    text = t_func("language_app")
    color = (255, 255, 255)
    return draw_interactive_button(frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=True)


def draw_audio_source_indicator(frame, mouse_pos, click_pos, audio_source_id,
                                 audio_source_status_text, t_func):
    """Indicador/selector del origen del audio para Keras.

    Click → ciclar (phone_mic → esp32_array → phone_mic …). Se coloca
    en el lado izquierdo bajo el mensaje ADB para no chocar con la
    columna derecha (EP32, Telegram, idioma, ...).
    """
    x = 10
    y = 110
    if audio_source_id == "esp32_array":
        color = (0, 200, 200)
        text = t_func("audio_source_esp32_array")
    else:
        color = (200, 200, 0)
        text = t_func("audio_source_phone_mic")
    if audio_source_status_text:
        text = f"{text}: {audio_source_status_text}"
    return draw_interactive_button(
        frame, text, x, y, 0, 0, color, mouse_pos, click_pos, align_right=False,
    )


def _draw_dpad_button(frame, mouse_pos, click_pos, x1, y1, x2, y2, label, action_id):
    """Dibuja un botón individual del D-pad y devuelve (frame, action_id|None)."""
    mx, my = mouse_pos
    is_hover = x1 <= mx <= x2 and y1 <= my <= y2
    _mark_hover(is_hover)
    clicked_action = None
    if click_pos:
        cx, cy = click_pos
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            clicked_action = action_id

    btn_overlay = frame.copy()
    cv2.rectangle(btn_overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(btn_overlay, 0.60 if is_hover else 0.40, frame, 0.40 if is_hover else 0.60, 0, frame)
    border_color = (0, 200, 255) if is_hover else (130, 130, 130)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return frame, clicked_action


def draw_ep32_floating_controls(frame, mouse_pos, click_pos, ep32_enabled, ep32_status_text, t_func, auto_tracking=False):
    """Dibuja panel flotante EP32 con D-pad y botón de auto-tracking.

    Devuelve (frame, dpad_action, autotrack_clicked).
    """
    if not ep32_enabled:
        return frame, None, False

    btn_s = 48
    gap = 6
    dpad_w = btn_s * 3 + gap * 2
    header_h = 40
    autotrack_h = 30
    panel_w = dpad_w + 20
    panel_h = header_h + btn_s * 3 + gap * 2 + gap + autotrack_h + 14
    panel_x = max(8, frame.shape[1] - panel_w - 18)
    panel_y = 230
    if panel_y + panel_h > frame.shape[0] - 8:
        panel_y = max(8, frame.shape[0] - panel_h - 8)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 1, cv2.LINE_AA)

    title = t_func("ep32_controls_title")
    status_prefix = t_func("ep32_status_label")
    status_text = ep32_status_text if ep32_status_text else t_func("ep32_status_unknown")
    cv2.putText(frame, title, (panel_x + 10, panel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{status_prefix}: {status_text}",
        (panel_x + 10, panel_y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (190, 255, 190),
        1,
        cv2.LINE_AA,
    )

    dpad_x = panel_x + (panel_w - dpad_w) // 2
    dpad_y = panel_y + header_h

    #        [ UP  ]
    # [LEFT] [    ] [RIGHT]
    #        [DOWN ]
    dpad_layout = {
        "up":    (dpad_x + btn_s + gap, dpad_y),
        "left":  (dpad_x, dpad_y + btn_s + gap),
        "right": (dpad_x + 2 * (btn_s + gap), dpad_y + btn_s + gap),
        "down":  (dpad_x + btn_s + gap, dpad_y + 2 * (btn_s + gap)),
    }
    dpad_labels = {
        "up": t_func("ep32_btn_up"),
        "down": t_func("ep32_btn_down"),
        "left": t_func("ep32_btn_left"),
        "right": t_func("ep32_btn_right"),
    }

    selected_action = None
    for action_id, (bx, by) in dpad_layout.items():
        frame, clicked = _draw_dpad_button(
            frame, mouse_pos, click_pos,
            bx, by, bx + btn_s, by + btn_s,
            dpad_labels[action_id], action_id,
        )
        if clicked:
            selected_action = clicked

    # --- Botón AUTO-TRACK debajo del D-pad ---
    at_y1 = dpad_y + btn_s * 3 + gap * 2 + gap
    at_x1 = panel_x + 10
    at_x2 = panel_x + panel_w - 10
    at_y2 = at_y1 + autotrack_h

    at_label = t_func("ep32_autotrack_on") if auto_tracking else t_func("ep32_autotrack_off")
    at_color = (0, 200, 0) if auto_tracking else (0, 0, 200)

    mx, my = mouse_pos
    at_hover = at_x1 <= mx <= at_x2 and at_y1 <= my <= at_y2
    _mark_hover(at_hover)
    at_clicked = False
    if click_pos:
        cx, cy = click_pos
        if at_x1 <= cx <= at_x2 and at_y1 <= cy <= at_y2:
            at_clicked = True

    at_overlay = frame.copy()
    cv2.rectangle(at_overlay, (at_x1, at_y1), (at_x2, at_y2), at_color, -1)
    cv2.addWeighted(at_overlay, 0.55 if at_hover else 0.40, frame, 0.45 if at_hover else 0.60, 0, frame)
    cv2.rectangle(frame, (at_x1, at_y1), (at_x2, at_y2), (255, 255, 255) if at_hover else (160, 160, 160), 1, cv2.LINE_AA)

    at_ts, _ = cv2.getTextSize(at_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    at_tx = at_x1 + (at_x2 - at_x1 - at_ts[0]) // 2
    at_ty = at_y1 + (at_y2 - at_y1 + at_ts[1]) // 2
    cv2.putText(frame, at_label, (at_tx, at_ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    return frame, selected_action, at_clicked


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


def draw_ip_selector_button(frame, mouse_pos, click_pos, icon, ip_text):
    """Dibuja un pequeño selector (flecha abajo) junto al icono de IP."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(ip_text, font, 0.5, 2)
    y_center = 15

    icon_w = icon.shape[1] if icon is not None else 16
    gear_right = 10 + text_size[0] + 20 + icon_w

    btn_w = 18
    btn_h = 16
    x1 = gear_right + 6
    y1 = y_center - btn_h // 2
    x2 = x1 + btn_w
    y2 = y1 + btn_h

    mx, my = mouse_pos
    is_hover = x1 <= mx <= x2 and y1 <= my <= y2
    _mark_hover(is_hover)
    is_clicked = False
    if click_pos:
        cx, cy = click_pos
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            is_clicked = True

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45 if not is_hover else 0.60, frame, 0.55 if not is_hover else 0.40, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255) if is_hover else (140, 140, 140), 1)

    tri = np.array(
        [
            [x1 + btn_w // 2 - 4, y1 + btn_h // 2 - 1],
            [x1 + btn_w // 2 + 4, y1 + btn_h // 2 - 1],
            [x1 + btn_w // 2, y1 + btn_h // 2 + 4],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, tri, (255, 255, 255))

    return frame, is_clicked


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
    drone_size_class="",
    drone_size_confidence=0.0,
):
    if not audio_detection_enabled:
        return frame

    with audio_detection_lock:
        is_drone = audio_detection_result["is_drone"]
        confidence = audio_detection_result["confidence"]

    y = frame.shape[0] - 30

    size_suffix = ""
    if is_drone and drone_size_class and drone_size_class != "inconclusive" and drone_size_confidence > 0.0:
        size_label = t_func(f"drone_size_{drone_size_class}")
        if size_label != f"drone_size_{drone_size_class}":
            size_suffix = f" [{size_label} {int(drone_size_confidence * 100)}%]"

    if is_drone:
        blink = int(time.time() * 2) % 2 == 0
        color = (0, 255, 255) if blink else (0, 128, 128)
        if audio_detection_alert_time is not None:
            alert_time_str = time.strftime("%H:%M:%S", time.localtime(audio_detection_alert_time))
            max_visual = min(100, int(audio_detection_max_confidence * audio_visual_multiplier * 100))
            text = f"AUDIO DRON DETECTADO A LAS {alert_time_str} - {max_visual}%{size_suffix}"
        else:
            text = t_func("audio_drone_detected", int(confidence * 100)) + size_suffix
    else:
        color = (0, 0, 255)
        text = t_func("no_audio_dron", int(confidence * 100))

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    x = (frame.shape[1] - text_size[0]) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 23), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
