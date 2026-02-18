import os

import cv2
import numpy as np


def get_yolo_settings_icon(settings_icon_path, current_icon):
    """Carga y devuelve el icono de ajustes, conservando caché."""
    if current_icon is not None:
        return current_icon

    if os.path.exists(settings_icon_path):
        icon = cv2.imread(settings_icon_path, cv2.IMREAD_UNCHANGED)
        if icon is not None:
            desired_size = 26
            icon = cv2.resize(icon, (desired_size, desired_size), interpolation=cv2.INTER_AREA)
            return icon

    size = 26
    fallback = np.zeros((size, size, 4), dtype=np.uint8)
    cv2.circle(fallback, (size // 2, size // 2), size // 2 - 2, (90, 90, 90, 255), -1, cv2.LINE_AA)
    return fallback


def get_audio_volume_icon(muted, mute_icon, vol_icon, mute_icon_path, vol_icon_path):
    """Carga y devuelve icono de volumen (mute/vol), conservando caché."""
    if muted and mute_icon is not None:
        return mute_icon, vol_icon, mute_icon
    if (not muted) and vol_icon is not None:
        return mute_icon, vol_icon, vol_icon

    icon_path = mute_icon_path if muted else vol_icon_path
    if os.path.exists(icon_path):
        icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
        if icon is not None:
            desired_size = 24
            icon = cv2.resize(icon, (desired_size, desired_size), interpolation=cv2.INTER_AREA)
            if muted:
                mute_icon = icon
            else:
                vol_icon = icon
            return mute_icon, vol_icon, icon

    size = 24
    fallback = np.zeros((size, size, 4), dtype=np.uint8)
    cv2.circle(fallback, (size // 2, size // 2), size // 3, (200, 200, 200, 255), 2)
    if muted:
        cv2.line(
            fallback,
            (size // 4, size // 4),
            (3 * size // 4, 3 * size // 4),
            (200, 200, 200, 255),
            2,
        )
        mute_icon = fallback
    else:
        vol_icon = fallback

    return mute_icon, vol_icon, fallback
