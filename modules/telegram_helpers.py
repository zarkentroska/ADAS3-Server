"""
Telegram configuration management and multimedia helpers.

Handles config normalization, persistence, chat ID detection,
and multimedia file preparation for Telegram notifications.
"""

import copy
import json
import os
import time
import wave
import tempfile

import cv2
import requests


TELEGRAM_DEFAULT_CONFIG = {
    "enabled": False,
    "token": "",
    "chat_id": "",
    "cooldowns": {
        "yolo": 30.0,
        "rf": 30.0,
        "audio": 30.0,
    },
    "send_yolo_photo": True,
    "send_rf_image": True,
    "send_audio_clip": True,
}


def normalize_telegram_config(raw_config):
    merged = copy.deepcopy(TELEGRAM_DEFAULT_CONFIG)
    if not isinstance(raw_config, dict):
        return merged

    merged["enabled"] = bool(raw_config.get("enabled", merged["enabled"]))
    merged["token"] = str(raw_config.get("token", merged["token"])).strip()
    merged["chat_id"] = str(raw_config.get("chat_id", merged["chat_id"])).strip()
    merged["send_yolo_photo"] = bool(raw_config.get("send_yolo_photo", merged["send_yolo_photo"]))
    merged["send_rf_image"] = bool(raw_config.get("send_rf_image", merged["send_rf_image"]))
    merged["send_audio_clip"] = bool(raw_config.get("send_audio_clip", merged["send_audio_clip"]))

    cooldowns_raw = raw_config.get("cooldowns", {})
    if isinstance(cooldowns_raw, dict):
        for key, default_value in merged["cooldowns"].items():
            value = cooldowns_raw.get(key, default_value)
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = default_value
            if parsed < 0:
                parsed = 0.0
            merged["cooldowns"][key] = parsed

    return merged


def load_telegram_config_from_file(config_file, t_func):
    loaded = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception as exc:
            print(f"{t_func('telegram_config_load_error')} ({exc})")
    return normalize_telegram_config(loaded)


def save_telegram_config_to_file(config, config_file, t_func):
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"{t_func('telegram_config_save_error')} ({exc})")


def detect_telegram_chat_id(token, t_func):
    token = str(token or "").strip()
    if not token:
        return None, t_func("telegram_chat_id_missing_token")

    try:
        response = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok", False):
            return None, payload.get("description", t_func("telegram_chat_id_detect_error"))

        results = payload.get("result", [])
        for update in reversed(results):
            message = update.get("message") or update.get("edited_message") or update.get("channel_post")
            if not isinstance(message, dict):
                continue
            chat = message.get("chat", {})
            chat_id = chat.get("id")
            if chat_id is not None:
                return str(chat_id), None

        return None, t_func("telegram_chat_id_not_found")
    except Exception as exc:
        return None, f"{t_func('telegram_chat_id_detect_error')}: {exc}"


def get_telegram_temp_dir():
    temp_dir = os.path.join(tempfile.gettempdir(), "adas3_telegram")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def build_telegram_message(event_type, *, timestamp, confidence=None, frequency_hz=None, t_func):
    hour_text = time.strftime("%H:%M:%S", time.localtime(timestamp))

    if event_type == "yolo":
        if confidence is None:
            return t_func("telegram_yolo_message", hour_text)
        return t_func("telegram_yolo_message_conf", hour_text, int(confidence * 100))

    if event_type == "rf":
        if frequency_hz:
            return t_func("telegram_rf_message_freq", hour_text, frequency_hz / 1e6, int((confidence or 0.0) * 100))
        return t_func("telegram_rf_message", hour_text, int((confidence or 0.0) * 100))

    return t_func("telegram_audio_message", hour_text, int((confidence or 0.0) * 100))


def save_frame_for_telegram(frame, event_type):
    if frame is None:
        return None
    temp_dir = get_telegram_temp_dir()
    timestamp_ms = int(time.time() * 1000)
    output_path = os.path.join(temp_dir, f"{event_type}_{timestamp_ms}.png")
    success = cv2.imwrite(output_path, frame)
    if not success:
        return None
    return output_path


def save_rf_image_for_telegram(tinysa_render_lock, tinysa_image_ready):
    with tinysa_render_lock:
        rf_image = None if tinysa_image_ready is None else tinysa_image_ready.copy()

    if rf_image is None:
        return None

    try:
        bgr_image = cv2.cvtColor(rf_image, cv2.COLOR_RGBA2BGR)
    except Exception:
        return None

    return save_frame_for_telegram(bgr_image, "rf")


def save_audio_clip_for_telegram(audio_recent_lock, audio_recent_chunks,
                                  audio_stream_sample_rate, audio_stream_channels,
                                  clip_seconds=5):
    with audio_recent_lock:
        if not audio_recent_chunks:
            return None
        raw_audio = b"".join(audio_recent_chunks)
        sample_rate = int(max(audio_stream_sample_rate, 8000))
        channels = int(max(audio_stream_channels, 1))

    max_bytes = int(sample_rate * channels * 2 * max(clip_seconds, 1))
    if len(raw_audio) > max_bytes:
        raw_audio = raw_audio[-max_bytes:]

    if not raw_audio:
        return None

    temp_dir = get_telegram_temp_dir()
    timestamp_ms = int(time.time() * 1000)
    output_path = os.path.join(temp_dir, f"audio_{timestamp_ms}.wav")
    try:
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio)
    except Exception as exc:
        print(f"[TELEGRAM] No se pudo crear clip WAV: {exc}")
        return None

    return output_path
