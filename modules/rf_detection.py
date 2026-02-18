import time

import numpy as np


def detect_drone_rf(
    freqs,
    levels,
    rf_history,
    peak_threshold,
    min_peak_height_db,
    min_peak_width_mhz,
    max_peak_width_mhz,
):
    """
    Detecta señales de drones en el espectro RF y retorna (resultado, historial_actualizado).
    """
    if len(freqs) == 0 or len(levels) == 0 or len(freqs) != len(levels):
        return {"is_drone": False, "confidence": 0.0, "frequency": None}, rf_history

    freqs = np.array(freqs)
    levels = np.array(levels)

    drone_bands = [
        (2400000000, 2500000000),
        (5725000000, 5875000000),
    ]

    valid_mask = np.isfinite(levels) & (levels > -150) & (levels < 0)
    if np.sum(valid_mask) < 10:
        return {"is_drone": False, "confidence": 0.0, "frequency": None}, rf_history

    freqs_valid = freqs[valid_mask]
    levels_valid = levels[valid_mask]

    noise_level = np.percentile(levels_valid, 10)
    peak_threshold_relative = noise_level + min_peak_height_db

    min_distance = max(1, len(levels_valid) // 50)
    peaks = []

    for i in range(min_distance, len(levels_valid) - min_distance):
        if levels_valid[i] < peak_threshold_relative:
            continue

        is_peak = True
        for j in range(max(0, i - min_distance), min(len(levels_valid), i + min_distance + 1)):
            if j != i and levels_valid[j] >= levels_valid[i]:
                is_peak = False
                break

        if is_peak:
            peaks.append(i)

    if len(peaks) == 0:
        return {"is_drone": False, "confidence": 0.0, "frequency": None}, rf_history

    peaks = np.array(peaks)

    best_confidence = 0.0
    best_frequency = None

    for peak_idx in peaks:
        peak_freq = freqs_valid[peak_idx]
        peak_level = levels_valid[peak_idx]

        in_drone_band = False
        for band_start, band_stop in drone_bands:
            if band_start <= peak_freq <= band_stop:
                in_drone_band = True
                break

        if not in_drone_band:
            continue

        half_max = peak_level - (peak_level - noise_level) / 2
        left_idx = peak_idx
        right_idx = peak_idx

        while left_idx > 0 and levels_valid[left_idx] > half_max:
            left_idx -= 1
        while right_idx < len(levels_valid) - 1 and levels_valid[right_idx] > half_max:
            right_idx += 1

        if left_idx < right_idx:
            bandwidth_hz = freqs_valid[right_idx] - freqs_valid[left_idx]
            bandwidth_mhz = bandwidth_hz / 1e6
        else:
            bandwidth_mhz = 0

        if peak_level > peak_threshold and min_peak_width_mhz <= bandwidth_mhz <= max_peak_width_mhz:
            height_above_noise = peak_level - noise_level
            height_confidence = min(1.0, height_above_noise / 40.0)

            optimal_bw = 22.5
            bw_diff = abs(bandwidth_mhz - optimal_bw)
            bw_confidence = max(0.0, 1.0 - (bw_diff / 20.0))

            power_confidence = min(1.0, (peak_level - peak_threshold) / 30.0)
            confidence = (height_confidence * 0.4 + bw_confidence * 0.3 + power_confidence * 0.3)

            if confidence > best_confidence:
                best_confidence = confidence
                best_frequency = peak_freq

    current_time = time.time()
    rf_history = [(t, freq, conf) for t, freq, conf in rf_history if current_time - t < 2.0]

    if best_confidence > 0.5:
        rf_history.append((current_time, best_frequency, best_confidence))
        if len(rf_history) >= 2:
            avg_confidence = np.mean([conf for _, _, conf in rf_history])
            avg_frequency = np.mean([freq for _, freq, _ in rf_history])
            return {
                "is_drone": True,
                "confidence": min(1.0, avg_confidence),
                "frequency": avg_frequency,
            }, rf_history

    return {"is_drone": False, "confidence": 0.0, "frequency": None}, rf_history
