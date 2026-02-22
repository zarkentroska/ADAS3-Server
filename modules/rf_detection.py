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
        # Rango ampliado para 5 GHz (según país/canal DJI puede usar ~5.0-5.9 GHz)
        (5000000000, 5895000000),
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

    peaks_5g = []

    for peak_idx in peaks:
        peak_freq = freqs_valid[peak_idx]
        peak_level = levels_valid[peak_idx]

        in_drone_band = False
        is_5g_band = False
        for band_start, band_stop in drone_bands:
            if band_start <= peak_freq <= band_stop:
                in_drone_band = True
                is_5g_band = band_start >= 5e9
                break

        if not in_drone_band:
            continue

        if is_5g_band and peak_level > (peak_threshold - 10):
            peaks_5g.append((peak_freq, peak_level))

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

        if peak_level > peak_threshold:
            height_above_noise = peak_level - noise_level
            height_confidence = min(1.0, height_above_noise / 40.0)

            # 2.4 GHz suele verse como meseta ancha; 5 GHz puede verse como
            # varios picos estrechos consecutivos, así que ajustamos por banda.
            if is_5g_band:
                optimal_bw = 8.0
                bw_diff = abs(bandwidth_mhz - optimal_bw)
                bw_confidence = max(0.0, 1.0 - (bw_diff / 18.0))
            else:
                optimal_bw = 22.5
                bw_diff = abs(bandwidth_mhz - optimal_bw)
                bw_confidence = max(0.0, 1.0 - (bw_diff / 20.0))

            # Suavizar el criterio de ancho:
            # - si es algo más estrecho/ancho de lo esperado, penalizamos confianza
            #   en lugar de descartarlo completamente.
            width_penalty = 1.0
            min_width_ref = 2.0 if is_5g_band else min_peak_width_mhz
            max_width_ref = max_peak_width_mhz if not is_5g_band else max(max_peak_width_mhz, 35.0)
            if bandwidth_mhz < min_width_ref:
                width_penalty = max(0.35 if is_5g_band else 0.2, bandwidth_mhz / max(min_width_ref, 0.1))
            elif bandwidth_mhz > max_width_ref:
                width_penalty = max(0.2, max_width_ref / max(bandwidth_mhz, 0.1))

            power_confidence = min(1.0, (peak_level - peak_threshold) / 30.0)
            confidence = (height_confidence * 0.4 + bw_confidence * 0.3 + power_confidence * 0.3) * width_penalty

            if confidence > best_confidence:
                best_confidence = confidence
                best_frequency = peak_freq

    # Heurística adicional para 5 GHz:
    # cuando la señal aparece como "tren" de picos (sube/baja), no como meseta.
    if len(peaks_5g) >= 3:
        peaks_5g.sort(key=lambda x: x[0])
        window_hz = 45e6
        best_cluster = []
        best_cluster_mean = -1e9

        for idx in range(len(peaks_5g)):
            start_freq = peaks_5g[idx][0]
            cluster = []
            for freq, level in peaks_5g[idx:]:
                if freq - start_freq <= window_hz:
                    cluster.append((freq, level))
                else:
                    break
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
                best_cluster_mean = float(np.mean([lvl for _, lvl in cluster]))
            elif len(cluster) == len(best_cluster) and cluster:
                mean_lvl = float(np.mean([lvl for _, lvl in cluster]))
                if mean_lvl > best_cluster_mean:
                    best_cluster = cluster
                    best_cluster_mean = mean_lvl

        if len(best_cluster) >= 3:
            cluster_count = len(best_cluster)
            cluster_conf = min(1.0, cluster_count / 7.0)
            mean_level = float(np.mean([lvl for _, lvl in best_cluster]))
            level_conf = min(1.0, max(0.0, (mean_level - (peak_threshold - 12.0)) / 24.0))
            span_mhz = (best_cluster[-1][0] - best_cluster[0][0]) / 1e6 if cluster_count >= 2 else 0.0
            span_conf = min(1.0, max(0.0, span_mhz / 30.0))
            comb_confidence = cluster_conf * 0.5 + level_conf * 0.35 + span_conf * 0.15

            if comb_confidence > best_confidence:
                best_confidence = comb_confidence
                best_frequency = float(np.mean([freq for freq, _ in best_cluster]))

    current_time = time.time()
    rf_history = [(t, freq, conf) for t, freq, conf in rf_history if current_time - t < 6.0]

    # Detección inmediata para señales muy claras.
    if best_confidence > 0.65:
        return {
            "is_drone": True,
            "confidence": min(1.0, best_confidence),
            "frequency": best_frequency,
        }, rf_history

    # Señales moderadas: confirmar por persistencia temporal.
    if best_confidence > 0.35:
        rf_history.append((current_time, best_frequency, best_confidence))
        if len(rf_history) >= 2:
            avg_confidence = np.mean([conf for _, _, conf in rf_history])
            avg_frequency = np.mean([freq for _, freq, _ in rf_history])
            if avg_confidence > 0.45:
                return {
                    "is_drone": True,
                    "confidence": min(1.0, avg_confidence),
                    "frequency": avg_frequency,
                }, rf_history

    return {"is_drone": False, "confidence": 0.0, "frequency": None}, rf_history
