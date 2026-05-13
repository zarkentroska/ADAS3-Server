import librosa
import numpy as np


# -----------------------------------------------------------------------------
# Clasificación del tamaño del dron por firma sonora (heurística espectral).
# -----------------------------------------------------------------------------
#
# Idea: la frecuencia de paso de pala (BPF = RPM/60 * n_palas) es más baja en
# drones con hélices grandes (a bajas rpm) y más alta en drones pequeños con
# hélices cortas (que giran mucho más rápido). Integramos la energía espectral
# en tres bandas típicas y elegimos la dominante.
#
# Las bandas se solapan intencionadamente poco para dar separación clara y
# evitar que ruido de fondo pulle la clasificación hacia "medium" por estar
# entre medias.
DRONE_SIZE_BANDS_HZ = {
    "large":  (40.0, 150.0),
    "medium": (150.0, 350.0),
    "small":  (350.0, 1500.0),
}

# Confianza relativa mínima (energía de la banda dominante / energía total en
# el rango 40–1500 Hz) para comprometerse con una clase. Por debajo de esto
# marcamos "inconclusive" para no mostrar un suffix erróneo.
_DRONE_SIZE_MIN_CONFIDENCE = 0.45


def classify_drone_size_from_audio(
    audio_window_bytes,
    source_sample_rate=44100,
    min_duration_seconds=0.5,
):
    """Clasifica un dron como ``small``/``medium``/``large`` por firma sonora.

    Parámetros
    ----------
    audio_window_bytes : bytes
        Ventana de audio PCM int16 mono tal y como llega del stream (sin
        resamplear ni normalizar).
    source_sample_rate : int
        Sample rate de ``audio_window_bytes``.
    min_duration_seconds : float
        Duración mínima de la ventana para intentar clasificar.

    Devuelve
    --------
    (size_class, confidence) : tuple[str, float]
        ``size_class`` es ``"small" | "medium" | "large" | "inconclusive"``.
        ``confidence`` es la fracción de energía [0, 1] que cae en la banda
        asignada respecto al total en 40–1500 Hz.
    """
    try:
        data = np.frombuffer(audio_window_bytes, dtype=np.int16)
    except (TypeError, ValueError):
        return "inconclusive", 0.0

    if data.size < int(source_sample_rate * min_duration_seconds):
        return "inconclusive", 0.0

    samples = data.astype(np.float32) / 32768.0

    # Ventaneado Hann para reducir spectral leakage del pico espectral.
    window = np.hanning(samples.size)
    spectrum = np.fft.rfft(samples * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(samples.size, 1.0 / float(source_sample_rate))

    band_energy = {}
    for name, (low, high) in DRONE_SIZE_BANDS_HZ.items():
        mask = (freqs >= low) & (freqs < high)
        band_energy[name] = float(power[mask].sum())

    total = sum(band_energy.values())
    if total <= 1e-9:
        return "inconclusive", 0.0

    best_name = max(band_energy, key=band_energy.get)
    confidence = band_energy[best_name] / total

    if confidence < _DRONE_SIZE_MIN_CONFIDENCE:
        return "inconclusive", float(confidence)
    return best_name, float(confidence)


def extract_features_realtime(
    audio_chunk,
    audio_sample_rate,
    audio_duration,
    n_mels,
    n_fft,
    hop_length,
    audio_mean,
    audio_std,
    spectrogram_sink=None,
):
    """Extrae features de un chunk de audio en tiempo real."""
    try:
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)

        raw_min, raw_max = np.min(audio_data), np.max(audio_data)
        raw_mean = np.mean(np.abs(audio_data))

        audio_data = audio_data / 32768.0

        norm_min, norm_max = np.min(audio_data), np.max(audio_data)
        norm_mean = np.mean(np.abs(audio_data))

        mean_abs_level = np.mean(np.abs(audio_data))

        if mean_abs_level < 0.005:
            audio_gain = 40.0
        elif mean_abs_level < 0.01:
            audio_gain = 30.0
        elif mean_abs_level < 0.02:
            audio_gain = 20.0
        else:
            audio_gain = 10.0

        audio_data = audio_data * audio_gain
        audio_data = np.clip(audio_data, -1.0, 1.0)

        if len(audio_data) > 0:
            audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=audio_sample_rate)

        required_length = audio_sample_rate * audio_duration
        if len(audio_data) < required_length:
            audio_data = np.pad(audio_data, (0, required_length - len(audio_data)))
        else:
            audio_data = audio_data[:required_length]

        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=audio_sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_min, mel_max = np.min(mel_spec_db), np.max(mel_spec_db)
        mel_mean = np.mean(mel_spec_db)

        if spectrogram_sink is not None:
            freqs_mel = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=audio_sample_rate / 2)
            spectrogram_sink(freqs_mel, mel_spec_db)

        if audio_mean is not None and audio_std is not None:
            mel_spec_db = (mel_spec_db - audio_mean) / (audio_std + 1e-8)

            norm_mel_min, norm_mel_max = np.min(mel_spec_db), np.max(mel_spec_db)
            norm_mel_mean = np.mean(mel_spec_db)

            if not hasattr(extract_features_realtime, "_call_count"):
                extract_features_realtime._call_count = 0
            extract_features_realtime._call_count += 1

            if extract_features_realtime._call_count % 10 == 0:
                gain_min, gain_max = np.min(audio_data), np.max(audio_data)
                gain_mean = np.mean(np.abs(audio_data))
                print(
                    f"[DEBUG AUDIO] Raw: min={raw_min:.0f}, max={raw_max:.0f}, mean_abs={raw_mean:.0f} | "
                    f"Norm: min={norm_min:.4f}, max={norm_max:.4f}, mean_abs={norm_mean:.4f} (level={mean_abs_level:.5f}) | "
                    f"Gain {audio_gain:.1f}x: min={gain_min:.4f}, max={gain_max:.4f}, mean_abs={gain_mean:.4f} | "
                    f"Mel dB: min={mel_min:.2f}, max={mel_max:.2f}, mean={mel_mean:.2f} | "
                    f"Mel norm: min={norm_mel_min:.2f}, max={norm_mel_max:.2f}, mean={norm_mel_mean:.2f}"
                )
        else:
            return None

        if mel_spec_db.shape[1] < 87:
            pad_width = 87 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)))
        else:
            mel_spec_db = mel_spec_db[:, :87]

        return mel_spec_db

    except Exception as e:
        print(f"[FEATURES] Error: {e}")
        return None
