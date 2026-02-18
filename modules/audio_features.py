import librosa
import numpy as np


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
