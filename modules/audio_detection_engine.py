import os
import queue
import time

import numpy as np
import tensorflow as tf

# Modo de dispositivo para TensorFlow en audio:
# - auto (default): usa GPU si está disponible; si falla, cae a CPU.
# - cpu: fuerza CPU.
# - gpu: fuerza intentar GPU.
_TF_AUDIO_DEVICE = os.environ.get("ADAS3_TF_AUDIO_DEVICE", "auto").strip().lower()
if _TF_AUDIO_DEVICE not in {"auto", "cpu", "gpu"}:
    _TF_AUDIO_DEVICE = "auto"

if _TF_AUDIO_DEVICE == "cpu":
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

# Parámetros de robustez para reducir falsos positivos por ruido de base.
_AUDIO_NOISE_FLOOR_ALPHA = 0.05
_AUDIO_DETECTION_CONFIRM_WINDOWS = 2
# Compuerta de silencio en escala int16 (mean abs). Si el nivel está por
# debajo, evitamos inferencia y reportamos 0% para reducir ruido base.
_AUDIO_SILENCE_GATE_MEAN_ABS = float(os.environ.get("ADAS3_AUDIO_SILENCE_GATE", "30"))


def load_audio_model(audio_model_path, audio_mean_path, audio_std_path):
    """Carga modelo de detección de audio y estadísticas de normalización."""
    try:
        if not audio_model_path:
            print("Error: ruta de modelo de audio vacía")
            return False, None, None, None

        if not (audio_mean_path and audio_std_path):
            print("ERROR: Rutas de normalización inválidas")
            return False, None, None, None

        if not os.path.exists(audio_model_path):
            print(f"Error: No se encuentra el modelo '{audio_model_path}'")
            return False, None, None, None

        if os.path.exists(audio_mean_path) and os.path.exists(audio_std_path):
            audio_mean = np.load(audio_mean_path)
            audio_std = np.load(audio_std_path)
            print(f"Estadísticas cargadas - Mean: {audio_mean:.4f}, Std: {audio_std:.4f}")
        else:
            print("ERROR: No se encontraron archivos de normalización")
            return False, None, None, None

        print("Cargando modelo de detección de audio...")
        # compile=False evita warning de métricas no construidas en modelos de inferencia.
        try:
            audio_model = tf.keras.models.load_model(audio_model_path, compile=False)
        except Exception as e:
            # En modo auto, si CUDA falla, reintentar una vez forzando CPU.
            err = str(e)
            cuda_failure = "cuda" in err.lower() or "cuinit" in err.lower()
            if _TF_AUDIO_DEVICE == "auto" and cuda_failure:
                print("[AUDIO] CUDA no disponible para TensorFlow audio, reintentando en CPU...")
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:
                    pass
                audio_model = tf.keras.models.load_model(audio_model_path, compile=False)
            else:
                raise
        print("Modelo de audio cargado correctamente")
        return True, audio_model, audio_mean, audio_std
    except Exception as e:
        print(f"Error al cargar modelo de audio: {e}")
        return False, None, None, None


def run_audio_detection_worker(
    is_detection_enabled_fn,
    audio_buffer,
    audio_duration_seconds,
    extract_features_fn,
    get_audio_model_fn,
    get_threshold_fn,
    visual_multiplier,
    alert_duration_seconds,
    get_alert_state_fn,
    set_alert_state_fn,
    set_detection_result_fn,
    on_detection_event_fn=None,
    classify_size_fn=None,
):
    """Loop del worker de detección de audio con dependencias inyectadas.

    ``classify_size_fn`` (opcional) es una callable ``(audio_bytes) ->
    (size_class: str, size_confidence: float)`` que se invoca cuando la
    predicción supera el umbral. Devuelve la clasificación por firma sonora
    (tamaño del dron). La clase persiste durante toda la ventana de alerta
    y se propaga a ``set_detection_result_fn`` y al ``on_detection_event_fn``.

    ``set_detection_result_fn`` debe aceptar kwargs ``size_class`` y
    ``size_confidence`` (los ignora si no los usa, son opcionales).
    ``get_alert_state_fn``/``set_alert_state_fn`` pueden operar con la tupla
    ampliada ``(alert_time, max_confidence, size_class, size_confidence)`` o
    con la clásica de 2 campos (compatibilidad hacia atrás).
    """
    accumulated_audio = b""
    required_bytes = int(44100 * audio_duration_seconds * 2)
    first_prediction_shown = False
    noise_floor = 0.0
    consecutive_above_threshold = 0
    silence_gate_hits = 0

    def _unpack_alert_state():
        """Normaliza get_alert_state_fn a (alert_time, max_conf, size, size_conf)."""
        state = get_alert_state_fn()
        if state is None:
            return None, 0.0, "", 0.0
        if len(state) == 2:
            alert_time, max_conf = state
            return alert_time, max_conf, "", 0.0
        alert_time, max_conf, size_class, size_conf = (
            state[0], state[1], state[2] if len(state) > 2 else "",
            state[3] if len(state) > 3 else 0.0,
        )
        return alert_time, max_conf, size_class or "", float(size_conf or 0.0)

    def _pack_alert_state(alert_time, max_conf, size_class, size_conf):
        try:
            set_alert_state_fn(alert_time, max_conf, size_class, size_conf)
        except TypeError:
            set_alert_state_fn(alert_time, max_conf)

    def _push_detection_result(is_drone, confidence, size_class, size_conf):
        try:
            set_detection_result_fn(
                is_drone, confidence,
                size_class=size_class,
                size_confidence=size_conf,
            )
        except TypeError:
            set_detection_result_fn(is_drone, confidence)

    print("[AUDIO] Worker iniciado")

    while is_detection_enabled_fn():
        try:
            chunk = audio_buffer.get(timeout=1)
            accumulated_audio += chunk

            if len(accumulated_audio) >= required_bytes:
                try:
                    audio_window = accumulated_audio[:required_bytes]
                    if _AUDIO_SILENCE_GATE_MEAN_ABS > 0:
                        mean_abs_int16 = float(np.mean(np.abs(np.frombuffer(audio_window, dtype=np.int16))))
                        if mean_abs_int16 < _AUDIO_SILENCE_GATE_MEAN_ABS:
                            # Silencio/ruido muy bajo: sin inferencia para evitar falsos "raw".
                            consecutive_above_threshold = 0
                            current_time = time.time()
                            alert_time, max_confidence, size_class, size_confidence = _unpack_alert_state()
                            is_drone = False
                            if alert_time is not None:
                                elapsed = current_time - alert_time
                                if elapsed < alert_duration_seconds:
                                    # Mantener alerta activa durante toda la ventana configurada.
                                    is_drone = True
                                else:
                                    alert_time = None
                                    max_confidence = 0.0
                                    size_class = ""
                                    size_confidence = 0.0
                            _pack_alert_state(alert_time, max_confidence, size_class, size_confidence)
                            _push_detection_result(is_drone, 0.0, size_class, size_confidence)
                            silence_gate_hits += 1
                            if silence_gate_hits == 1 or silence_gate_hits % 20 == 0:
                                print(
                                    f"[AUDIO] Silencio detectado (mean_abs={mean_abs_int16:.1f} < "
                                    f"{_AUDIO_SILENCE_GATE_MEAN_ABS:.1f}). Predicción forzada a 0%."
                                )
                            overlap_bytes = int(44100 * 0.5 * 2)
                            accumulated_audio = accumulated_audio[-overlap_bytes:]
                            continue

                    features = extract_features_fn(accumulated_audio[:required_bytes])
                    audio_model = get_audio_model_fn()

                    if features is not None and audio_model is not None:
                        features = features[..., np.newaxis]
                        features = np.expand_dims(features, axis=0)
                        prediction = float(audio_model.predict(features, verbose=0)[0][0])

                        threshold = float(get_threshold_fn())
                        current_time = time.time()
                        alert_time, max_confidence, size_class, size_confidence = _unpack_alert_state()

                        # Actualizar suelo de ruido cuando estamos por debajo del umbral.
                        if prediction < threshold:
                            noise_floor = (
                                (1.0 - _AUDIO_NOISE_FLOOR_ALPHA) * noise_floor
                                + _AUDIO_NOISE_FLOOR_ALPHA * prediction
                            )
                        effective_prediction = max(0.0, prediction - noise_floor)
                        visual_confidence = min(1.0, effective_prediction * visual_multiplier)

                        if not first_prediction_shown:
                            print(
                                f"[AUDIO] Primera predicción: {visual_confidence*100:.1f}% "
                                f"(raw: {prediction*100:.1f}%)"
                            )
                            first_prediction_shown = True

                        if prediction >= threshold:
                            consecutive_above_threshold += 1
                        else:
                            consecutive_above_threshold = 0

                        # Clasificar la firma sonora cuando la predicción actual
                        # dispara (aunque todavía no hayamos confirmado la alerta):
                        # así aprovechamos la energía reciente que hizo subir el modelo.
                        current_size_class = ""
                        current_size_confidence = 0.0
                        if classify_size_fn is not None and prediction >= threshold:
                            try:
                                classified = classify_size_fn(audio_window)
                                if isinstance(classified, tuple) and len(classified) >= 2:
                                    current_size_class = str(classified[0] or "")
                                    current_size_confidence = float(classified[1] or 0.0)
                            except Exception as size_err:
                                print(f"[AUDIO] Clasificador de tamaño falló: {size_err}")

                        if consecutive_above_threshold >= _AUDIO_DETECTION_CONFIRM_WINDOWS:
                            if alert_time is None:
                                alert_time = current_time
                                max_confidence = effective_prediction
                                size_class = current_size_class
                                size_confidence = current_size_confidence
                                alert_time_str = time.strftime("%H:%M:%S", time.localtime(current_time))
                                visual_pct = min(100, int(visual_confidence * 100))
                                size_tag = (
                                    f" | Tamaño: {size_class} {int(size_confidence * 100)}%"
                                    if size_class and size_class != "inconclusive"
                                    else ""
                                )
                                print(
                                    f"[AUDIO] ⚠ DRON DETECTADO A LAS {alert_time_str} - "
                                    f"{visual_pct}% (raw: {prediction*100:.1f}%){size_tag}"
                                )
                                if on_detection_event_fn:
                                    try:
                                        on_detection_event_fn(
                                            {
                                                "type": "initial",
                                                "timestamp": current_time,
                                                "visual_confidence": float(visual_confidence),
                                                "raw_prediction": float(prediction),
                                                "size_class": size_class,
                                                "size_confidence": float(size_confidence),
                                            }
                                        )
                                    except Exception as callback_error:
                                        print(f"[AUDIO] Callback de detección falló: {callback_error}")
                            else:
                                if effective_prediction > max_confidence:
                                    max_confidence = effective_prediction
                                # Mejorar la clase de tamaño sólo si la nueva
                                # clasificación es más fiable que la guardada.
                                if (
                                    current_size_class
                                    and current_size_class != "inconclusive"
                                    and current_size_confidence > size_confidence
                                ):
                                    size_class = current_size_class
                                    size_confidence = current_size_confidence
                                if prediction > 0.5:
                                    alert_time = current_time
                                    alert_time_str = time.strftime("%H:%M:%S", time.localtime(current_time))
                                    visual_pct = min(100, int(visual_confidence * 100))
                                    size_tag = (
                                        f" | Tamaño: {size_class} {int(size_confidence * 100)}%"
                                        if size_class and size_class != "inconclusive"
                                        else ""
                                    )
                                    print(
                                        f"[AUDIO] ⚠ NUEVA DETECCIÓN A LAS {alert_time_str} - "
                                        f"{visual_pct}% (raw: {prediction*100:.1f}%){size_tag}"
                                    )
                                    if on_detection_event_fn:
                                        try:
                                            on_detection_event_fn(
                                                {
                                                    "type": "refresh",
                                                    "timestamp": current_time,
                                                    "visual_confidence": float(visual_confidence),
                                                    "raw_prediction": float(prediction),
                                                    "size_class": size_class,
                                                    "size_confidence": float(size_confidence),
                                                }
                                            )
                                        except Exception as callback_error:
                                            print(f"[AUDIO] Callback de detección falló: {callback_error}")

                        is_drone = False
                        if alert_time is not None:
                            elapsed = current_time - alert_time
                            if elapsed < alert_duration_seconds:
                                is_drone = True
                            else:
                                alert_time = None
                                max_confidence = 0.0
                                size_class = ""
                                size_confidence = 0.0

                        _pack_alert_state(alert_time, max_confidence, size_class, size_confidence)
                        _push_detection_result(
                            is_drone,
                            float(visual_confidence),
                            size_class,
                            size_confidence,
                        )

                        status = "⚠ ALERTA ACTIVA" if is_drone else ""
                        visual_pct = min(100, int(visual_confidence * 100))
                        size_suffix = (
                            f" | Tamaño: {size_class} {int(size_confidence * 100)}%"
                            if is_drone and size_class and size_class != "inconclusive"
                            else ""
                        )
                        print(
                            f"[AUDIO] Predicción: {visual_pct}% (raw: {prediction*100:.1f}%) | "
                            f"Drone: {is_drone} {status}{size_suffix}"
                        )
                except Exception as e:
                    print(f"[AUDIO] Error: {e}")

                overlap_bytes = int(44100 * 0.5 * 2)
                accumulated_audio = accumulated_audio[-overlap_bytes:]

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[AUDIO] Error crítico: {e}")

    print("[AUDIO] Worker finalizado")
