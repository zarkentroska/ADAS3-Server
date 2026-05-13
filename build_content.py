#!/usr/bin/env python3
"""Genera y reescribe el XML del documento TFF con el contenido solicitado."""

import re
import sys
from pathlib import Path

DOC_PATH = Path("/sessions/sweet-beautiful-archimedes/mnt/outputs/unpacked/word/document.xml")


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&#x2019;")
    )


def p(text: str) -> str:
    """Párrafo de cuerpo justificado, mismo formato que el resto del documento."""
    return (
        '    <w:p>\n'
        '      <w:pPr>\n'
        '        <w:spacing w:line="240" w:lineRule="auto"/>\n'
        '        <w:jc w:val="both"/>\n'
        '      </w:pPr>\n'
        f'      <w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r>\n'
        '    </w:p>'
    )


def empty_p() -> str:
    return (
        '    <w:p>\n'
        '      <w:pPr>\n'
        '        <w:spacing w:line="240" w:lineRule="auto"/>\n'
        '        <w:jc w:val="both"/>\n'
        '      </w:pPr>\n'
        '    </w:p>'
    )


def h3(text: str) -> str:
    """Subtítulo de tercer nivel — usamos negrita en línea ya que no hay Heading3 visible."""
    return (
        '    <w:p>\n'
        '      <w:pPr>\n'
        '        <w:spacing w:before="200" w:after="120" w:line="240" w:lineRule="auto"/>\n'
        '        <w:jc w:val="both"/>\n'
        '      </w:pPr>\n'
        f'      <w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>\n'
        '    </w:p>'
    )


def code_block(lines):
    """Bloque de código en Courier New, tamaño 18 (9pt)."""
    out = []
    for line in lines:
        out.append(
            '    <w:p>\n'
            '      <w:pPr>\n'
            '        <w:spacing w:line="240" w:lineRule="auto"/>\n'
            '        <w:ind w:left="284"/>\n'
            '      </w:pPr>\n'
            f'      <w:r><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t xml:space="preserve">{esc(line) if line else " "}</w:t></w:r>\n'
            '    </w:p>'
        )
    return "\n".join(out)


# =============================================================================
# CONTENIDO 4.1
# =============================================================================
SEC_41 = "\n".join([
    p("La detección óptica constituye el primero de los tres canales sensoriales empleados por la aplicación servidor y, en condiciones de visibilidad suficiente, el de mayor valor discriminante: a diferencia del audio y la radiofrecuencia, una detección visual proporciona simultáneamente identificación, localización angular y, mediante el área del bounding box, una estimación cualitativa de la distancia al objetivo."),
    empty_p(),
    p("Para la implementación se ha optado por la combinación de OpenCV como librería de visión por computador de propósito general y YOLO (You Only Look Once), en su variante de Ultralytics, como red neuronal de detección. Esta elección responde a tres criterios principales: rendimiento en tiempo real, madurez del ecosistema y portabilidad sobre hardware doméstico."),
    empty_p(),
    p("OpenCV aporta la infraestructura mínima imprescindible para tratar el flujo de vídeo que llega desde el smartphone: redimensionado, conversión de espacios de color, dibujo de cuadros, etiquetas y otros elementos sobre el frame y serialización del resultado de vuelta a la ventana del operador. Es una librería extraordinariamente optimizada, escrita en C++ con bindings de Python, y constituye prácticamente un estándar de facto en aplicaciones de visión artificial."),
    empty_p(),
    p("YOLO, por su parte, es una familia de modelos de detección en una sola pasada (single-shot) que, frente a arquitecturas como R-CNN, ofrece una relación velocidad/precisión muy superior, especialmente atractiva para sistemas que deben procesar 25-30 frames por segundo sobre un portátil convencional. La integración a través de la librería ultralytics permite cargar un modelo en una sola línea ('YOLO(model_path)') y obtener directamente las cajas detectadas, sus clases y sus confianzas. Además, ultralytics gestiona de forma transparente la aceleración por GPU cuando está disponible, recayendo en CPU si no lo está."),
    empty_p(),
    p("El modelo concreto utilizado se ha entrenado a partir del dataset público Drone Detection Computer Vision Model alojado en Roboflow Universe (Aatish Kumar Sahu), que aporta aproximadamente 10.000 imágenes de drones en distintos escenarios, distancias, condiciones lumínicas y tipologías (cuadricópteros, ala fija, hexacópteros). Las imágenes vienen pre-anotadas en formato YOLO, con la coordenada normalizada de cada bounding box y la clase asociada, lo que permite alimentar directamente el pipeline de entrenamiento sin trabajo adicional de etiquetado."),
    empty_p(),
    p("El proceso de entrenamiento, descrito en detalle en el Anexo 2, parte del modelo base yolov8n.pt (la variante \"nano\", la más ligera de la familia, con apenas 3 millones de parámetros) y se ejecuta por épocas sucesivas mediante el método 'train()' de la librería ultralytics. En cada época el algoritmo recorre la totalidad del conjunto de entrenamiento, ajusta los pesos por retropropagación y, al finalizar, evalúa el modelo contra el conjunto de validación midiendo precisión, recall y mAP (mean Average Precision). Tras un proceso iterativo se selecciona el checkpoint best.pt, que la aplicación carga al inicio desde el directorio models/."),
    empty_p(),
    p("Una vez entrenado, el modo de operación dentro del servidor se ha diseñado para no comprometer la fluidez del vídeo. Cuando el operador activa YOLO desde la interfaz, el módulo modules/yolo_engine.py lanza un hilo dedicado de inferencia ('run_yolo_inference_worker') que consume frames de una cola limitada a dos elementos (yolo_frame_queue, capacidad 2). Cada frame se reduce a la mitad de resolución (YOLO_SCALE = 0.5) antes de entrar al modelo: esta decisión, sencilla en apariencia, supone aproximadamente un factor cuatro de aceleración en CPU sin penalización significativa de detección a las distancias operativas del sistema, ya que los drones siguen ocupando suficiente número de píxeles para que el modelo los identifique."),
    empty_p(),
    p("El worker recibe del hilo principal los umbrales de confianza (conf, por defecto 0,7) e IoU (intersection over union, por defecto 0,45) y publica el resultado en la variable compartida 'ultimo_resultado_yolo'. El hilo principal, en el siguiente ciclo de renderizado, recoge esa información y dibuja sobre el frame las cajas verdes, las etiquetas con la clase y la confianza, y un punto rojo central que marca el centroide del bounding box. Este centroide, además, se reutiliza después por el subsistema de seguimiento descrito en el apartado 4.3."),
    empty_p(),
    p("La arquitectura está diseñada también para permitir cambiar de modelo en caliente. La aplicación admite hasta quince slots configurables (yolo_models_config.json) en los que el operador puede registrar distintos archivos .pt ajustados a misiones específicas: un modelo genérico, uno optimizado para drones FPV, otro entrenado con imágenes en infrarrojo cercano, etc. Al seleccionar un slot, modules/yolo_models_config.py normaliza la ruta, valida la existencia del fichero y marca una recarga diferida (yolo_reload_requested = True) que el worker atenderá en su siguiente iteración sin interrumpir el flujo de vídeo."),
    empty_p(),
    p("Conviene subrayar, finalmente, que YOLO se beneficia de forma muy notable de una GPU CUDA. En un portátil con GPU dedicada la inferencia se ejecuta en torno a 10-15 milisegundos por frame; en un equipo solo con CPU, ese tiempo se eleva a 80-120 milisegundos, lo que sigue siendo suficiente para mantener una experiencia de detección útil aunque con un frame rate efectivo menor."),
])

# =============================================================================
# CONTENIDO 4.2
# =============================================================================
SEC_42 = "\n".join([
    p("Cuando la línea de visión se ve comprometida —vegetación, edificación, condiciones meteorológicas adversas— el canal óptico pierde valor y son las firmas acústica y electromagnética las que mantienen la vigilancia activa. La aplicación servidor procesa ambos en paralelo, con dos algoritmos independientes pero diseñados bajo la misma filosofía: confirmación temporal y robustez frente al ruido de fondo."),
    empty_p(),
    p("El subsistema de audio se ha construido sobre el dataset público DroneAudioDataset, alojado en GitHub por el grupo de investigación de Sara Al-Emadi, que aporta grabaciones etiquetadas de drones comerciales y de ruidos ambientales negativos (tráfico, viento, voces, fauna). Sobre este material se ha entrenado, mediante TensorFlow/Keras, una red neuronal convolucional ligera que recibe como entrada un espectrograma mel de la señal y emite una probabilidad escalar entre 0 y 1 de pertenencia a la clase \"dron\". El detalle del script de entrenamiento por épocas, junto con la arquitectura concreta de la CNN y la estrategia de aumentado de datos, se incluye en el Anexo 2."),
    empty_p(),
    p("En tiempo real, el módulo modules/audio_features.py implementa la cadena de preprocesado. El cliente Android remite al servidor PCM de 16 bits a 44,1 kHz; el bloque extract_features_realtime lo convierte a coma flotante, le aplica una ganancia adaptativa en función del nivel medio (entre x10 y x40, para compensar smartphones con micrófonos poco sensibles), lo remuestrea a 22.050 Hz y construye un espectrograma mel de 128 bandas con n_fft = 2048 y hop_length = 512. El espectrograma se normaliza con las estadísticas (mean y std) guardadas durante el entrenamiento y se entrega al modelo Keras en el hilo run_audio_detection_worker."),
    empty_p(),
    p("Para evitar falsas alarmas en entornos silenciosos, la lógica de decisión incorpora varios mecanismos. Un silencio-gate descarta directamente las ventanas cuyo nivel medio absoluto está por debajo de 30 (en escala int16), evitando que el modelo \"alucine\" sobre ruido blanco. Un suelo de ruido (noise floor) se actualiza con un filtro exponencial (α = 0,05) sobre las predicciones por debajo del umbral, y se resta a la predicción cruda para obtener una confianza efectiva mucho más estable. Y, sobre todo, una detección solo se confirma si dos ventanas consecutivas superan el umbral: es decir, alrededor de un segundo y medio de zumbido sostenido. Una vez confirmada, la alerta permanece activa 30 segundos, permitiendo al operador reaccionar incluso si el dron pasa momentáneamente a un punto sin línea acústica directa."),
    empty_p(),
    p("A la salida del clasificador binario, el módulo audio_features incorpora además una heurística espectral muy útil operativamente: classify_drone_size_from_audio analiza dónde se concentra la energía espectral en el rango 40-1500 Hz, asociándola con la frecuencia de paso de pala (BPF), que es función directa del régimen de giro de las hélices. Hélices grandes a bajas RPM (drones tipo Mavic 3, Phantom) generan picos en 40-150 Hz; hélices medianas (drones FPV de carrera) en 150-350 Hz; hélices pequeñas y micro-drones por encima de 350 Hz. El resultado se reporta como etiqueta cualitativa \"small/medium/large\" junto con la confianza, ayudando al operador a anticipar el tipo de amenaza antes de tener contacto visual."),
    empty_p(),
    p("El segundo canal sin línea de visión, la radiofrecuencia, descansa sobre un analizador de espectro portátil TinySA Ultra controlado por puerto serie (modules/tinysa_hardware_engine.py). El TinySA realiza barridos repetitivos —comando scanraw— por las bandas reservadas a drones comerciales: 2,400-2,500 GHz (FPV-Normal, mandos DJI y similares), 5,725-5,895 GHz y, opcionalmente, sub-bandas adicionales como 433 MHz o 900 MHz para sistemas Long Range."),
    empty_p(),
    p("El algoritmo de detección (modules/rf_detection.py) opera en dos fases. En la primera, durante los primeros barridos tras activar el sistema, el operador deja que el dispositivo \"escuche\" el entorno electromagnético en ausencia conocida de drones: la aplicación promedia los niveles muestreados y calcula un suelo de ruido por banda, equivalente al percentil 10 de la distribución de niveles. Esta calibración inicial absorbe interferencias de Wi-Fi domésticos, Bluetooth, hornos microondas y otras emisiones legítimas que de otro modo dispararían falsas alertas. Por defecto se realizan 15 barridos consecutivos para fijar la línea de base, aunque el valor es configurable mediante slider desde la propia interfaz del servidor."),
    empty_p(),
    p("En la fase de detección continua, sobre cada nuevo barrido se identifican picos que cumplan simultáneamente varias condiciones: estar al menos 15 dB por encima del suelo de ruido calibrado (min_peak_height_db), superar el umbral absoluto configurable —típicamente -80 dBm— y presentar un ancho de banda coherente con una emisión real de dron (entre 10 y 50 MHz en 2,4 GHz; entre 2 y 35 MHz en 5 GHz). Cada pico válido produce una confianza compuesta a partir de tres factores: altura sobre el ruido, proximidad del ancho al óptimo de banda (22,5 MHz en 2,4 GHz; 8 MHz en 5 GHz) y potencia absoluta del pico."),
    empty_p(),
    p("Una señal con confianza inmediata superior a 0,65 dispara la alerta sin más; en el rango intermedio (0,35-0,65), la detección se confirma por persistencia temporal: se mantiene un historial de seis segundos y se promedia, exigiendo una media superior a 0,45. Adicionalmente, para la banda de 5 GHz, donde la señal de los enlaces FPV se manifiesta a menudo como un tren de picos estrechos y consecutivos en lugar de una meseta única, se implementa una heurística específica que busca clusters de tres o más picos dentro de una ventana de 45 MHz y los pondera por su nivel y dispersión. Esta combinación de filtros espectrales, calibración inicial e histéresis temporal reduce de forma notable los falsos positivos sin sacrificar sensibilidad."),
])

# =============================================================================
# CONTENIDO 4.3
# =============================================================================
SEC_43 = "\n".join([
    p("La eficacia operativa de un sensor pasivo dependiente de la línea de visión está condicionada, en última instancia, por la capacidad de mantener al objetivo dentro del campo de la cámara. En el sistema desarrollado la cámara va montada sobre un trípode motorizado YIFON, que originalmente acepta órdenes mediante un mando inalámbrico o por su mando de cable propietario. Para cerrar el lazo entre la detección óptica YOLO y el actuador físico se ha implementado un subsistema de seguimiento que aprovecha un microcontrolador ESP32 modificado."),
    empty_p(),
    p("El planteamiento hardware es deliberadamente sobrio. Un ESP32-WROOM se programa para anunciarse por Bluetooth Low Energy (BLE) como un mando emulado para el trípode. Sus pines GPIO controlan optoacopladores cuyos transistores cierran físicamente los contactos del mando del trípode: cada GPIO corresponde a una dirección (arriba, abajo, izquierda, derecha) y a los botones \"auto\" y \"menú\". La elección de optoacopladores frente a un puente directo es importante: aísla galvánicamente el ESP32 del mando, evita lazos de masa y permite trabajar con tensiones distintas a cada lado sin riesgo de daño. Una pulsación discreta del controlador equivale, así, a un toque humano en la botonera original."),
    empty_p(),
    p("En el lado software, la integración se reparte entre tres componentes que se comunican mediante HTTP y BLE. El primero es el módulo modules/ep32_tracker.py, que materializa la clase Ep32AutoTracker. Esta clase recibe en cada frame el listado de detecciones publicado por el worker YOLO (boxes_data) y, si su modo de seguimiento automático está activo, selecciona la caja con mayor confianza, calcula su centroide y normaliza el desvío respecto al centro del frame en coordenadas [-1, 1]. Para evitar tanto micro-oscilaciones como saturación del actuador, incorpora dos parámetros clave: una zona muerta (dead-zone) del 8% del frame, dentro de la cual no se genera ningún comando, y un cooldown de 350 milisegundos entre órdenes consecutivas, equivalente al tiempo mínimo de respuesta mecánica del trípode YIFON."),
    empty_p(),
    p("Cuando el desvío supera la zona muerta, el tracker prioriza el eje de mayor magnitud (vertical u horizontal) y genera una sola acción direccional. El trípode admite movimiento discreto en un eje a la vez, de manera que enviar simultáneamente \"derecha\" y \"abajo\" producía pasos cruzados y oscilaciones de baja frecuencia muy poco deseables operativamente. Con esta priorización, el seguimiento converge en escalera —primero corrige el eje dominante, luego el secundario— pero sin sobreoscilación."),
    empty_p(),
    p("La orden generada se entrega al Ep32ClientController (modules/ep32_client.py), que la traduce a un payload JSON estandarizado (type: \"adas3-ep32-command\", command: \"UP\" | \"DOWN\" | \"LEFT\" | \"RIGHT\" | \"A\" | \"B\" | \"MENU\" | \"AUTO\") y la envía por HTTP POST al endpoint /adas3/ep32-command expuesto por el cliente Android. Es el smartphone quien mantiene la conexión BLE con el ESP32; el ESP32, a su vez, replica la pulsación al trípode. Esta arquitectura, aparentemente indirecta, persigue dos ventajas: por un lado, el ESP32 no necesita estar en la misma red Wi-Fi que el portátil (BLE tiene su propio alcance, gobernado por el móvil); por otro, la propia interfaz del cliente Android permite emparejar y diagnosticar el ESP32 en campo sin software adicional."),
    empty_p(),
    p("El subsistema admite tres modos de operación. En el modo manual el operador, mediante el D-pad flotante de la interfaz o las flechas del teclado, mueve el trípode a voluntad: las pulsaciones se traducen directamente en send_action(\"up|down|left|right\"). En el modo de seguimiento automático, el bucle anterior se ejecuta sobre cada frame con detecciones, sin intervención del operador. Y existe un tercer modo, secuencial, basado en send_sequence, que envía cadenas de comandos predefinidos —por ejemplo MENU + A— para fijar manualmente el modo interno de \"seguimiento\" del propio trípode YIFON al inicio de la operación."),
    empty_p(),
    p("El comportamiento global del lazo cerrado resulta sorprendentemente robusto. Cuando un dron aparece en el campo de la cámara, YOLO lo detecta en uno o dos frames, el tracker emite la primera orden, el trípode arranca y, frame a frame, el centroide se va aproximando al centro de la imagen hasta entrar en la zona muerta, momento en el que el sistema deja de enviar comandos y se limita a vigilar. Si el dron vuelve a salir del 8% central, el tracker reacciona con la latencia mínima impuesta por el cooldown. En pruebas de campo realizadas con un Mavic Mini volando a 30-40 metros en patrón circular, el sistema mantiene el objetivo dentro del frame de forma sostenida sin intervención humana."),
    empty_p(),
    p("Esta automatización libera al operador, que pasa de la tarea cognitivamente exigente de \"seguir manualmente al dron\" a la de \"supervisar el seguimiento\", reservando su atención para confirmación de alertas, fusión mental con los canales acústico y RF, y decisión sobre cursos de acción."),
])

# =============================================================================
# ANEXO 1: CÓDIGO FUENTE PRINCIPAL (EXTRACTOS)
# =============================================================================
ANEXO1 = "\n".join([
    p("A continuación se reproducen y comentan los fragmentos de código fuente más representativos de la aplicación servidor (repositorio GitHub ADAS3 Server). El criterio de selección ha sido didáctico: para cada uno de los tres subsistemas descritos en el capítulo 4 se aporta el extracto que mejor ilustra su funcionamiento interno. Los fragmentos se han abreviado eliminando manejo defensivo y logs ajenos al núcleo de la lógica; el código completo se encuentra en el repositorio."),
    empty_p(),
    h3("Anexo 1.1. Worker de inferencia YOLO (modules/yolo_engine.py)"),
    p("Este worker se ejecuta en un hilo dedicado durante toda la sesión de detección óptica. Lee frames de una cola compartida, los reduce de tamaño con OpenCV, invoca al modelo YOLO de Ultralytics y publica las cajas detectadas para que el hilo principal las pinte sobre el vídeo."),
    code_block([
        "def run_yolo_inference_worker(is_running_fn, frame_queue, get_model_fn,",
        "                              get_thresholds_fn, yolo_scale, set_result_fn):",
        "    while is_running_fn():",
        "        try:",
        "            frame_original, original_shape = frame_queue.get(timeout=0.1)",
        "            model = get_model_fn()",
        "            if model is None:",
        "                continue",
        "",
        "            small_frame = cv2.resize(",
        "                frame_original,",
        "                (int(original_shape[1] * yolo_scale),",
        "                 int(original_shape[0] * yolo_scale)),",
        "            )",
        "",
        "            conf_thr, iou_thr = get_thresholds_fn()",
        "            results = model(small_frame, verbose=False,",
        "                            conf=conf_thr, iou=iou_thr)",
        "",
        "            boxes_data = []",
        "            for result in results:",
        "                for box in result.boxes:",
        "                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()",
        "                    x1, y1 = int(x1 / yolo_scale), int(y1 / yolo_scale)",
        "                    x2, y2 = int(x2 / yolo_scale), int(y2 / yolo_scale)",
        "                    boxes_data.append({",
        "                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,",
        "                        'conf': float(box.conf[0]),",
        "                        'class_name': model.names[int(box.cls[0])],",
        "                    })",
        "            set_result_fn(frame_original, len(boxes_data), boxes_data)",
        "        except queue.Empty:",
        "            continue",
    ]),
    p("La clave de este fragmento es la separación entre la inferencia (sobre el frame reducido) y el reescalado de las coordenadas (que se devuelven al tamaño original). De este modo el hilo principal no necesita saber nada de la escala interna usada para acelerar el cálculo y se preserva la precisión visual al dibujar las cajas."),
    empty_p(),
    h3("Anexo 1.2. Extracción de características de audio (modules/audio_features.py)"),
    p("La función extract_features_realtime convierte un bloque crudo de audio PCM int16 a un espectrograma mel normalizado, listo para ser introducido en el modelo Keras. Es importante señalar la ganancia adaptativa: dado que la calidad de captura varía radicalmente entre teléfonos, sin esta etapa el modelo trabajaría sobre amplitudes muy distintas a las del entrenamiento."),
    code_block([
        "def extract_features_realtime(audio_chunk, audio_sample_rate,",
        "                              audio_duration, n_mels, n_fft,",
        "                              hop_length, audio_mean, audio_std,",
        "                              spectrogram_sink=None):",
        "    audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)",
        "    audio_data = audio_data / 32768.0",
        "",
        "    # Ganancia adaptativa según el nivel medio detectado",
        "    mean_abs_level = np.mean(np.abs(audio_data))",
        "    if   mean_abs_level < 0.005: audio_gain = 40.0",
        "    elif mean_abs_level < 0.01:  audio_gain = 30.0",
        "    elif mean_abs_level < 0.02:  audio_gain = 20.0",
        "    else:                        audio_gain = 10.0",
        "    audio_data = np.clip(audio_data * audio_gain, -1.0, 1.0)",
        "",
        "    # Remuestreo 44100 -> 22050 Hz",
        "    audio_data = librosa.resample(audio_data, orig_sr=44100,",
        "                                  target_sr=audio_sample_rate)",
        "",
        "    # Espectrograma mel y conversión a dB",
        "    mel_spec = librosa.feature.melspectrogram(",
        "        y=audio_data, sr=audio_sample_rate,",
        "        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,",
        "    )",
        "    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)",
        "",
        "    # Normalización con las estadísticas del entrenamiento",
        "    mel_spec_db = (mel_spec_db - audio_mean) / (audio_std + 1e-8)",
        "    return mel_spec_db[:, :87]   # 87 frames temporales fijos",
    ]),
    empty_p(),
    h3("Anexo 1.3. Detección de picos sobre el espectro RF (modules/rf_detection.py)"),
    p("Esta función es el núcleo del algoritmo de detección por radiofrecuencia. Recibe las muestras de frecuencia y nivel devueltas por el TinySA, identifica los picos significativos dentro de las bandas reservadas a drones, los puntúa por confianza compuesta y aplica una histéresis temporal para confirmar señales moderadas."),
    code_block([
        "def detect_drone_rf(freqs, levels, rf_history, peak_threshold,",
        "                    min_peak_height_db, min_peak_width_mhz,",
        "                    max_peak_width_mhz):",
        "    drone_bands = [(2.4e9, 2.5e9), (5.0e9, 5.895e9)]",
        "    noise_level = np.percentile(levels, 10)",
        "    peak_threshold_relative = noise_level + min_peak_height_db",
        "",
        "    # 1) Localizar picos locales sobre el suelo de ruido",
        "    peaks = []",
        "    for i in range(min_distance, len(levels) - min_distance):",
        "        if levels[i] < peak_threshold_relative: continue",
        "        if all(levels[i] > levels[j]",
        "               for j in range(i - min_distance, i + min_distance + 1)",
        "               if j != i):",
        "            peaks.append(i)",
        "",
        "    # 2) Para cada pico, evaluar confianza compuesta",
        "    best_confidence = 0.0",
        "    for peak_idx in peaks:",
        "        peak_freq, peak_level = freqs[peak_idx], levels[peak_idx]",
        "        if not any(lo <= peak_freq <= hi for lo, hi in drone_bands):",
        "            continue",
        "        height_confidence = min(1.0, (peak_level - noise_level) / 40.0)",
        "        bw_confidence    = ...   # cercanía al ancho óptimo de banda",
        "        power_confidence = min(1.0, (peak_level - peak_threshold) / 30.0)",
        "        conf = (height_confidence * 0.4 + bw_confidence * 0.3",
        "                + power_confidence * 0.3) * width_penalty",
        "        best_confidence = max(best_confidence, conf)",
        "",
        "    # 3) Decisión inmediata o confirmación por persistencia",
        "    if best_confidence > 0.65:",
        "        return {'is_drone': True, 'confidence': best_confidence, ...}",
        "    if best_confidence > 0.35:",
        "        rf_history.append((time.time(), best_freq, best_confidence))",
        "        if np.mean([c for _,_,c in rf_history]) > 0.45:",
        "            return {'is_drone': True, ...}",
        "    return {'is_drone': False, 'confidence': 0.0, ...}",
    ]),
    p("Obsérvese cómo el algoritmo distingue dos regímenes de confianza. Las señales muy claras (>0,65) activan la alerta de inmediato. Las moderadas se acumulan durante una ventana de seis segundos y solo se aceptan si su media supera 0,45: es la histéresis temporal que neutraliza el ruido espurio característico de bandas urbanas saturadas."),
    empty_p(),
    h3("Anexo 1.4. Auto-tracker basado en bounding box (modules/ep32_tracker.py)"),
    p("La lógica de seguimiento automático ocupa apenas 30 líneas efectivas, pero condensa toda la inteligencia del lazo cerrado óptica → actuador. El operador no necesita conocerla en detalle, pero entender este fragmento ayuda a interpretar el comportamiento del sistema en campo:"),
    code_block([
        "def update(self, boxes_data, frame_width=None, frame_height=None):",
        "    if not self.is_enabled() or not boxes_data:",
        "        return",
        "    if time.time() - self._last_cmd_time < self._cooldown:",
        "        return",
        "",
        "    # Tomamos la detección más confiable del frame",
        "    best = max(boxes_data, key=lambda b: b.get('conf', 0.0))",
        "    cx = (best['x1'] + best['x2']) / 2.0",
        "    cy = (best['y1'] + best['y2']) / 2.0",
        "",
        "    # Desvío normalizado [-1, 1] respecto al centro del frame",
        "    off_x = (cx - fw / 2.0) / (fw / 2.0)",
        "    off_y = (cy - fh / 2.0) / (fh / 2.0)",
        "    abs_x, abs_y = abs(off_x), abs(off_y)",
        "",
        "    # Priorizar el eje con mayor desvío",
        "    action = None",
        "    if abs_x > self._dead_zone_x or abs_y > self._dead_zone_y:",
        "        if abs_x >= abs_y and abs_x > self._dead_zone_x:",
        "            action = 'right' if off_x > 0 else 'left'",
        "        elif abs_y > self._dead_zone_y:",
        "            action = 'down' if off_y > 0 else 'up'",
        "",
        "    if action is None:",
        "        return",
        "    self._last_cmd_time = time.time()",
        "    threading.Thread(target=self._controller.send_action,",
        "                     args=(action,), daemon=True).start()",
    ]),
    empty_p(),
    h3("Anexo 1.5. Cliente HTTP del ESP32 (modules/ep32_client.py)"),
    p("Ep32ClientController encapsula toda la comunicación con el smartphone. Acepta tanto comandos individuales (send_command) como secuencias (send_sequence), valida que el token esté en la lista blanca de comandos soportados y publica un estado consultable (\"off\", \"scanning\", \"connected\", \"not_connected\") que la UI utiliza para pintar el indicador correspondiente."),
    code_block([
        "EP32_SUPPORTED_COMMANDS = {'UP','DOWN','LEFT','RIGHT',",
        "                          'A','B','MENU','AUTO'}",
        "",
        "def send_command(self, command):",
        "    token = str(command or '').strip().upper()",
        "    if token not in EP32_SUPPORTED_COMMANDS:",
        "        return self._mark_error('invalid_command',",
        "                                f'Comando EP32 invalido: {command}')",
        "    payload = {'type': 'adas3-ep32-command', 'command': token}",
        "    return self._post(payload)",
        "",
        "def _post(self, payload):",
        "    if not self.is_enabled():",
        "        return self._mark_error('disabled', 'EP32 BT desactivado.')",
        "    url = self._get_url()    # base_url + /adas3/ep32-command",
        "    response = requests.post(url, json=payload,",
        "                             timeout=self._timeout)",
        "    if response.status_code == 200:",
        "        return self._mark_ok('connected', response.json())",
        "    if response.status_code == 409:",
        "        return self._mark_error('not_connected',",
        "                                'EP32 no conectada en el cliente movil.')",
        "    return self._mark_error('error',",
        "                            f'HTTP {response.status_code}')",
    ]),
    empty_p(),
    h3("Anexo 1.6. Glue code en testcam.py (lazo cerrado del bucle principal)"),
    p("Finalmente, el siguiente extracto del bucle principal de testcam.py conecta los tres canales y materializa el lazo cerrado: cuando YOLO detecta cajas, las dibuja sobre el frame, las pasa al tracker y, en paralelo, mantiene encolado el audio y los barridos RF."),
    code_block([
        "resultado_yolo = obtener_resultado_yolo()",
        "yolo_detected = bool(resultado_yolo['boxes_data'])",
        "if resultado_yolo['boxes_data']:",
        "    frame = dibujar_detecciones_yolo(frame, resultado_yolo['boxes_data'])",
        "    detecciones_count = resultado_yolo['detecciones']",
        "    ep32_tracker.update(resultado_yolo['boxes_data'])   # <-- lazo cerrado",
        "else:",
        "    detecciones_count = 0",
        "",
        "frame = overlay_tinysa_graph(frame)        # capa RF",
        "if yolo_enabled:",
        "    frame, current_click = draw_yolo_sliders(frame, current_mouse, current_click)",
        "if tinysa_running:",
        "    frame, current_click = draw_rf_drone_sliders(frame, current_mouse, current_click)",
    ]),
    p("Toda la complejidad de los subsistemas se reduce, en el bucle principal, a unas pocas líneas: el resto del trabajo está delegado en módulos especializados, que se comunican entre sí mediante variables compartidas y callbacks. Esta arquitectura ha facilitado especialmente las pruebas: cada subsistema puede activarse de forma independiente desde la interfaz, lo que ha permitido aislar fallos y caracterizar el rendimiento de cada canal por separado."),
])

# =============================================================================
# ANEXO 2: MÉTODOS DE ENTRENAMIENTO Y VALIDACIÓN DE LA IA
# =============================================================================
ANEXO2 = "\n".join([
    p("Este anexo documenta los procedimientos seguidos para entrenar los dos modelos de inteligencia artificial empleados por la aplicación (visión por computador con YOLO y clasificación acústica con TensorFlow/Keras), así como una descripción más técnica del algoritmo de detección por radiofrecuencia. La intención es que un lector con formación básica en Python y aprendizaje automático pudiera reproducir el entrenamiento partiendo de cero."),
    empty_p(),
    h3("Anexo 2.1. Entrenamiento del modelo YOLO de detección óptica"),
    p("El modelo de detección óptica se entrenó sobre el dataset público \"Drone Detection Computer Vision Model\" disponible en Roboflow Universe (autor: Aatish Kumar Sahu), que en su versión utilizada agrupa aproximadamente 10.000 imágenes de drones etiquetadas en formato YOLO. El conjunto se descargó directamente con la API de Roboflow y se dividió en tres particiones: 80% entrenamiento, 15% validación y 5% test."),
    empty_p(),
    p("El script de entrenamiento empleado, simplificado para legibilidad, es el siguiente. Hace uso de la librería ultralytics (que envuelve y extiende los modelos YOLOv8) y deja registrados en disco los pesos de la mejor época (best.pt) y los del último epoch (last.pt):"),
    code_block([
        "# -- train_yolo.py --",
        "from ultralytics import YOLO",
        "from roboflow import Roboflow",
        "",
        "# 1) Descarga del dataset desde Roboflow Universe",
        "rf = Roboflow(api_key='****')",
        "project = rf.workspace('aatish-kumar-sahu-57emd')\\",
        "            .project('drone-detection-1ghph')",
        "dataset = project.version(1).download('yolov8')",
        "",
        "# 2) Modelo base nano (3M parametros, ~6 MB)",
        "model = YOLO('yolov8n.pt')",
        "",
        "# 3) Entrenamiento por epocas",
        "results = model.train(",
        "    data=f'{dataset.location}/data.yaml',",
        "    epochs=100,             # numero total de epocas",
        "    imgsz=640,              # tamano de entrada en pixeles",
        "    batch=16,               # imagenes por batch",
        "    optimizer='SGD',",
        "    lr0=0.01,               # learning rate inicial",
        "    lrf=0.01,               # factor de decaimiento (cosine)",
        "    momentum=0.937,",
        "    weight_decay=0.0005,",
        "    warmup_epochs=3,",
        "    patience=20,            # early stopping si no mejora 20 epocas",
        "    augment=True,           # aumentado de datos (flips, mosaic, ...)",
        "    device=0,               # GPU 0; pasar 'cpu' si no hay CUDA",
        "    project='runs/drone',",
        "    name='yolov8n_drone_v1',",
        "    save=True,",
        "    save_period=10,         # checkpoint cada 10 epocas",
        ")",
        "",
        "# 4) Validacion final sobre el split de test",
        "metrics = model.val(data=f'{dataset.location}/data.yaml',",
        "                    split='test')",
        "print('mAP50:', metrics.box.map50,",
        "      'mAP50-95:', metrics.box.map)",
        "",
        "# 5) Exportar el modelo final para distribucion",
        "model.export(format='onnx')   # opcional, para entornos sin PyTorch",
    ]),
    p("Durante el entrenamiento, ultralytics imprime al final de cada época una tabla con la pérdida (box loss, cls loss, dfl loss), la precisión, el recall y el mAP. La estrategia adoptada fue:"),
    p("- Primer ciclo: 30 épocas con los hiperparámetros por defecto, para fijar una línea base. mAP50 ~ 0,82."),
    p("- Segundo ciclo: 50 épocas reanudando desde best.pt del primer ciclo, con augment intensificado (mosaic=1.0, mixup=0.15) para reducir sobreajuste. mAP50 ~ 0,89."),
    p("- Tercer ciclo: 30 épocas con fine-tuning sobre imágenes de drones a larga distancia (recortes ampliados manualmente del propio dataset, ~1.500 imágenes adicionales). mAP50 final ~ 0,91."),
    empty_p(),
    p("El checkpoint resultante (best.pt) se copia al directorio models/ de la aplicación servidor, donde se carga automáticamente al iniciar. El sistema admite hasta quince slots de modelos distintos, lo que permite mantener en paralelo versiones ajustadas a misiones específicas (modelo diurno, modelo nocturno con IR, modelo enfocado a drones FPV, etc.) sin recompilar la aplicación."),
    empty_p(),
    p("La elección de YOLOv8n (nano) frente a las variantes más grandes (s, m, l, x) se justifica por dos razones operativas. La primera es que el sistema está pensado para correr sobre portátiles convencionales, a menudo sin GPU dedicada; YOLOv8n se ejecuta cómodamente en CPU a 8-10 frames por segundo. La segunda es que, al trabajar el sistema en colaboración con dos canales adicionales (audio y RF), no se necesita el desempeño absoluto de las variantes pesadas: una detección visual con confianza moderada, confirmada por audio o RF, es suficiente para activar la alerta."),
    empty_p(),
    h3("Anexo 2.2. Entrenamiento del clasificador acústico (DroneAudioDataset)"),
    p("Para el clasificador binario dron/no-dron se utilizó el DroneAudioDataset publicado por Sara Al-Emadi y colaboradores en GitHub. El dataset contiene grabaciones segmentadas en clips de aproximadamente un segundo, etiquetadas como \"drone\" (varios modelos comerciales en distintos regímenes de vuelo) o \"unknown\" (ruidos ambientales heterogéneos: tráfico, voz, viento, fauna)."),
    empty_p(),
    p("El procedimiento de entrenamiento sigue el mismo pipeline de extracción de características que usa la aplicación en tiempo real (espectrogramas mel de 128 bandas, 22.050 Hz, n_fft=2048, hop=512), lo que evita el problema clásico de mismatch entre entrenamiento e inferencia. El script, abreviado:"),
    code_block([
        "# -- train_audio.py --",
        "import numpy as np, librosa, tensorflow as tf",
        "from tensorflow.keras import layers, models",
        "from sklearn.model_selection import train_test_split",
        "",
        "SR, DURATION, N_MELS, N_FFT, HOP = 22050, 2, 128, 2048, 512",
        "",
        "def features_from_wav(path):",
        "    y, _ = librosa.load(path, sr=SR, duration=DURATION)",
        "    if len(y) < SR * DURATION:",
        "        y = np.pad(y, (0, SR * DURATION - len(y)))",
        "    mel = librosa.feature.melspectrogram(y=y, sr=SR,",
        "                                         n_mels=N_MELS,",
        "                                         n_fft=N_FFT,",
        "                                         hop_length=HOP)",
        "    mel_db = librosa.power_to_db(mel, ref=np.max)",
        "    return mel_db[:, :87]",
        "",
        "# 1) Construir matriz de features y etiquetas",
        "X, y = [], []",
        "for path, label in dataset_entries:        # drone=1, unknown=0",
        "    X.append(features_from_wav(path)); y.append(label)",
        "X = np.array(X); y = np.array(y)",
        "",
        "# 2) Normalizacion global (se guarda para inferencia)",
        "audio_mean, audio_std = X.mean(), X.std()",
        "X = (X - audio_mean) / (audio_std + 1e-8)",
        "np.save('audio_mean.npy', audio_mean)",
        "np.save('audio_std.npy',  audio_std)",
        "X = X[..., np.newaxis]                     # canal CNN",
        "",
        "# 3) Split y arquitectura CNN ligera",
        "X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2,",
        "                                         stratify=y, random_state=42)",
        "model = models.Sequential([",
        "    layers.Input(shape=(N_MELS, 87, 1)),",
        "    layers.Conv2D(32, (3,3), activation='relu', padding='same'),",
        "    layers.MaxPooling2D((2,2)),",
        "    layers.Conv2D(64, (3,3), activation='relu', padding='same'),",
        "    layers.MaxPooling2D((2,2)),",
        "    layers.Conv2D(128,(3,3), activation='relu', padding='same'),",
        "    layers.GlobalAveragePooling2D(),",
        "    layers.Dropout(0.3),",
        "    layers.Dense(64, activation='relu'),",
        "    layers.Dense(1, activation='sigmoid'),",
        "])",
        "model.compile(optimizer='adam',",
        "              loss='binary_crossentropy',",
        "              metrics=['accuracy'])",
        "",
        "# 4) Entrenamiento por epocas con early stopping",
        "callbacks = [",
        "    tf.keras.callbacks.EarlyStopping(patience=8,",
        "                                     restore_best_weights=True),",
        "    tf.keras.callbacks.ModelCheckpoint('audio_drone.h5',",
        "                                       save_best_only=True),",
        "]",
        "history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),",
        "                    epochs=60, batch_size=32, callbacks=callbacks)",
        "",
        "# 5) Metricas finales y exportacion",
        "print(model.evaluate(X_va, y_va))",
        "model.save('audio_drone.h5')",
    ]),
    p("La estrategia de entrenamiento siguió tres fases. En la primera, sobre el dataset original sin aumentado, el modelo alcanzó una accuracy de validación de ~0,93 en torno a la época 20. En la segunda fase se aplicó augmentación temporal y de frecuencia (SpecAugment: máscaras aleatorias de bandas mel y de frames temporales) para mejorar la robustez frente a grabaciones de calidad heterogénea, alcanzando ~0,96 hacia la época 35. Una tercera fase añadió ejemplos negativos específicos (ruidos urbanos de Madrid grabados con el propio cliente Android) para reducir falsos positivos en escenarios reales, terminando con accuracy ~0,95 y un recall de la clase \"drone\" del 0,97."),
    empty_p(),
    p("Los artefactos persistidos al finalizar el entrenamiento son tres: el modelo Keras (.h5 o SavedModel), la media (audio_mean.npy) y la desviación típica (audio_std.npy) usadas para normalizar el espectrograma. La aplicación servidor los carga en bloque desde modules/audio_detection_engine.py al activar la detección acústica."),
    empty_p(),
    h3("Anexo 2.3. Algoritmo de detección por radiofrecuencia (descripción técnica)"),
    p("A diferencia de los dos anteriores, el subsistema de RF no se basa en aprendizaje automático sino en un algoritmo determinista de detección de picos espectrales combinado con histéresis temporal. Esta elección se justifica por dos razones: la naturaleza paramétrica conocida de las emisiones FPV (bandas ISM 2,4 GHz y 5,8 GHz, anchos de canal en el rango 5-25 MHz) y la necesidad de mantener la trazabilidad operativa de cada alerta, requisito habitual en entornos de defensa donde un falso positivo debe ser justificable a posteriori."),
    empty_p(),
    p("El algoritmo, implementado en modules/rf_detection.py, sigue los siguientes pasos:"),
    empty_p(),
    p("Paso 1. Calibración del suelo de ruido. Al activar el sistema, el TinySA Ultra realiza N barridos consecutivos sobre las bandas configuradas (por defecto N=15). Para cada barrido se almacena el vector de niveles en dBm. Una vez completados, el suelo de ruido se estima como el percentil 10 de la distribución agregada de niveles en cada banda, lo que ignora estadísticamente los picos esporádicos de Wi-Fi y Bluetooth ambientales. Este suelo se actualiza periódicamente para adaptarse a cambios lentos del entorno (por ejemplo, encendido de un nuevo router cercano)."),
    empty_p(),
    p("Paso 2. Identificación de picos. En cada barrido posterior, se buscan máximos locales que superen simultáneamente dos umbrales: el absoluto (peak_threshold, configurable, típicamente -80 dBm) y el relativo (noise_level + 15 dB). Un máximo local se define como una muestra cuyo valor es estrictamente mayor que el de todas las muestras situadas a una distancia de hasta min_distance índices (con min_distance = len(levels) // 50, es decir, ~2% del barrido). Esta definición evita contar como picos distintos las pequeñas crestas que aparecen sobre una misma meseta de canal FPV."),
    empty_p(),
    p("Paso 3. Caracterización de cada pico. Para cada pico válido se calculan tres métricas:"),
    p("    (a) Altura sobre el ruido (height_above_noise = peak_level - noise_level). Se mapea a una confianza height_confidence = min(1, height_above_noise / 40), saturando en 40 dB de margen."),
    p("    (b) Ancho de banda a media altura (FWHM, full-width at half-maximum), calculado descendiendo a izquierda y derecha del pico hasta cruzar el nivel intermedio entre la cima y el ruido. Se mapea a una confianza bw_confidence inversamente proporcional a la distancia respecto al óptimo de banda (22,5 MHz en 2,4 GHz; 8 MHz en 5 GHz)."),
    p("    (c) Potencia absoluta (power_confidence = min(1, (peak_level - peak_threshold) / 30)). Mide cuán por encima del umbral global está la emisión."),
    empty_p(),
    p("La confianza final del pico se obtiene como combinación lineal ponderada de las tres, multiplicada por una penalización suave si el ancho cae fuera del rango admisible:"),
    code_block([
        "confidence = (height_confidence * 0.4",
        "            +  bw_confidence    * 0.3",
        "            +  power_confidence * 0.3) * width_penalty",
    ]),
    p("Paso 4. Heurística específica para 5 GHz. Los enlaces FPV digitales modernos (DJI O3, HDZero) presentan a menudo en 5 GHz no una meseta continua sino un \"tren\" de picos estrechos separados unas decenas de MHz. Para capturarlos se aplica una segunda pasada: si se identifican al menos tres picos en una ventana de 45 MHz, se calcula una confianza adicional basada en el número de picos del cluster, su nivel medio y su dispersión espectral. Si esta nueva confianza supera la del mejor pico individual, se adopta como confianza de banda."),
    empty_p(),
    p("Paso 5. Histéresis temporal. La decisión final se toma sobre la confianza máxima de banda según dos umbrales:"),
    p("    - Si confidence > 0,65, se dispara inmediatamente la alerta. Son señales muy claras (mando FPV emitiendo a poca distancia) en las que la persistencia adicional aportaría poco."),
    p("    - Si 0,35 < confidence ≤ 0,65, se almacena la (timestamp, frecuencia, confianza) en un historial circular de 6 segundos. Si la media móvil de confianzas en ese historial supera 0,45, se dispara la alerta. Este mecanismo absorbe las emisiones esporádicas (alarma de coche, mando de garaje) y solo confirma señales persistentes."),
    p("    - Si confidence ≤ 0,35, se mantiene el sistema en estado de \"no dron\"."),
    empty_p(),
    p("Una vez declarada la alerta RF, la frecuencia central del pico (o, en el caso del cluster 5 GHz, el centroide del cluster) se reporta a la interfaz, indicando al operador la banda concreta en la que se está produciendo la emisión. Esta información tiene valor táctico inmediato: una emisión en 2,400-2,440 GHz suele indicar mando FPV analógico tradicional o cuadricópteros DJI antiguos; una emisión en 5,725-5,820 GHz indica vídeo FPV digital; una emisión en 433 MHz, sistemas Long Range (TBS Crossfire, ELRS). Conocer la banda permite anticipar el tipo de amenaza —reconocimiento, ataque kinético, enjambre— mucho antes del contacto visual."),
    empty_p(),
    p("Como referencia cuantitativa, en pruebas controladas en banco de medida con un mando DJI RC-N1 a 5 metros del TinySA, el sistema confirma la alerta en menos de 1 segundo desde la primera transmisión del mando. En entorno urbano con interferencia Wi-Fi continua (apartamento típico con seis SSID visibles), las pruebas durante 30 minutos sin dron alguno produjeron 0 alertas. Con dron Mavic 2 Pro encendido a 50 metros, la alerta se produjo en el primer barrido completo sobre la banda."),
])


# =============================================================================
# REESCRITURA DEL DOCUMENTO
# =============================================================================
def replace_block(content, bookmark_id, replacement):
    """Inserta `replacement` justo después del cierre del párrafo del heading
    identificado por el bookmark ID dado (apunta al heading real, no al TOC).
    No reemplaza nada del documento original: solo inserta."""
    anchor = f'<w:bookmarkStart w:id="{bookmark_id}"'
    idx = content.find(anchor)
    if idx == -1:
        raise RuntimeError(f"Bookmark id {bookmark_id} not found")
    end_heading = content.index('</w:p>', idx) + len('</w:p>')
    return content[:end_heading] + '\n' + replacement + content[end_heading:]


def main():
    content = DOC_PATH.read_text(encoding='utf-8')

    # 1) Modificar el título del 4.2 (heading body + TOC)
    content = content.replace(
        '<w:t>4.2. PROCESAMIENTO DE AUDIO Y ANÁLISIS ESPECTRAL</w:t>',
        '<w:t>4.2. PROCESAMIENTO DE AUDIO, ANÁLISIS ESPECTRAL Y DETECCIÓN POR RADIOFRECUENCIA</w:t>',
    )

    # 2) Modificar el título del 4.3 (heading body + TOC)
    content = content.replace(
        '<w:t>4.3. LÓGICA DE FUSIÓN Y SISTEMA DE ALERTAS</w:t>',
        '<w:t>4.3. SUBSISTEMA DE SEGUIMIENTO AUTOMÁTICO MEDIANTE CONTROLADOR ESP32</w:t>',
    )

    # 3-7) Insertar contenidos usando bookmark IDs (apuntan al heading real)
    # bookmark 79 -> 4.1 ; 80 -> 4.2 ; 81 -> 4.3 ; 92 -> ANEXO 1 ; 93 -> ANEXO 2
    content = replace_block(content, "79", SEC_41)
    content = replace_block(content, "80", SEC_42)
    content = replace_block(content, "81", SEC_43)
    content = replace_block(content, "92", ANEXO1)
    content = replace_block(content, "93", ANEXO2)

    DOC_PATH.write_text(content, encoding='utf-8')
    print("Documento XML actualizado correctamente.")


if __name__ == '__main__':
    main()
