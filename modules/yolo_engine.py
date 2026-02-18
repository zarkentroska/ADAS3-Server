import os
import queue

import cv2
from ultralytics import YOLO


def load_yolo_model(model_path):
    """Carga el modelo YOLO desde disco."""
    try:
        if not model_path:
            print("Error: ruta de modelo YOLO vacía")
            return False, None
        if not os.path.exists(model_path):
            print(f"Error: No se encuentra el modelo '{model_path}'")
            return False, None

        print("Cargando modelo YOLO...")
        model = YOLO(model_path)
        print(f"Modelo YOLO cargado - Dispositivo: {model.device}")
        return True, model
    except Exception as e:
        print(f"Error al cargar modelo YOLO: {e}")
        return False, None


def run_yolo_inference_worker(
    is_running_fn,
    frame_queue,
    get_model_fn,
    get_thresholds_fn,
    yolo_scale,
    set_result_fn,
):
    """Worker dedicado para inferencia YOLO."""
    print("[YOLO] Worker thread iniciado")

    while is_running_fn():
        try:
            frame_original, original_shape = frame_queue.get(timeout=0.1)

            model = get_model_fn()
            if model is None:
                continue

            small_frame = cv2.resize(
                frame_original,
                (int(original_shape[1] * yolo_scale), int(original_shape[0] * yolo_scale)),
            )

            conf_thr, iou_thr = get_thresholds_fn()
            results = model(small_frame, verbose=False, conf=conf_thr, iou=iou_thr)

            boxes_data = []
            detecciones = 0

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1 = int(x1 / yolo_scale)
                    y1 = int(y1 / yolo_scale)
                    x2 = int(x2 / yolo_scale)
                    y2 = int(y2 / yolo_scale)

                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls] if cls < len(model.names) else f"Class {cls}"
                    boxes_data.append(
                        {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "class_name": class_name}
                    )
                    detecciones += 1

            set_result_fn(frame_original, detecciones, boxes_data)
            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[YOLO] Error en worker: {e}")

    print("[YOLO] Worker thread finalizado")


def clear_queue_safely(frame_queue):
    """Vacía una cola sin bloquear."""
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break


def draw_yolo_detections(frame, boxes_data):
    """Dibuja las detecciones YOLO en el frame."""
    for box in boxes_data:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        conf = box["conf"]
        class_name = box["class_name"]

        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name}: {conf:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(
            frame,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1,
        )
        cv2.putText(frame, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    return frame
