import os
import queue

import cv2
from ultralytics import YOLO

# Tamaño típico de entrenamiento YOLOv8 / Roboflow para este proyecto (ver build_content.py / train).
YOLO_DEFAULT_TRAIN_IMGSZ = 640


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


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ab = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def merge_detection_boxes(boxes, iou_thr=0.45):
    """NMS greedy por clase para fusionar detecciones solapadas (p. ej. entre tiles)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: -float(b.get("conf", 0.0)))
    kept = []
    while boxes:
        b = boxes.pop(0)
        kept.append(b)
        cls_b = b.get("cls", 0)
        nxt = []
        for x in boxes:
            if x.get("cls", 0) != cls_b or _box_iou(b, x) < iou_thr:
                nxt.append(x)
        boxes = nxt
    return kept


def _resolve_spatial_mode(mode, width, height, dual_h_min_width, dual_h_min_height):
    if mode != "auto":
        return mode
    if width >= dual_h_min_width or height >= dual_h_min_height:
        return "dual_h"
    return "single"


def _tiles_dual_horizontal(width, height, overlap_frac):
    mid = width // 2
    ow = max(1, int(width * overlap_frac))
    return [
        (0, 0, min(mid + ow, width), height),
        (max(0, mid - ow), 0, width, height),
    ]


def _tiles_quad(width, height, overlap_frac):
    mid_x, mid_y = width // 2, height // 2
    ow = max(1, int(width * overlap_frac))
    oh = max(1, int(height * overlap_frac))
    return [
        (0, 0, min(mid_x + ow, width), min(mid_y + oh, height)),
        (max(0, mid_x - ow), 0, width, min(mid_y + oh, height)),
        (0, max(0, mid_y - oh), min(mid_x + ow, width), height),
        (max(0, mid_x - ow), max(0, mid_y - oh), width, height),
    ]


def _tiles_grid_3x2(width, height, overlap_frac):
    """Seis recortes solapados (3 columnas × 2 filas) para máxima cobertura en 16:9."""
    w1, w2 = width // 3, 2 * width // 3
    h1 = height // 2
    ow = max(1, int(width * overlap_frac * 0.38))
    oh = max(1, int(height * overlap_frac * 0.38))
    col_ranges = [
        (0, min(w1 + ow, width)),
        (max(0, w1 - ow), min(w2 + ow, width)),
        (max(0, w2 - ow), width),
    ]
    row_ranges = [
        (0, min(h1 + oh, height)),
        (max(0, h1 - oh), height),
    ]
    rects = []
    for x1, x2 in col_ranges:
        for y1, y2 in row_ranges:
            rects.append((x1, y1, x2, y2))
    return rects


def _collect_tiles(frame, spatial_mode, overlap_frac, width, height):
    if spatial_mode == "single":
        return [(frame, 0, 0)]
    if spatial_mode == "dual_h":
        rects = _tiles_dual_horizontal(width, height, overlap_frac)
    elif spatial_mode == "quad":
        rects = _tiles_quad(width, height, overlap_frac)
    elif spatial_mode == "grid_3x2":
        rects = _tiles_grid_3x2(width, height, overlap_frac)
    else:
        return [(frame, 0, 0)]
    out = []
    for x1, y1, x2, y2 in rects:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        out.append((roi, x1, y1))
    return out if out else [(frame, 0, 0)]


def _extract_boxes_from_result(result, model, yolo_scale, ox, oy):
    boxes_data = []
    res_boxes = result.boxes
    if res_boxes is None or len(res_boxes) == 0:
        return boxes_data
    for box in res_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        inv = 1.0 / yolo_scale
        x1 = int(x1 * inv + ox)
        y1 = int(y1 * inv + oy)
        x2 = int(x2 * inv + ox)
        y2 = int(y2 * inv + oy)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names[cls] if cls < len(model.names) else f"Class {cls}"
        boxes_data.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": conf,
                "cls": cls,
                "class_name": class_name,
            }
        )
    return boxes_data


def run_yolo_inference_worker(
    is_running_fn,
    frame_queue,
    get_model_fn,
    get_thresholds_fn,
    get_infer_config_fn,
    set_result_fn,
):
    """Worker dedicado para inferencia YOLO.

    ``get_infer_config_fn`` debe devolver un dict (actualizable en caliente) con:
      - spatial_mode: 'single' | 'dual_h' | 'quad' | 'grid_3x2' | 'auto'
      - yolo_scale: float
      - tile_overlap: float
      - infer_imgsz: int o None
      - merge_iou: float o None (None = usar IoU del slider)
      - dual_h_min_width, dual_h_min_height: int (solo si spatial_mode es 'auto')

    spatial_mode:
      - 'single': un pase sobre el frame completo (máximo FPS).
      - 'dual_h': dos mitades horizontales solapadas (mejor para drones pequeños en HD).
      - 'quad': cuatro cuadrantes solapados.
      - 'grid_3x2': seis teselas 3×2 con solape (máxima cobertura por software en esta app).
      - 'auto': dual_h si ancho o alto superan los umbrales, si no single.

    Los recortes se envían en un solo batch al modelo cuando hay varios, para amortizar
    la latencia en GPU frente a varias llamadas secuenciales.
    """
    print("[YOLO] Worker thread iniciado")

    while is_running_fn():
        item = None
        try:
            item = frame_queue.get(timeout=0.1)
            frame_original, original_shape = item
            model = get_model_fn()
            if model is None:
                continue

            cfg = get_infer_config_fn()
            spatial_mode = cfg.get("spatial_mode", "auto")
            yolo_scale = float(cfg.get("yolo_scale", 0.5))
            tile_overlap = float(cfg.get("tile_overlap", 0.18))
            infer_imgsz = cfg.get("infer_imgsz")
            merge_iou = cfg.get("merge_iou")
            dual_h_min_width = int(cfg.get("dual_h_min_width", 800))
            dual_h_min_height = int(cfg.get("dual_h_min_height", 800))

            h0, w0 = int(original_shape[0]), int(original_shape[1])
            mode = _resolve_spatial_mode(spatial_mode, w0, h0, dual_h_min_width, dual_h_min_height)
            tiles = _collect_tiles(frame_original, mode, tile_overlap, w0, h0)

            conf_thr, iou_thr = get_thresholds_fn()
            merge = merge_iou if merge_iou is not None else iou_thr

            smalls = []
            metas = []
            for roi, ox, oy in tiles:
                rh, rw = roi.shape[:2]
                sw, sh = int(rw * yolo_scale), int(rh * yolo_scale)
                if sw < 1 or sh < 1:
                    continue
                smalls.append(cv2.resize(roi, (sw, sh)))
                metas.append((ox, oy, yolo_scale))

            if not smalls:
                continue

            pred_kw = dict(verbose=False, conf=conf_thr, iou=iou_thr)
            iz = infer_imgsz
            if iz is not None:
                pred_kw["imgsz"] = int(iz)

            batch_in = smalls if len(smalls) > 1 else smalls[0]
            try:
                raw_out = model(batch_in, **pred_kw)
            except Exception as e:
                print(f"[YOLO] Inferencia por lote falló ({e}); reintentando tile a tile.")
                raw_out = []
                for s in smalls:
                    one = model(s, **pred_kw)
                    raw_out.append(one)
            if not isinstance(raw_out, list):
                raw_out = [raw_out]

            boxes_all = []
            for res, (ox, oy, ys) in zip(raw_out, metas):
                boxes_all.extend(_extract_boxes_from_result(res, model, ys, ox, oy))

            merged = merge_detection_boxes(boxes_all, merge)
            detecciones = len(merged)

            set_result_fn(frame_original, detecciones, merged)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[YOLO] Error en worker: {e}")
        finally:
            if item is not None:
                try:
                    frame_queue.task_done()
                except ValueError:
                    pass

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
