import json
import os


def normalize_model_path(path, base_dir):
    """Normaliza una ruta de modelo para desarrollo y ejecutable compilado."""
    if not path:
        return ""

    if os.path.isabs(path):
        try:
            rel_path = os.path.relpath(path, base_dir)
            if not rel_path.startswith(".."):
                normalized = os.path.join(base_dir, os.path.basename(path))
                if os.path.exists(normalized):
                    return normalized
        except Exception:
            pass

        if os.path.exists(path):
            return path

        filename = os.path.basename(path)
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            return candidate
        candidate_models = os.path.join(base_dir, "models", filename)
        if os.path.exists(candidate_models):
            return candidate_models
        return path

    full_path = os.path.join(base_dir, path)
    if os.path.exists(full_path):
        return full_path

    filename = os.path.basename(path) if path else ""
    if filename:
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            return candidate
        candidate_models = os.path.join(base_dir, "models", filename)
        if os.path.exists(candidate_models):
            return candidate_models

    return path


def load_yolo_models_config(config_path, base_dir):
    """Carga configuración de slots YOLO y devuelve slots/default/model_path."""
    default_slots = [{"path": "best.pt", "description": "Modelo por defecto"}] + [
        {"path": "", "description": ""} for _ in range(14)
    ]
    yolo_model_slots = default_slots
    yolo_default_slot = 0

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                slots = data.get("slots", [])
                while len(slots) < 15:
                    slots.append({"path": "", "description": ""})

                for slot in slots:
                    if slot.get("path"):
                        slot["path"] = normalize_model_path(slot["path"], base_dir)
                        if os.path.isabs(slot["path"]):
                            try:
                                rel_path = os.path.relpath(slot["path"], base_dir)
                                if not rel_path.startswith(".."):
                                    slot["path"] = os.path.basename(slot["path"])
                            except Exception:
                                pass

                yolo_model_slots = slots[:15]
                yolo_default_slot = int(data.get("default_slot", 0))
        except Exception as e:
            print(f"[YOLO] No se pudo leer configuración de modelos: {e}")
            yolo_model_slots = default_slots
            yolo_default_slot = 0

    if not (0 <= yolo_default_slot < len(yolo_model_slots)):
        yolo_default_slot = 0

    default_path = yolo_model_slots[yolo_default_slot].get("path") or "best.pt"
    default_path = normalize_model_path(default_path, base_dir)
    if not default_path or not os.path.exists(default_path):
        default_path_models = os.path.join(base_dir, "models", "best.pt")
        default_path = default_path_models if os.path.exists(default_path_models) else os.path.join(base_dir, "best.pt")

    return yolo_model_slots, yolo_default_slot, default_path


def save_yolo_models_config(config_path, base_dir, yolo_model_slots, yolo_default_slot):
    """Guarda configuración de slots YOLO."""
    try:
        slots_to_save = []
        for slot in yolo_model_slots:
            slot_copy = slot.copy()
            path = slot_copy.get("path", "")
            if path:
                if os.path.isabs(path):
                    try:
                        rel_path = os.path.relpath(path, base_dir)
                        if not rel_path.startswith(".."):
                            slot_copy["path"] = os.path.basename(path)
                        else:
                            slot_copy["path"] = path
                    except Exception:
                        if os.path.exists(os.path.join(base_dir, os.path.basename(path))):
                            slot_copy["path"] = os.path.basename(path)
                        else:
                            slot_copy["path"] = path
                else:
                    slot_copy["path"] = path
            slots_to_save.append(slot_copy)

        data = {"slots": slots_to_save, "default_slot": yolo_default_slot}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[YOLO] No se pudo guardar configuración de modelos: {e}")
