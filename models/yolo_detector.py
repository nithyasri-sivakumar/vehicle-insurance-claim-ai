import os
from typing import List, Optional, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

COCO_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "scooter"}
YOLO_MODEL_NAME = "yolov8n.pt"

# If ultralytics is not installed or the model fails to load,
# the detector will gracefully fall back to the rest of the
# vehicle detection pipeline instead of raising an import error.


def load_yolo_model() -> Optional[YOLO]:
    if YOLO is None:
        print("[YOLO Detector] ultralytics is not installed; YOLO detection disabled")
        return None

    model_path = os.path.join(os.path.dirname(__file__), YOLO_MODEL_NAME)

    if os.path.exists(model_path):
        try:
            return YOLO(model_path)
        except Exception as e:
            print(f"[YOLO Detector] Failed to load local model: {e}")
            print("[YOLO Detector] Continuing without YOLO, using fallback detections")
            return None

    try:
        return YOLO(YOLO_MODEL_NAME)
    except Exception as e:
        print(f"[YOLO Detector] Failed to load YOLO model by name: {e}")
        print("[YOLO Detector] Continuing without YOLO, using fallback detections")
        return None


yolo_model = load_yolo_model()


def detect_yolo_vehicles(image_path: str, confidence_threshold: float = 0.4) -> Tuple[float, Optional[str], List[str]]:
    """
    Detect vehicles using YOLO and COCO classes.

    Returns:
        confidence: Best detection confidence
        best_class: Detected vehicle class
        debug_logs: List of per-detection strings
    """
    if yolo_model is None:
        return 0.0, None, ["[YOLO Detector] YOLO unavailable - fallback only"]

    if not os.path.exists(image_path):
        return 0.0, None, ["[YOLO Detector] Image path does not exist"]

    debug_logs = []
    best_confidence = 0.0
    best_class = None

    try:
        results = yolo_model(image_path, imgsz=640, conf=confidence_threshold, verbose=False)
        if not results:
            return 0.0, None, ["[YOLO Detector] No results returned"]

        data = results[0]
        boxes = getattr(data, 'boxes', None)
        if boxes is None:
            return 0.0, None, ["[YOLO Detector] No boxes in results"]

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = data.names.get(class_id, 'unknown') if hasattr(data, 'names') else str(class_id)
            debug_logs.append(f"detected: {class_name} ({confidence:.2f})")

            if class_name in COCO_VEHICLE_CLASSES and confidence > best_confidence:
                best_confidence = confidence
                best_class = class_name

        return best_confidence, best_class, debug_logs

    except Exception as e:
        return 0.0, None, [f"[YOLO Detector] detection error: {e}"]


def create_yolo_data_yaml(dataset_dir: str, output_path: str = "data.yaml") -> None:
    """
    Create a YOLO dataset config file for training on a custom dataset.
    """
    content = f"""
train: {os.path.join(dataset_dir, 'train')}
val: {os.path.join(dataset_dir, 'val')}
nc: {len(COCO_VEHICLE_CLASSES)}
names: {sorted(list(COCO_VEHICLE_CLASSES))}
"""
    with open(output_path, 'w') as f:
        f.write(content.strip())
    print(f"[YOLO Detector] Created data config at {output_path}")


def train_yolo_model(data_yaml: str, epochs: int = 50, imgsz: int = 640):
    """
    Train or fine-tune a YOLO model on your custom dataset.
    """
    if YOLO is None:
        raise ImportError("ultralytics is required to train YOLO models")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"{data_yaml} not found")

    model = YOLO(YOLO_MODEL_NAME)
    print(f"[YOLO Detector] Starting training on {data_yaml} for {epochs} epochs")
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
    print("[YOLO Detector] Training complete")
    return model
