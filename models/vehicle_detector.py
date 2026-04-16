import cv2
import numpy as np
import os
import re
from typing import Tuple, Optional, Dict, List

from models.yolo_detector import detect_yolo_vehicles

class AdvancedVehicleDamageDetector:
    """
    Advanced vehicle detection and damage analysis system using computer vision.
    """

    # Allowed COCO class labels for vehicles
    VEHICLE_CLASS_IDS = {
        3: 'car',
        4: 'motorcycle',
        6: 'bus',
        8: 'truck'
    }

    MODEL_CONFIG = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    MODEL_WEIGHTS = 'frozen_inference_graph.pb'
    DNN_MIN_CONFIDENCE = 0.4

    # Damage-related keywords for fraud detection
    DAMAGE_KEYWORDS = {
        'front': ['front', 'hood', 'bumper', 'headlight', 'grille'],
        'rear': ['rear', 'back', 'trunk', 'tail', 'taillight'],
        'left': ['left', 'driver', 'side'],
        'right': ['right', 'passenger', 'side'],
        'roof': ['roof', 'top'],
        'undercarriage': ['under', 'bottom', 'undercarriage']
    }

    def __init__(self):
        self.vehicle_cascade = None
        self.load_cascades()

    def load_cascades(self):
        """Load Haar cascades for vehicle detection."""
        try:
            cascade_path = cv2.data.haarcascades + 'cars.xml'
            if os.path.exists(cascade_path):
                self.vehicle_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                print("Warning: Haar cascade file not found")
        except Exception as e:
            print(f"Warning: Could not load Haar cascades: {e}")

    def detect_vehicle(self, image_path: str, threshold: float = 0.4) -> Tuple[bool, float, Optional[str]]:
        """
        Detect if image contains a vehicle using pretrained model fallback and heuristic analysis.

        Returns:
            (is_vehicle, confidence, vehicle_type)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print("[Vehicle Detection] Could not read image")
                return False, 0.0, None

            h, w = image.shape[:2]

            if w < 200 or h < 200:
                print(f"[Vehicle Detection] Image too small: {w}x{h}")
                return False, 0.0, None

            # Try pretrained YOLO detection first, if available
            yolo_confidence, yolo_type, yolo_logs = detect_yolo_vehicles(image_path, threshold)
            for det in yolo_logs:
                print(f"[YOLO Output] {det}")

            if yolo_type:
                print(f"[Vehicle Detection] YOLO detected vehicle: {yolo_type} ({yolo_confidence:.2f})")

            if yolo_confidence >= threshold:
                return True, yolo_confidence, yolo_type

            # Try pretrained DNN vehicle detection next
            dnn_confidence, dnn_type, detections = self._detect_with_dnn(image)
            for det in detections:
                print(f"[Detection Output] {det}")

            if dnn_type:
                print(f"[Vehicle Detection] DNN detected vehicle: {dnn_type} ({dnn_confidence:.2f})")

            # If pretrained model detects a vehicle above threshold, accept it
            if dnn_confidence >= threshold:
                inferred_type = dnn_type or self._classify_vehicle_type(image, dnn_confidence)
                return True, dnn_confidence, inferred_type or 'unknown'

            # Accept near-threshold DNN detections for strong vehicle classes
            if dnn_type and dnn_confidence >= max(0.2, threshold - 0.08):
                inferred_type = dnn_type or self._classify_vehicle_type(image, dnn_confidence)
                print(f"[Vehicle Detection] Near-threshold DNN detection accepted: {inferred_type} ({dnn_confidence:.2f})")
                return True, dnn_confidence, inferred_type or 'unknown'

            # Otherwise use heuristic detection
            cascade_confidence = self._detect_with_cascade(image)
            shape_confidence = self._analyze_vehicle_shape(image)
            edge_confidence = self._analyze_edges_and_contours(image)

            total_confidence = max(
                dnn_confidence,
                cascade_confidence * 0.35 + shape_confidence * 0.35 + edge_confidence * 0.3
            )

            if self._looks_like_screen_or_document(image):
                print("[Vehicle Detection] Screen/document-like image rejected")
                return False, min(total_confidence, 0.2), None

            vehicle_type = dnn_type or self._classify_vehicle_type(image, total_confidence)

            print(f"[Vehicle Detection] cascade={cascade_confidence:.2f}, shape={shape_confidence:.2f}, edge={edge_confidence:.2f}, total={total_confidence:.2f}")

            if total_confidence >= threshold:
                if total_confidence < 0.5:
                    print("[Vehicle Detection] Low confidence detection accepted")
                if not vehicle_type:
                    vehicle_type = self._classify_vehicle_type(image, total_confidence) or 'unknown'
                return True, total_confidence, vehicle_type

            if total_confidence >= max(0.22, threshold - 0.08) and vehicle_type in {'car', 'truck', 'bus', 'motorcycle', 'scooter'}:
                print(f"[Vehicle Detection] Near-threshold vehicle-like image accepted as {vehicle_type} ({total_confidence:.2f})")
                return True, total_confidence, vehicle_type

            print("[Vehicle Detection] No valid vehicle detected")
            return False, total_confidence, vehicle_type

        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            return False, 0.0, None

    def _looks_like_screen_or_document(self, image: np.ndarray) -> bool:
        """Reject screenshots, documents, and similar flat UI-like images."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            edges = cv2.Canny(gray, 80, 160)

            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=120,
                minLineLength=80,
                maxLineGap=10
            )

            line_count = 0 if lines is None else len(lines)
            hv_count = 0

            if lines is not None:
                for line in lines[:, 0, :]:
                    x1, y1, x2, y2 = line
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    if dx == 0 or dy == 0 or max(dx, dy) / max(1, min(dx, dy)) > 8:
                        hv_count += 1

            hv_ratio = hv_count / max(1, line_count)
            saturation_mean = float(np.mean(hsv[:, :, 1]))
            brightness_mean = float(np.mean(hsv[:, :, 2]))
            bright_ratio = float(np.mean(gray > 210))

            thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresholded, 8)
            textish_components = 0
            for label_index in range(1, num_labels):
                _, _, width, height, area = stats[label_index]
                if 3 <= width <= 80 and 5 <= height <= 40 and 15 <= area <= 1200:
                    textish_components += 1

            # Browser windows, screenshots, and paper/document captures are
            # typically very bright, low saturation, and dominated by long
            # horizontal/vertical lines.
            if (
                line_count >= 80
                and hv_ratio >= 0.9
                and saturation_mean < 25
                and brightness_mean > 200
            ):
                return True

            if (
                textish_components >= 80
                and saturation_mean < 30
                and brightness_mean > 215
                and bright_ratio > 0.72
            ):
                return True

            if (
                line_count >= 180
                and hv_ratio >= 0.82
                and brightness_mean > 175
                and saturation_mean < 65
            ):
                return True

            return False

        except Exception as e:
            print(f"Screen/document rejection error: {e}")
            return False

    def _detect_with_dnn(self, image: np.ndarray) -> Tuple[float, Optional[str], List[str]]:
        """Detect vehicles using a pretrained COCO DNN model if available."""
        model_path = os.path.join(os.path.dirname(__file__), self.MODEL_WEIGHTS)
        config_path = os.path.join(os.path.dirname(__file__), self.MODEL_CONFIG)

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            return 0.0, None, []

        try:
            net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()

            best_confidence = 0.0
            best_class = None
            output_logs = []

            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                class_id = int(detections[0, 0, i, 1])
                class_name = self.VEHICLE_CLASS_IDS.get(class_id, 'background')
                output_logs.append(f"detected: {class_name} ({confidence:.2f})")

                if class_id in self.VEHICLE_CLASS_IDS and confidence > best_confidence:
                    best_confidence = confidence
                    best_class = self.VEHICLE_CLASS_IDS[class_id]

            return best_confidence, best_class, output_logs

        except Exception as e:
            print(f"DNN detection error: {e}")
            return 0.0, None, []

    def _detect_with_cascade(self, image: np.ndarray) -> float:
        """Detect vehicles using Haar cascades."""
        if self.vehicle_cascade is None or self.vehicle_cascade.empty():
            return 0.0

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            vehicles = self.vehicle_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50)
            )

            if len(vehicles) > 0:
                total_area = sum([w * h for (x, y, w, h) in vehicles])
                image_area = image.shape[0] * image.shape[1]
                confidence = min(0.9, (total_area / image_area) * 3.0)
                return confidence
        except Exception as e:
            print(f"Cascade detection error: {e}")

        return 0.0

    def _analyze_vehicle_shape(self, image: np.ndarray) -> float:
        """Analyze image for vehicle-like shapes and colors."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Look for typical vehicle colors (black, white, metallic)
            # Black detection
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 80])
            mask_black = cv2.inRange(hsv, lower_black, upper_black)

            # White detection
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)

            # Metallic/reflective colors
            lower_metallic = np.array([0, 30, 100])
            upper_metallic = np.array([180, 150, 200])
            mask_metallic = cv2.inRange(hsv, lower_metallic, upper_metallic)

            # Calculate color ratios
            total_pixels = image.shape[0] * image.shape[1]
            black_ratio = np.sum(mask_black > 0) / total_pixels
            white_ratio = np.sum(mask_white > 0) / total_pixels
            metallic_ratio = np.sum(mask_metallic > 0) / total_pixels

            # Vehicles typically have significant black/white/metallic content
            color_score = black_ratio * 0.4 + white_ratio * 0.4 + metallic_ratio * 0.2

            # Shape analysis - look for rectangular regions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rectangular_score = 0
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 100:  # Filter small contours
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                    if len(approx) == 4:  # Rectangle
                        rectangular_score += 0.1

            rectangular_score = min(0.5, rectangular_score)

            return min(1.0, color_score + rectangular_score)

        except Exception as e:
            print(f"Shape analysis error: {e}")
            return 0.0

    def _analyze_edges_and_contours(self, image: np.ndarray) -> float:
        """Analyze edges and contours for vehicle-like features."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = np.sum(edges > 0) / (image.shape[0] * image.shape[1])

            # Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Analyze contour properties
            large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
            contour_score = min(0.5, len(large_contours) * 0.1)

            # Aspect ratio analysis (vehicles are often wider than tall)
            if large_contours:
                areas = [cv2.contourArea(c) for c in large_contours]
                max_area_idx = np.argmax(areas)
                x, y, w, h = cv2.boundingRect(large_contours[max_area_idx])

                if h > 0:
                    aspect_ratio = w / h
                    # Vehicles typically have aspect ratio between 1.5 and 4
                    if 1.5 <= aspect_ratio <= 4.0:
                        aspect_score = 0.3
                    else:
                        aspect_score = 0.1
                else:
                    aspect_score = 0.0
            else:
                aspect_score = 0.0

            return min(1.0, edge_ratio * 0.4 + contour_score + aspect_score)

        except Exception as e:
            print(f"Edge analysis error: {e}")
            return 0.0

    def _classify_vehicle_type(self, image: np.ndarray, confidence: float) -> Optional[str]:
        """Classify the type of vehicle detected."""
        try:
            h, w = image.shape[:2]
            aspect_ratio = w / h if h > 0 else 1.0
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour_ratio = 0.0
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                _, _, w_contour, h_contour = cv2.boundingRect(main_contour)
                if h_contour > 0:
                    largest_contour_ratio = w_contour / h_contour

            # Improved classification logic
            if aspect_ratio >= 2.5:  # Very wide - truck/bus
                return "truck"
            elif aspect_ratio <= 1.4:  # Tall or square - likely motorcycle/scooter
                return "motorcycle"
            elif aspect_ratio >= 1.8:  # Wide - car
                return "car"
            else:  # Middle range 1.4-1.8
                if largest_contour_ratio < 1.3:  # Narrower contour - motorcycle
                    return "motorcycle"
                else:  # Wider contour - car
                    return "car"

        except Exception as e:
            print(f"Vehicle classification error: {e}")
            return "unknown"

    def analyze_damage(self, image_path: str) -> Dict:
        """
        Analyze the image for damage location, severity, and characteristics.

        Returns:
            dict with damage analysis results
        """
        try:
            # Debug logging
            print(f"[DEBUG] Analyzing damage for image: {image_path}")

            # Check if file exists
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not load image - unsupported format or corrupted file: {image_path}"}

            print(f"[DEBUG] Image loaded successfully. Shape: {image.shape}")

            # Image preprocessing
            # Resize to standard size for consistent analysis
            target_size = (256, 256)
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            print(f"[DEBUG] Image resized to: {image.shape}")

            # Convert to RGB (OpenCV loads as BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Normalize pixel values (optional, for some models)
            # image_normalized = image_rgb.astype(np.float32) / 255.0

            h, w = image.shape[:2]

            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Damage detection using multiple indicators
            damage_indicators = self._detect_damage_indicators(image, gray, hsv)
            print(f"[DEBUG] Found {len(damage_indicators)} damage indicators")

            # Area mapping first so the broader location label can use it.
            damage_areas = self._map_damage_areas(image, damage_indicators)
            damage_location = self._analyze_damage_location(image, damage_indicators, damage_areas)

            # Severity assessment
            severity_score = self._assess_damage_severity(damage_indicators, image)
            damage_extent = self._estimate_damage_extent(damage_indicators, image)

            # Damage type classification
            damage_type = self._classify_damage_type(damage_indicators)
            severity_level = self._severity_to_level(
                severity_score,
                damage_extent,
                len(damage_areas),
                damage_type
            )
            damage_character = self._classify_damage_character(
                damage_indicators,
                damage_extent,
                severity_level,
                damage_type
            )

            # Calculate confidence
            confidence = self._estimate_damage_confidence(damage_indicators, severity_score, image)
            print(f"[DEBUG] Analysis confidence: {confidence:.3f}")
            damage_detected = self._is_damage_detected(
                damage_type,
                confidence,
                severity_score,
                damage_extent,
                len(damage_indicators),
                len(damage_areas)
            )
            damage_description = self._build_damage_description(
                damage_detected, damage_areas, damage_character, severity_level
            )

            return {
                "damage_detected": damage_detected,
                "damage_location": damage_location,
                "damage_areas": damage_areas,
                "severity_score": severity_score,
                "severity_level": severity_level,
                "damage_type": damage_type,
                "damage_character": damage_character,
                "damage_description": damage_description,
                "damage_extent": damage_extent,
                "confidence": confidence,
                "indicators_found": len(damage_indicators)
            }

        except Exception as e:
            print(f"[ERROR] Damage analysis failed: {e}")
            import traceback
            traceback.print_exc()

            # Return safe fallback results instead of crashing
            return {
                "damage_detected": False,
                "damage_location": "unknown",
                "damage_areas": [],
                "severity_score": 0.1,
                "severity_level": "None",
                "damage_type": "unknown",
                "damage_character": "none",
                "damage_description": "Damage: None",
                "damage_extent": 0.0,
                "confidence": 0.2,
                "indicators_found": 0,
                "fallback": True,
                "error_message": str(e)
            }

    def _detect_damage_indicators(self, image: np.ndarray, gray: np.ndarray, hsv: np.ndarray) -> List[Dict]:
        """Detect various damage indicators in the image."""
        indicators = []

        try:
            h, w = image.shape[:2]

            # 1. Color anomalies (unusual colors on vehicle surface)
            # Look for bright colors that might indicate exposed parts
            bright_mask = cv2.inRange(hsv, np.array([0, 50, 200]), np.array([180, 255, 255]))
            bright_regions = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            for contour in bright_regions:
                area = cv2.contourArea(contour)
                if area > 1000:  # Increased threshold for significant bright region
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    indicators.append({
                        "type": "color_anomaly",
                        "location": "unknown",
                        "severity": "medium",
                        "area": area,
                        "bbox": (x, y, w_box, h_box)
                    })

            # 2. Edge irregularities (dents, creases)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = cv2.filter2D(edges, -1, np.ones((5, 5), np.float32) / 25)

            # Find regions with high edge density (potential damage)
            _, high_edge_mask = cv2.threshold(edge_density, 100, 255, cv2.THRESH_BINARY)
            high_edge_contours = cv2.findContours(high_edge_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            for contour in high_edge_contours:
                area = cv2.contourArea(contour)
                if 2000 < area < 50000:  # Increased min threshold for medium-sized irregular regions
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    indicators.append({
                        "type": "edge_irregularity",
                        "location": "unknown",
                        "severity": "medium",
                        "area": area,
                        "bbox": (x, y, w_box, h_box)
                    })

            # 3. Shadow analysis (deep shadows might indicate dents)
            shadow_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
            shadow_contours = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            for contour in shadow_contours:
                area = cv2.contourArea(contour)
                if area > 4000:  # Increased threshold for significant shadow
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    indicators.append({
                        "type": "shadow_anomaly",
                        "location": "unknown",
                        "severity": "high",
                        "area": area,
                        "bbox": (x, y, w_box, h_box)
                    })

            # 4. Texture analysis (rough surfaces might indicate damage)
            # Simple texture analysis using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)

            if texture_variance > 9000:  # High threshold to avoid flagging normal reflections
                indicators.append({
                    "type": "texture_anomaly",
                    "location": "unknown",
                    "severity": "medium",
                    "variance": texture_variance,
                    "area": h * w * 0.05,
                    "bbox": (0, 0, w, h)
                })

        except Exception as e:
            print(f"Damage indicator detection error: {e}")

        return indicators

    def _map_damage_areas(self, image: np.ndarray, indicators: List[Dict]) -> List[str]:
        """Map detected regions into claim-friendly vehicle area labels."""
        if not indicators:
            return []

        try:
            h, w = image.shape[:2]
            detected_areas = []

            for indicator in indicators:
                bbox = indicator.get("bbox")
                if not bbox:
                    continue
                detected_areas.extend(self._map_bbox_to_damage_areas(bbox, w, h))

            ordered_areas = []
            for area_name in ["front bumper", "hood", "door", "headlight", "side panel"]:
                if area_name in detected_areas and area_name not in ordered_areas:
                    ordered_areas.append(area_name)

            return ordered_areas

        except Exception as e:
            print(f"Damage area mapping error: {e}")
            return []

    def _map_bbox_to_damage_areas(
        self,
        bbox: Tuple[int, int, int, int],
        image_width: int,
        image_height: int
    ) -> List[str]:
        """Convert a bounding box into one or two strongest supported damage areas."""
        x, y, w_box, h_box = bbox
        bbox_rect = (x, y, x + w_box, y + h_box)

        candidate_regions = [
            ("hood", (0.24, 0.08, 0.80, 0.42)),
            ("door", (0.30, 0.36, 0.74, 0.78)),
            ("front bumper", (0.12, 0.66, 0.90, 1.00)),
            ("headlight", (0.00, 0.40, 0.28, 0.78)),
            ("headlight", (0.72, 0.40, 1.00, 0.78)),
            ("side panel", (0.00, 0.30, 0.28, 0.84)),
            ("side panel", (0.72, 0.30, 1.00, 0.84)),
        ]

        label_scores = {}
        for label, region in candidate_regions:
            region_rect = self._denormalize_region(region, image_width, image_height)
            overlap_ratio = self._compute_overlap_ratio(bbox_rect, region_rect)
            if overlap_ratio <= 0:
                continue
            label_scores[label] = max(label_scores.get(label, 0.0), overlap_ratio)

        if not label_scores:
            return []

        ranked_labels = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        top_score = ranked_labels[0][1]
        selected = []

        for label, score in ranked_labels:
            if score < 0.16:
                continue
            if len(selected) >= 2:
                break
            if score >= max(0.24, top_score * 0.7):
                selected.append(label)

        normalized_bottom = (y + h_box) / max(1, image_height)
        normalized_left = x / max(1, image_width)
        normalized_right = (x + w_box) / max(1, image_width)
        touches_side_edge = normalized_left < 0.18 or normalized_right > 0.82

        if "door" in selected and ("front bumper" in selected or "headlight" in selected):
            if normalized_bottom > 0.72 or touches_side_edge:
                selected = [label for label in selected if label != "door"]

        return selected or [ranked_labels[0][0]]

    def _denormalize_region(
        self,
        region: Tuple[float, float, float, float],
        image_width: int,
        image_height: int
    ) -> Tuple[int, int, int, int]:
        """Convert a normalized area definition into image pixel coordinates."""
        x1, y1, x2, y2 = region
        return (
            int(x1 * image_width),
            int(y1 * image_height),
            int(x2 * image_width),
            int(y2 * image_height),
        )

    def _compute_overlap_ratio(
        self,
        source_rect: Tuple[int, int, int, int],
        target_rect: Tuple[int, int, int, int]
    ) -> float:
        """Compute overlap as intersection area over source rectangle area."""
        sx1, sy1, sx2, sy2 = source_rect
        tx1, ty1, tx2, ty2 = target_rect

        inter_x1 = max(sx1, tx1)
        inter_y1 = max(sy1, ty1)
        inter_x2 = min(sx2, tx2)
        inter_y2 = min(sy2, ty2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        source_area = max(1, (sx2 - sx1) * (sy2 - sy1))
        return intersection / source_area

    def _analyze_damage_location(
        self,
        image: np.ndarray,
        indicators: List[Dict],
        damage_areas: Optional[List[str]] = None
    ) -> str:
        """Analyze where the damage is located on the vehicle."""
        if damage_areas:
            area_set = set(damage_areas)
            if area_set.intersection({"front bumper", "headlight", "hood"}):
                if area_set.intersection({"door", "side panel"}):
                    return "multiple_areas"
                return "front"
            if area_set.intersection({"door", "side panel"}):
                return "side"

        if not indicators:
            return "no_damage_detected"

        try:
            h, w = image.shape[:2]
            locations = []

            for indicator in indicators:
                bbox = indicator.get("bbox")
                if not bbox:
                    continue

                x, y, w_box, h_box = bbox
                center_x = x + w_box / 2
                center_y = y + h_box / 2

                if center_x < w * 0.3:
                    locations.append("left")
                elif center_x > w * 0.7:
                    locations.append("right")
                elif center_y < h * 0.4:
                    locations.append("front")
                elif center_y > h * 0.6:
                    locations.append("rear")
                else:
                    locations.append("multiple_areas")

            if not locations:
                return "unknown"

            unique_locations = set(locations)
            if len(unique_locations) > 1:
                return "multiple_areas"
            return locations[0]

        except Exception as e:
            print(f"Location analysis error: {e}")

        return "unknown"

    def _estimate_damage_confidence(self, indicators: List[Dict], severity_score: float, image: np.ndarray) -> float:
        """Estimate confidence from damage evidence and severity."""
        if not indicators:
            return 0.0

        try:
            h, w = image.shape[:2]
            image_area = h * w
            total_area = sum(ind.get("area", 0) for ind in indicators)
            area_ratio = min(1.0, total_area / max(1, image_area))
            indicator_count = len(indicators)
            indicator_score = min(0.4, indicator_count * 0.08)

            confidence = 0.2 + severity_score * 0.35 + area_ratio * 0.35 + indicator_score
            return min(0.95, confidence)

        except Exception as e:
            print(f"Confidence estimation error: {e}")
            return 0.0

    def _estimate_damage_extent(self, indicators: List[Dict], image: np.ndarray) -> float:
        """Estimate how much of the visible vehicle surface appears affected."""
        if not indicators:
            return 0.0

        h, w = image.shape[:2]
        image_area = max(1, h * w)
        total_indicator_area = sum(indicator.get("area", 0) for indicator in indicators)
        return min(1.0, total_indicator_area / image_area)

    def _assess_damage_severity(self, indicators: List[Dict], image: np.ndarray) -> float:
        """Assess the overall severity of damage."""
        if not indicators:
            return 0.0

        try:
            # Calculate severity based on indicators
            severity_weights = {
                "color_anomaly": 0.3,
                "edge_irregularity": 0.4,
                "shadow_anomaly": 0.6,
                "texture_anomaly": 0.2
            }

            total_severity = 0
            max_possible = 0

            for indicator in indicators:
                indicator_type = indicator.get("type", "")
                weight = severity_weights.get(indicator_type, 0.1)
                total_severity += weight
                max_possible += 0.6  # Maximum weight

            # Normalize to 0-1 scale
            severity_score = min(1.0, total_severity / max(1, len(indicators)))

            return severity_score

        except Exception as e:
            print(f"Severity assessment error: {e}")
            return 0.0

    def _severity_to_level(
        self,
        severity_score: float,
        damage_extent: float,
        area_count: int,
        damage_type: str
    ) -> str:
        """Convert severity score to descriptive level."""
        if severity_score < 0.12 and damage_extent < 0.03:
            return "None"
        if damage_type == "paint_damage":
            if damage_extent < 0.06:
                return "Minor"
            return "Moderate"
        if severity_score >= 0.72 or damage_extent >= 0.45 or area_count >= 4:
            return "Total Loss"
        if severity_score >= 0.45 or damage_extent >= 0.18 or area_count >= 3:
            return "Severe"
        if severity_score >= 0.22 or damage_extent >= 0.08 or area_count >= 2:
            return "Moderate"
        return "Minor"

    def _is_damage_detected(
        self,
        damage_type: str,
        confidence: float,
        severity_score: float,
        damage_extent: float,
        indicator_count: int,
        area_count: int
    ) -> bool:
        """Apply conservative rules so clean vehicle images are not over-classified as damaged."""
        if indicator_count == 0 or area_count == 0:
            return False

        if damage_type == "paint_damage":
            return (
                indicator_count >= 2
                and confidence >= 0.55
                and 0.03 <= damage_extent <= 0.24
            )

        if damage_type == "surface_damage":
            return (
                indicator_count >= 2
                and confidence >= 0.65
                and 0.04 <= damage_extent <= 0.14
            )

        return (
            confidence >= 0.45
            and indicator_count >= 1
            and (severity_score >= 0.25 or damage_extent >= 0.08)
        )

    def _classify_damage_type(self, indicators: List[Dict]) -> str:
        """Classify the type of damage detected."""
        if not indicators:
            return "no_damage"

        # Weight indicator types by area and severity so real crush damage
        # is not outvoted by multiple cosmetic regions.
        type_scores = {}
        severity_weights = {"medium": 1.0, "high": 1.5}
        for indicator in indicators:
            indicator_type = indicator.get("type", "unknown")
            area_weight = max(1.0, float(indicator.get("area", 0)) / 1500.0)
            severity_weight = severity_weights.get(indicator.get("severity", "medium"), 1.0)
            type_scores[indicator_type] = type_scores.get(indicator_type, 0.0) + (area_weight * severity_weight)

        # Determine primary damage type
        primary_type = max(type_scores, key=type_scores.get)

        type_mapping = {
            "color_anomaly": "paint_damage",
            "edge_irregularity": "structural_damage",
            "shadow_anomaly": "dent_damage",
            "texture_anomaly": "surface_damage"
        }

        return type_mapping.get(primary_type, "unknown_damage")

    def _classify_damage_character(
        self,
        indicators: List[Dict],
        damage_extent: float,
        severity_level: str,
        damage_type: str
    ) -> str:
        """Translate low-level indicators into claim-friendly damage wording."""
        if not indicators or severity_level == "None":
            return "none"

        if severity_level == "Total Loss" or damage_extent >= 0.45:
            return "heavily damaged"
        if damage_type == "dent_damage":
            return "dented"
        if damage_type == "structural_damage":
            return "broken"
        if damage_type == "paint_damage":
            return "scratched"
        return "damaged"

    def _build_damage_description(
        self,
        damage_detected: bool,
        damage_areas: List[str],
        damage_character: str,
        severity_level: str
    ) -> str:
        """Create a human-readable description for the final claim result."""
        if not damage_detected:
            return "Damage: None"

        if not damage_areas:
            return f"Visible {severity_level.lower()} damage detected."

        described_areas = []
        for area in damage_areas[:3]:
            described_areas.append(f"{area} {damage_character}")

        return ", ".join(described_areas).capitalize()

    def detect_fraud_indicators(
        self,
        image_path: str,
        description: str,
        damage_analysis: Optional[Dict] = None,
        estimated_amount: Optional[float] = None
    ) -> Dict:
        """
        Detect potential fraud by comparing description with image analysis.

        Returns:
            dict with fraud analysis results
        """
        try:
            # Analyze the image
            damage_analysis = damage_analysis or self.analyze_damage(image_path)

            if "error" in damage_analysis:
                return {"error": damage_analysis["error"]}

            # Analyze description
            desc_lower = description.lower()
            fraud_indicators = []

            # Check for damage keywords
            damage_mentioned = any(keyword in desc_lower for keyword in
                                 ['damage', 'accident', 'collision', 'crash', 'hit', 'dent', 'scratch'])

            # Check location consistency
            location_consistent = self._check_location_consistency(desc_lower, damage_analysis)

            # Check severity consistency
            severity_consistent = self._check_severity_consistency(desc_lower, damage_analysis)

            # Calculate fraud score
            fraud_score = 0.0

            if damage_mentioned and not damage_analysis["damage_detected"]:
                fraud_score += 0.4  # Description claims damage but image shows none
                fraud_indicators.append("Description mentions damage but the image does not show visible damage.")

            if not location_consistent:
                fraud_score += 0.3  # Location mismatch
                fraud_indicators.append("Damage location in the description does not match the visible damage area.")

            if not severity_consistent:
                fraud_score += 0.2  # Severity mismatch
                fraud_indicators.append("Claimed damage severity looks different from the visible damage level.")

            # Additional fraud indicators
            confidence = damage_analysis.get("confidence", 0.0)
            indicators_found = damage_analysis.get("indicators_found", 0)

            if damage_analysis["damage_detected"] and confidence < 0.5:
                fraud_score += 0.3  # Low confidence damage detection suggests potential fraud
                fraud_indicators.append("Visible damage confidence is low, so the claim needs manual review.")

            if indicators_found < 3 and damage_analysis["damage_detected"]:
                fraud_score += 0.2  # Few indicators but damage detected
                fraud_indicators.append("The visible damage evidence is limited compared to the claim.")

            # Length-based fraud detection
            if len(description.strip()) < 20:
                fraud_score += 0.2  # Suspiciously short description
                fraud_indicators.append("Description is unusually short for a damage claim.")

            # Suspicious keywords
            suspicious_words = ['fake', 'fraud', 'scam', 'test', 'joke']
            if any(word in desc_lower for word in suspicious_words):
                fraud_score += 0.5
                fraud_indicators.append("Description contains suspicious keywords.")

            claimed_amount = self._extract_claim_amount(description)
            if (
                claimed_amount
                and estimated_amount
                and estimated_amount > 0
                and claimed_amount > estimated_amount * 1.75
            ):
                fraud_score += 0.3
                fraud_indicators.append("Claimed amount looks too high for the visible damage.")

            fraud_score = min(1.0, fraud_score)

            return {
                "fraud_score": fraud_score,
                "fraud_level": "high" if fraud_score > 0.7 else "medium" if fraud_score > 0.4 else "low",
                "damage_mentioned_in_desc": damage_mentioned,
                "damage_detected_in_image": damage_analysis["damage_detected"],
                "location_consistent": location_consistent,
                "severity_consistent": severity_consistent,
                "description_length": len(description.strip()),
                "claimed_amount": claimed_amount,
                "indicators": fraud_indicators or ["No major fraud indicators detected."]
            }

        except Exception as e:
            print(f"Fraud detection error: {e}")
            return {"error": str(e)}

    def _check_location_consistency(self, description: str, damage_analysis: Dict) -> bool:
        """Check if description location matches damage analysis."""
        # This is a simplified check - in reality, you'd do more sophisticated NLP
        desc_lower = description.lower()
        damage_location = damage_analysis.get("damage_location", "")
        damage_areas = " ".join(damage_analysis.get("damage_areas", []))

        # Simple keyword matching
        location_keywords = {
            "front": ["front", "hood", "bumper", "headlight"],
            "rear": ["rear", "back", "trunk", "tail"],
            "left": ["left", "driver"],
            "right": ["right", "passenger"],
            "roof": ["roof", "top"]
        }

        for location, keywords in location_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                return (
                    damage_location == location
                    or damage_location == "multiple_areas"
                    or any(keyword in damage_areas for keyword in keywords)
                )

        return True  # If no specific location mentioned, assume consistent

    def _check_severity_consistency(self, description: str, damage_analysis: Dict) -> bool:
        """Check if description severity matches damage analysis."""
        desc_lower = description.lower()
        severity_level = damage_analysis.get("severity_level", "Minor")

        # Check for severity keywords
        mild_keywords = ['minor', 'small', 'slight', 'tiny', 'little']
        moderate_keywords = ['moderate', 'medium', 'some', 'fair']
        severe_keywords = ['severe', 'major', 'extensive', 'heavy', 'bad', 'terrible', 'total loss']

        if any(keyword in desc_lower for keyword in severe_keywords):
            return severity_level in ['Severe', 'Total Loss']
        elif any(keyword in desc_lower for keyword in moderate_keywords):
            return severity_level in ['Moderate', 'Severe', 'Total Loss']
        elif any(keyword in desc_lower for keyword in mild_keywords):
            return severity_level in ['Minor', 'Moderate']

        return True  # If no severity mentioned, assume consistent

    def _extract_claim_amount(self, description: str) -> Optional[float]:
        """Extract a rough claimed amount from the description text if present."""
        amount_patterns = [
            r'(?:rs\.?|inr|rupees?)\s*([0-9][0-9,]*)',
            r'([0-9][0-9,]*)\s*(?:rs\.?|inr|rupees?)'
        ]

        for pattern in amount_patterns:
            match = re.search(pattern, description.lower())
            if not match:
                continue

            try:
                amount_text = match.group(1).replace(",", "")
                return float(amount_text)
            except ValueError:
                continue

        return None


# Global instance for use in the main function
detector = AdvancedVehicleDamageDetector()

def is_vehicle_image(image_path, confidence_threshold=0.4):
    """
    Check if the uploaded image contains a vehicle.

    Returns:
        tuple: (is_vehicle, confidence, vehicle_type)
    """
    return detector.detect_vehicle(image_path, threshold=confidence_threshold)

def analyze_damage(image_path):
    """
    Analyze damage in the vehicle image.

    Returns:
        dict with damage analysis
    """
    return detector.analyze_damage(image_path)

def detect_fraud_indicators(image_path, description, damage_analysis=None, estimated_amount=None):
    """
    Detect fraud indicators by comparing image and description.

    Returns:
        dict with fraud analysis
    """
    return detector.detect_fraud_indicators(
        image_path,
        description,
        damage_analysis=damage_analysis,
        estimated_amount=estimated_amount
    )
