#!/usr/bin/env python3
"""
Test script for vehicle detection functionality.
"""
import os
import sys

sys.path.append('.')

from models.vehicle_detector import is_vehicle_image


def test_vehicle_detection():
    """Test the vehicle detection with a simple non-vehicle image."""
    print("Testing Vehicle Detection System")
    print("=" * 40)

    test_image_path = "static/uploads/test_image.jpg"

    try:
        import cv2
        import numpy as np

        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:] = (100, 150, 200)
        cv2.rectangle(img, (50, 100), (350, 200), (255, 255, 255), -1)

        os.makedirs("static/uploads", exist_ok=True)
        cv2.imwrite(test_image_path, img)
        print(f"Created test image: {test_image_path}")

    except ImportError:
        print("OpenCV not available for creating test image")
        return

    is_vehicle, confidence, detected_class = is_vehicle_image(test_image_path)

    print(f"Test Image: {test_image_path}")
    print(f"Is Vehicle: {is_vehicle}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Detected Class: {detected_class}")

    if not is_vehicle:
        print("PASS: Non-vehicle image correctly rejected")
    else:
        print("FAIL: Non-vehicle image incorrectly accepted")

    if os.path.exists(test_image_path):
        try:
            os.remove(test_image_path)
            print(f"Cleaned up test image: {test_image_path}")
        except PermissionError:
            print(f"Cleanup skipped because the file is still in use: {test_image_path}")


if __name__ == "__main__":
    test_vehicle_detection()
