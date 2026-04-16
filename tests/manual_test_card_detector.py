import cv2
from src.vision.card_detector import detect_card_contour

image = cv2.imread("tests/assets/sample_card_photo.jpg")
result = detect_card_contour(image)

print("success:", result["success"])
print("used_fallback:", result["used_fallback"])
print("metrics:")
for k, v in result["metrics"].items():
    print(f"  {k}: {v}")

debug = result["debug_images"]["detected_contour"]
cv2.imwrite("tests/assets/debug_detected_contour.jpg", debug)
print("saved: tests/assets/debug_detected_contour.jpg")