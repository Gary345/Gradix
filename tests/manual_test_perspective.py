import cv2

from src.vision.card_detector import detect_card_contour
from src.vision.perspective import warp_card_perspective

image = cv2.imread("tests/assets/sample_card_photo.jpg")

det = detect_card_contour(image)
corners = det["corners"]

result = warp_card_perspective(image, corners)

print("success:", result["success"])
print("output_size:", result["output_size"])
print("metrics:")
for k, v in result["metrics"].items():
    print(f"  {k}: {v}")

cv2.imwrite("tests/assets/debug_warped_card.jpg", result["warped_image"])
print("saved: tests/assets/debug_warped_card.jpg")