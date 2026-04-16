import cv2
from src.features.whitening_surface_features import compute_whitening_surface_features

image = cv2.imread("tests/assets/sample_rectified_card.jpg")
result = compute_whitening_surface_features(image)

for k, v in result.items():
    print(f"{k}: {v}")