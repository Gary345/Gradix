import cv2

from src.features.corner_features import (
    compute_corner_features,
    draw_corner_patch_overlay,
)


def main():
    image_path = "tests/assets/sample_rectified_card.jpg"
    output_overlay_path = "tests/assets/sample_rectified_card_corners_overlay.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    features = compute_corner_features(image)

    print("=== CORNER FEATURES ===")
    for key, value in features.items():
        print(f"{key}: {value}")

    overlay = draw_corner_patch_overlay(image)
    cv2.imwrite(output_overlay_path, overlay)

    print(f"\nOverlay guardado en: {output_overlay_path}")


if __name__ == "__main__":
    main()