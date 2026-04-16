from src.scoring.condition_score import (
    compute_corner_score,
    compute_gradix_condition_stub_v3,
)


def main():
    corner_features = {
        "corner_score_raw": 86.53,
        "corner_confidence": 0.438,
        "top_left_corner_score": 87.74,
        "top_right_corner_score": 76.57,
        "bottom_left_corner_score": 100.0,
        "bottom_right_corner_score": 81.82,
    }

    corner_result = compute_corner_score(corner_features)

    print("=== TEST CORNER SCORE ===")
    print(corner_result)

    condition_result = compute_gradix_condition_stub_v3(
        preliminary_gradix_score=8.2,
        centering_score=7.6,
        gradix_edge_score=8.0,
        gradix_corner_score=corner_result["gradix_corner_score"],
        edge_confidence=0.78,
        corner_confidence=corner_features["corner_confidence"],
    )

    print("\n=== TEST CONDITION V3 ===")
    print(condition_result)


if __name__ == "__main__":
    main()