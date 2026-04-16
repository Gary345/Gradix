from src.scoring.condition_score import (
    compute_edge_score,
    compute_gradix_condition_stub_v2,
)


def main():
    edge_features = {
        "edge_score": 82.5,
        "edge_confidence": 0.78,
        "top_edge_score": 85.0,
        "bottom_edge_score": 79.0,
        "left_edge_score": 83.0,
        "right_edge_score": 81.0,
    }

    edge_result = compute_edge_score(edge_features)

    condition_result = compute_gradix_condition_stub_v2(
        preliminary_gradix_score=8.2,
        centering_score=7.6,
        gradix_edge_score=edge_result["gradix_edge_score"],
        edge_confidence=edge_features["edge_confidence"],
    )

    print("=== TEST EDGE RESULT ===")
    print(edge_result)

    print("\n=== TEST CONDITION V2 RESULT ===")
    print(condition_result)


if __name__ == "__main__":
    main()