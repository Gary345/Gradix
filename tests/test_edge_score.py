from src.scoring.condition_score import compute_edge_score


def main():
    edge_features = {
        "edge_score": 82.5,
        "edge_confidence": 0.78,
        "top_edge_score": 85.0,
        "bottom_edge_score": 79.0,
        "left_edge_score": 83.0,
        "right_edge_score": 81.0,
    }

    result = compute_edge_score(edge_features)

    print("=== TEST EDGE SCORE ===")
    print(result)


if __name__ == "__main__":
    main()