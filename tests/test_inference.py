import pandas as pd

from src.business_entity_resolution.inference import (
    validate_prediction_columns,
    apply_threshold,
    build_match_dictionary,
    generate_predictions,
)
def test_tune_threshold():
    predictions = create_test_predictions()

    ground_truth = {
        "S1-100": {
            "S2-200",
            "S3-300",
        },
        "S1-101": {
            "S2-400",
        },
    }

    from src.business_entity_resolution.inference import (
        tune_threshold,
    )

    results = tune_threshold(
        predictions,
        ground_truth,
        thresholds=[
            0.50,
            0.80,
            0.95,
        ],
    )

    assert len(results) == 3

    assert list(results["threshold"]) == [
        0.50,
        0.80,
        0.95,
    ]

    assert "precision" in results.columns
    assert "recall" in results.columns
    assert "f05" in results.columns


def create_test_predictions():
    return pd.DataFrame({
        "source1_entity_id": [
            "S1-100",
            "S1-100",
            "S1-100",
            "S1-101",
        ],
        "candidate_entity_id": [
            "S2-200",
            "S3-300",
            "S2-999",
            "S2-400",
        ],
        "score": [
            0.95,
            0.91,
            0.20,
            0.85,
        ],
    })


def test_validate_prediction_columns():
    predictions = create_test_predictions()

    validate_prediction_columns(
        predictions
    )


def test_threshold():
    predictions = create_test_predictions()

    result = apply_threshold(
        predictions,
        threshold=0.80
    )

    assert len(result) == 3

    assert "S2-999" not in (
        result["candidate_entity_id"].tolist()
    )


def test_build_match_dictionary():
    predictions = create_test_predictions()

    result = build_match_dictionary(
        predictions
    )

    assert result["S1-100"] == {
        "S2-200",
        "S3-300",
        "S2-999",
    }

    assert result["S1-101"] == {
        "S2-400"
    }


def test_generate_predictions():
    predictions = create_test_predictions()

    result = generate_predictions(
        predictions,
        threshold=0.80
    )

    assert result["S1-100"] == {
        "S2-200",
        "S3-300",
    }

    assert result["S1-101"] == {
        "S2-400"
    }


def test_invalid_threshold():
    predictions = create_test_predictions()

    try:
        apply_threshold(
            predictions,
            threshold=1.5
        )

        assert False

    except ValueError:
        assert True

def test_tune_threshold():
    predictions = create_test_predictions()

    ground_truth = {
        "S1-100": {
            "S2-200",
            "S3-300",
        },
        "S1-101": {
            "S2-400",
        },
    }

    from src.business_entity_resolution.inference import (
        tune_threshold,
    )

    results = tune_threshold(
        predictions,
        ground_truth,
        thresholds=[
            0.50,
            0.80,
            0.95,
        ],
    )

    assert len(results) == 3

    assert list(results["threshold"]) == [
        0.50,
        0.80,
        0.95,
    ]

    assert "precision" in results.columns
    assert "recall" in results.columns
    assert "f05" in results.columns