from src.business_entity_resolution.evaluate import (
    calculate_precision,
    calculate_recall,
    calculate_f05,
)


def test_precision():
    true_matches = {
        "S2-100",
        "S2-200",
        "S3-300"
    }

    predicted_matches = {
        "S2-100",
        "S2-200",
        "S2-500"
    }

    precision = calculate_precision(
        true_matches,
        predicted_matches
    )

    assert precision == 2 / 3


def test_recall():
    true_matches = {
        "S2-100",
        "S2-200",
        "S3-300"
    }

    predicted_matches = {
        "S2-100",
        "S2-200"
    }

    recall = calculate_recall(
        true_matches,
        predicted_matches
    )

    assert recall == 2 / 3


def test_f05():
    precision = 0.8
    recall = 0.6

    result = calculate_f05(
        precision,
        recall
    )

    expected = (
        1.25 * precision * recall
    ) / (
        0.25 * precision + recall
    )

    assert result == expected