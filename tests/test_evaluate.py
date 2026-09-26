from src.business_entity_resolution.evaluate import (
    parse_match_ids,
    calculate_precision,
    calculate_recall,
    calculate_f05,
    evaluate_entity,
    evaluate_all_entities,
    load_ground_truth,
)

def test_parse_match_ids():
    value = "S2-100,S2-200,S3-300"

    result = parse_match_ids(value)

    assert result == {
        "S2-100",
        "S2-200",
        "S3-300"
    }


def test_parse_empty_match_ids():
    assert parse_match_ids("") == set()
    assert parse_match_ids(None) == set()


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


def test_correct_empty_match():
    true_matches = set()
    predicted_matches = set()

    result = evaluate_entity(
        true_matches,
        predicted_matches
    )

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f05"] == 1.0


def test_false_match_when_no_true_match():
    true_matches = set()

    predicted_matches = {
        "S2-100"
    }

    result = evaluate_entity(
        true_matches,
        predicted_matches
    )

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f05"] == 0.0

def test_evaluate_all_entities():
    ground_truth = {
        "S1-100": {
            "S2-200",
            "S3-300"
        },
        "S1-101": {
            "S2-400"
        },
        "S1-102": set()
    }

    predictions = {
        "S1-100": {
            "S2-200",
            "S3-300"
        },
        "S1-101": {
            "S2-999"
        },
        "S1-102": set()
    }

    result = evaluate_all_entities(
        ground_truth,
        predictions
    )

    assert result["precision"] == (
        1.0 + 0.0 + 1.0
    ) / 3

    assert result["recall"] == (
        1.0 + 0.0 + 1.0
    ) / 3

    assert result["f05"] == (
        1.0 + 0.0 + 1.0
    ) / 3

def test_load_ground_truth(tmp_path):
    ground_truth_file = tmp_path / "ground_truth.tsv"

    ground_truth_file.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-100\tS2-200,S3-300\n"
        "S1-101\tS2-400\n"
        "S1-102\t\n",
        encoding="utf-8"
    )

    result = load_ground_truth(
        ground_truth_file
    )

    assert result["S1-100"] == {
        "S2-200",
        "S3-300"
    }

    assert result["S1-101"] == {
        "S2-400"
    }

    assert result["S1-102"] == set()