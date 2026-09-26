import pandas as pd
import pytest

from src.business_entity_resolution.submission import (
    validate_submission_outputs,
)


def valid_data():

    source1_ids = [
        "S1-100",
        "S1-101",
    ]

    candidate_pairs = pd.DataFrame({
        "source1_entity_id": [
            "S1-100",
            "S1-101",
        ],
        "candidate_entity_ids": [
            "S2-200,S3-300",
            "S2-400",
        ],
    })

    matching_results = pd.DataFrame({
        "source1_entity_id": [
            "S1-100",
            "S1-101",
        ],
        "matched_entity_ids": [
            "S2-200,S3-300",
            "",
        ],
    })

    return (
        matching_results,
        candidate_pairs,
        source1_ids,
    )


def test_valid_submission():

    (
        matching_results,
        candidate_pairs,
        source1_ids,
    ) = valid_data()

    assert validate_submission_outputs(
        matching_results,
        candidate_pairs,
        source1_ids,
    ) is True


def test_duplicate_source1_rejected():

    (
        matching_results,
        candidate_pairs,
        source1_ids,
    ) = valid_data()

    matching_results.loc[
        1,
        "source1_entity_id"
    ] = "S1-100"

    with pytest.raises(ValueError):
        validate_submission_outputs(
            matching_results,
            candidate_pairs,
            source1_ids,
        )


def test_missing_source1_rejected():

    (
        matching_results,
        candidate_pairs,
        source1_ids,
    ) = valid_data()

    matching_results = matching_results.iloc[:1]

    with pytest.raises(ValueError):
        validate_submission_outputs(
            matching_results,
            candidate_pairs,
            source1_ids,
        )


def test_invalid_entity_id_rejected():

    (
        matching_results,
        candidate_pairs,
        source1_ids,
    ) = valid_data()

    matching_results.loc[
        0,
        "matched_entity_ids"
    ] = "S4-999"

    with pytest.raises(ValueError):
        validate_submission_outputs(
            matching_results,
            candidate_pairs,
            source1_ids,
        )


def test_non_candidate_match_rejected():

    (
        matching_results,
        candidate_pairs,
        source1_ids,
    ) = valid_data()

    matching_results.loc[
        0,
        "matched_entity_ids"
    ] = "S2-999"

    with pytest.raises(ValueError):
        validate_submission_outputs(
            matching_results,
            candidate_pairs,
            source1_ids,
        )


def test_duplicate_candidate_rejected():

    (
        matching_results,
        candidate_pairs,
        source1_ids,
    ) = valid_data()

    candidate_pairs.loc[
        0,
        "candidate_entity_ids"
    ] = "S2-200,S2-200"

    with pytest.raises(ValueError):
        validate_submission_outputs(
            matching_results,
            candidate_pairs,
            source1_ids,
        )


def test_empty_match_allowed():

    (
        matching_results,
        candidate_pairs,
        source1_ids,
    ) = valid_data()

    matching_results.loc[
        0,
        "matched_entity_ids"
    ] = ""

    assert validate_submission_outputs(
        matching_results,
        candidate_pairs,
        source1_ids,
    ) is True