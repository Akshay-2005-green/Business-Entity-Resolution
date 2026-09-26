import pandas as pd

from src.business_entity_resolution.pipeline import (
    run_pipeline,
    create_matching_results,
)


def mock_preprocess(
    source1,
    source2,
    source3,
):
    return {
        "source1": source1,
        "source2": source2,
        "source3": source3,
    }


def mock_candidate_generation(
    source1,
    source2,
    source3,
):
    return pd.DataFrame({
        "source1_entity_id": [
            "S1-100",
            "S1-100",
            "S1-101",
        ],
        "candidate_entity_id": [
            "S2-200",
            "S3-300",
            "S2-400",
        ],
    })


def mock_prediction(
    candidate_pairs,
    source1,
    source2,
    source3,
):
    result = candidate_pairs.copy()

    result["score"] = [
        0.95,
        0.90,
        0.30,
    ]

    return result


def test_run_pipeline():

    source1 = pd.DataFrame({
        "entity_id": [
            "S1-100",
            "S1-101",
        ]
    })

    source2 = pd.DataFrame({
        "entity_id": [
            "S2-200",
            "S2-400",
        ]
    })

    source3 = pd.DataFrame({
        "entity_id": [
            "S3-300",
        ]
    })

    result = run_pipeline(
        source1,
        source2,
        source3,
        mock_preprocess,
        mock_candidate_generation,
        mock_prediction,
        threshold=0.80,
    )

    assert result["matches"]["S1-100"] == {
        "S2-200",
        "S3-300",
    }

    assert "S1-101" not in result["matches"]


def test_create_matching_results():

    matches = {
        "S1-100": {
            "S2-200",
            "S3-300",
        }
    }

    source1_ids = [
        "S1-100",
        "S1-101",
    ]

    result = create_matching_results(
        matches,
        source1_ids,
    )

    assert len(result) == 2

    assert result.iloc[0]["source1_entity_id"] == "S1-100"

    assert result.iloc[0]["matched_entity_ids"] == (
        "S2-200,S3-300"
    )

    assert result.iloc[1]["source1_entity_id"] == "S1-101"

    assert result.iloc[1]["matched_entity_ids"] == ""

def test_validate_output_consistency():

    from src.business_entity_resolution.pipeline import (
        validate_output_consistency,
    )

    candidate_pairs = pd.DataFrame({
        "source1_entity_id": [
            "S1-100",
            "S1-100",
            "S1-101",
        ],
        "candidate_entity_id": [
            "S2-200",
            "S3-300",
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

    assert validate_output_consistency(
        matching_results,
        candidate_pairs,
    ) is True

def test_validate_output_consistency_rejects_invalid_match():

    from src.business_entity_resolution.pipeline import (
        validate_output_consistency,
    )

    candidate_pairs = pd.DataFrame({
        "source1_entity_id": [
            "S1-100",
        ],
        "candidate_entity_id": [
            "S2-200",
        ],
    })

    matching_results = pd.DataFrame({
        "source1_entity_id": [
            "S1-100",
        ],
        "matched_entity_ids": [
            "S2-999",
        ],
    })

    try:
        validate_output_consistency(
            matching_results,
            candidate_pairs,
        )

        assert False

    except ValueError:
        assert True

def test_end_to_end_output_generation(tmp_path):

    from src.business_entity_resolution.pipeline import (
        save_matching_results,
        save_candidate_pairs,
        validate_output_consistency,
    )

    # -----------------------------------------
    # Mock candidate pairs
    # -----------------------------------------

    candidate_pairs = pd.DataFrame({
        "source1_entity_id": [
            "S1-100",
            "S1-100",
            "S1-101",
        ],
        "candidate_entity_id": [
            "S2-200",
            "S3-300",
            "S2-400",
        ],
    })

    # -----------------------------------------
    # Mock accepted matches
    # -----------------------------------------

    matches = {
        "S1-100": {
            "S2-200",
            "S3-300",
        }
    }

    source1_ids = [
        "S1-100",
        "S1-101",
    ]

    # -----------------------------------------
    # Create output directory
    # -----------------------------------------

    output_dir = tmp_path / "output"

    matching_path = (
        output_dir / "matching_results.tsv"
    )

    candidate_path = (
        output_dir / "candidate_pairs.tsv"
    )

    # -----------------------------------------
    # Save both required files
    # -----------------------------------------

    matching_results = save_matching_results(
        matches,
        source1_ids,
        matching_path,
    )

    save_candidate_pairs(
    candidate_pairs,
    source1_ids,
    candidate_path,
)

    # -----------------------------------------
    # Check files exist
    # -----------------------------------------

    assert matching_path.exists()
    assert candidate_path.exists()

    # -----------------------------------------
    # Check required columns
    # -----------------------------------------

    assert list(matching_results.columns) == [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    # -----------------------------------------
    # Validate final matches
    # -----------------------------------------

    assert validate_output_consistency(
        matching_results,
        candidate_pairs,
    ) is True

    # -----------------------------------------
    # Check Source 1 coverage
    # -----------------------------------------

    assert len(matching_results) == 2

    assert set(
        matching_results["source1_entity_id"]
    ) == {
        "S1-100",
        "S1-101",
    }