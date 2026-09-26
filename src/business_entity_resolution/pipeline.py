from pathlib import Path

import pandas as pd

from .inference import generate_predictions
from .evaluate import parse_match_ids

def run_pipeline(
    source1_df,
    source2_df,
    source3_df,
    preprocess_fn,
    candidate_fn,
    predict_fn,
    threshold=0.80,
):
    """
    Run the complete business entity resolution pipeline.

    Parameters
    ----------
    source1_df : pandas.DataFrame
        Source 1 records.

    source2_df : pandas.DataFrame
        Source 2 records.

    source3_df : pandas.DataFrame
        Source 3 records.

    preprocess_fn : callable
        Function supplied by Member 1.

    candidate_fn : callable
        Function supplied by Member 2.

    predict_fn : callable
        Function supplied by Member 3.

    threshold : float
        Minimum model score required to accept a match.

    Returns
    -------
    dict
        {
            "candidate_pairs": ...,
            "predictions": ...,
            "matches": ...
        }
    """

    # ---------------------------------------------------------
    # 1. PREPROCESSING
    # ---------------------------------------------------------

    preprocessed = preprocess_fn(
        source1_df,
        source2_df,
        source3_df,
    )

    source1_processed = preprocessed["source1"]
    source2_processed = preprocessed["source2"]
    source3_processed = preprocessed["source3"]

    # ---------------------------------------------------------
    # 2. CANDIDATE GENERATION
    # ---------------------------------------------------------

    candidate_pairs = candidate_fn(
        source1_processed,
        source2_processed,
        source3_processed,
    )

    # ---------------------------------------------------------
    # 3. MODEL PREDICTION
    # ---------------------------------------------------------

    predictions = predict_fn(
        candidate_pairs,
        source1_processed,
        source2_processed,
        source3_processed,
    )

    # ---------------------------------------------------------
    # 4. INFERENCE / THRESHOLDING
    # ---------------------------------------------------------

    matches = generate_predictions(
        predictions,
        threshold=threshold,
    )

    return {
        "candidate_pairs": candidate_pairs,
        "predictions": predictions,
        "matches": matches,
    }


def create_matching_results(matches, source1_ids):
    """
    Create the final matching-results DataFrame.

    Every Source 1 entity receives exactly one row.

    Entities with no accepted matches receive an empty
    matched_entity_ids value.
    """

    rows = []

    for source1_id in source1_ids:

        matched_ids = matches.get(
            source1_id,
            set(),
        )

        # Sort for deterministic output.
        matched_ids = sorted(matched_ids)

        rows.append({
            "source1_entity_id": source1_id,
            "matched_entity_ids": ",".join(matched_ids),
        })

    return pd.DataFrame(rows)


def save_matching_results(
    matches,
    source1_ids,
    output_path,
):
    """
    Save matching results as a TSV file.
    """

    results = create_matching_results(
        matches,
        source1_ids,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    return results


def create_candidate_results(candidate_pairs, source1_ids):
    """
    Convert internal candidate pairs into the official
    challenge candidate_pairs.tsv format.

    Internal format:
        source1_entity_id
        candidate_entity_id

    Output format:
        source1_entity_id
        candidate_entity_ids
    """

    candidate_lookup = {}

    # Build candidate ID set for every Source 1 entity.
    for _, row in candidate_pairs.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        candidate_id = str(
            row["candidate_entity_id"]
        ).strip()

        if not source1_id or not candidate_id:
            continue

        candidate_lookup.setdefault(
            source1_id,
            set(),
        ).add(candidate_id)

    rows = []

    # IMPORTANT:
    # Every Source 1 entity must have a row.
    for source1_id in source1_ids:

        candidate_ids = sorted(
            candidate_lookup.get(
                source1_id,
                set(),
            )
        )

        rows.append({
            "source1_entity_id": source1_id,
            "candidate_entity_ids": ",".join(
                candidate_ids
            ),
        })

    return pd.DataFrame(rows)


def save_candidate_pairs(
    candidate_pairs,
    source1_ids,
    output_path,
):
    """
    Save candidate pairs in the official challenge
    submission format.
    """

    results = create_candidate_results(
        candidate_pairs,
        source1_ids,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    return results

def validate_output_consistency(
    matching_results,
    candidate_pairs,
):
    """
    Validate that every final match exists in the
    generated candidate pairs.

    Parameters
    ----------
    matching_results : pandas.DataFrame
        Columns:
            source1_entity_id
            matched_entity_ids

    candidate_pairs : pandas.DataFrame
        Columns:
            source1_entity_id
            candidate_entity_id

    Returns
    -------
    bool
        True when all final matches are valid.

    Raises
    ------
    ValueError
        If a final match does not exist in candidate pairs.
    """

    required_matching_columns = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    required_candidate_columns = {
        "source1_entity_id",
        "candidate_entity_id",
    }

    missing_matching = (
        required_matching_columns
        - set(matching_results.columns)
    )

    missing_candidates = (
        required_candidate_columns
        - set(candidate_pairs.columns)
    )

    if missing_matching:
        raise ValueError(
            f"Missing matching result columns: "
            f"{sorted(missing_matching)}"
        )

    if missing_candidates:
        raise ValueError(
            f"Missing candidate pair columns: "
            f"{sorted(missing_candidates)}"
        )

    # Build:
    #
    # S1-100 -> {S2-200, S3-300}
    #
    candidate_lookup = {}

    for _, row in candidate_pairs.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        candidate_id = str(
            row["candidate_entity_id"]
        ).strip()

        if not source1_id or not candidate_id:
            continue

        candidate_lookup.setdefault(
            source1_id,
            set(),
        ).add(candidate_id)

    # Check every final match.
    invalid_matches = []

    for _, row in matching_results.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        matched_ids = parse_match_ids(
            row["matched_entity_ids"]
        )

        valid_candidates = candidate_lookup.get(
            source1_id,
            set(),
        )

        for candidate_id in matched_ids:

            if candidate_id not in valid_candidates:

                invalid_matches.append(
                    (
                        source1_id,
                        candidate_id,
                    )
                )

    if invalid_matches:

        raise ValueError(
            "Final matches contain candidates that "
            "were not present in candidate_pairs: "
            f"{invalid_matches[:10]}"
        )

    return True