from pathlib import Path

import pandas as pd

from .inference import generate_predictions


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


def save_candidate_pairs(
    candidate_pairs,
    output_path,
):
    """
    Save candidate pairs as a TSV file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_pairs.to_csv(
        output_path,
        sep="\t",
        index=False,
    )