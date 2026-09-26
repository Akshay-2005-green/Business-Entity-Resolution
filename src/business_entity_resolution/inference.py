import pandas as pd


REQUIRED_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "score",
}


def validate_prediction_columns(predictions):
    """
    Validate the prediction DataFrame.

    Required columns:
        source1_entity_id
        candidate_entity_id
        score
    """

    missing_columns = REQUIRED_COLUMNS - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def apply_threshold(predictions, threshold=0.80):
    """
    Keep candidate pairs whose model score is greater
    than or equal to the supplied threshold.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Prediction DataFrame containing:
            source1_entity_id
            candidate_entity_id
            score

    threshold : float
        Minimum score required to accept a match.

    Returns
    -------
    pandas.DataFrame
        Filtered candidate pairs.
    """

    validate_prediction_columns(predictions)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    result = predictions[
        predictions["score"] >= threshold
    ].copy()

    return result


def build_match_dictionary(predictions):
    """
    Convert accepted candidate pairs into a dictionary.

    Example input:

        S1-100 | S2-200 | 0.95
        S1-100 | S3-300 | 0.91
        S1-101 | S2-400 | 0.88

    Returns:

        {
            "S1-100": {"S2-200", "S3-300"},
            "S1-101": {"S2-400"}
        }
    """

    validate_prediction_columns(predictions)

    matches = {}

    for _, row in predictions.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        candidate_id = str(
            row["candidate_entity_id"]
        ).strip()

        if not source1_id or not candidate_id:
            continue

        matches.setdefault(
            source1_id,
            set()
        ).add(candidate_id)

    return matches


def generate_predictions(predictions, threshold=0.80):
    """
    Complete inference process.

    Model predictions
        ↓
    threshold
        ↓
    accepted matches
        ↓
    match dictionary
    """

    filtered_predictions = apply_threshold(
        predictions,
        threshold
    )

    matches = build_match_dictionary(
        filtered_predictions
    )

    return matches