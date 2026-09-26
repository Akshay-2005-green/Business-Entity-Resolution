def parse_match_ids(match_value):
    """
    Convert a comma-separated match string into a set of entity IDs.

    Example:
        "S2-100,S2-200,S3-300"
        -> {"S2-100", "S2-200", "S3-300"}

    Empty or missing values return an empty set.
    """

    if match_value is None:
        return set()

    # Handle pandas NaN without requiring pandas here.
    try:
        if match_value != match_value:
            return set()
    except Exception:
        pass

    match_value = str(match_value).strip()

    if not match_value:
        return set()

    return {
        entity_id.strip()
        for entity_id in match_value.split(",")
        if entity_id.strip()
    }


def calculate_precision(true_matches, predicted_matches):
    """
    Calculate precision for one Source 1 entity.

    Precision =
        correct predicted matches / total predicted matches
    """

    true_matches = set(true_matches)
    predicted_matches = set(predicted_matches)

    # Correct empty prediction for an entity with no true matches.
    if len(predicted_matches) == 0:
        if len(true_matches) == 0:
            return 1.0
        return 0.0

    correct_matches = len(
        true_matches.intersection(predicted_matches)
    )

    return correct_matches / len(predicted_matches)


def calculate_recall(true_matches, predicted_matches):
    """
    Calculate recall for one Source 1 entity.

    Recall =
        correct predicted matches / total actual matches
    """

    true_matches = set(true_matches)
    predicted_matches = set(predicted_matches)

    # If there are no true matches:
    # an empty prediction is correct.
    if len(true_matches) == 0:
        if len(predicted_matches) == 0:
            return 1.0
        return 0.0

    correct_matches = len(
        true_matches.intersection(predicted_matches)
    )

    return correct_matches / len(true_matches)


def calculate_f05(precision, recall):
    """
    Calculate F0.5 score.

    F0.5 =
        (1.25 * Precision * Recall)
        / (0.25 * Precision + Recall)
    """

    denominator = (0.25 * precision) + recall

    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def evaluate_entity(true_matches, predicted_matches):
    """
    Evaluate the matches for one Source 1 entity.

    Returns:
        {
            "precision": ...,
            "recall": ...,
            "f05": ...
        }
    """

    precision = calculate_precision(
        true_matches,
        predicted_matches
    )

    recall = calculate_recall(
        true_matches,
        predicted_matches
    )

    f05 = calculate_f05(
        precision,
        recall
    )

    return {
        "precision": precision,
        "recall": recall,
        "f05": f05
    }


def evaluate_predictions(true_matches, predicted_matches):
    """
    Backward-compatible evaluation function.

    This evaluates a single Source 1 entity.
    """

    return evaluate_entity(
        true_matches,
        predicted_matches
    )
def evaluate_all_entities(ground_truth, predictions):
    """
    Calculate macro-average Precision, Recall and F0.5
    across all Source 1 entities.

    Parameters
    ----------
    ground_truth : dict
        {
            "S1-100": {"S2-200", "S3-300"},
            "S1-101": set(),
        }

    predictions : dict
        {
            "S1-100": {"S2-200", "S3-999"},
            "S1-101": set(),
        }

    Returns
    -------
    dict
        Macro-average precision, recall and F0.5.
    """

    all_entity_ids = set(ground_truth) | set(predictions)

    if not all_entity_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f05": 0.0
        }

    precision_scores = []
    recall_scores = []
    f05_scores = []

    for entity_id in all_entity_ids:

        true_matches = ground_truth.get(
            entity_id,
            set()
        )

        predicted_matches = predictions.get(
            entity_id,
            set()
        )

        result = evaluate_entity(
            true_matches,
            predicted_matches
        )

        precision_scores.append(
            result["precision"]
        )

        recall_scores.append(
            result["recall"]
        )

        f05_scores.append(
            result["f05"]
        )

    return {
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
        "f05": sum(f05_scores) / len(f05_scores)
    }

def load_ground_truth(file_path):
    """
    Load the challenge ground-truth TSV file.

    Expected columns:
        source1_entity_id
        matched_entity_ids

    Returns
    -------
    dict
        {
            "S1-100": {"S2-200", "S3-300"},
            "S1-101": set(),
        }
    """

    import pandas as pd

    df = pd.read_csv(
        file_path,
        sep="\t"
    )

    required_columns = {
        "source1_entity_id",
        "matched_entity_ids"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    ground_truth = {}

    for _, row in df.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        matches = parse_match_ids(
            row["matched_entity_ids"]
        )

        ground_truth[source1_id] = matches

    return ground_truth