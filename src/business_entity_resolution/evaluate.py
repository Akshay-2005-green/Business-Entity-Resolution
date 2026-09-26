def calculate_precision(true_matches, predicted_matches):
    """
    Calculate precision.

    Precision = correct predicted matches / total predicted matches
    """

    true_matches = set(true_matches)
    predicted_matches = set(predicted_matches)

    if len(predicted_matches) == 0:
        return 0.0

    correct_matches = len(
        true_matches.intersection(predicted_matches)
    )

    return correct_matches / len(predicted_matches)


def calculate_recall(true_matches, predicted_matches):
    """
    Calculate recall.

    Recall = correct predicted matches / total actual matches
    """

    true_matches = set(true_matches)
    predicted_matches = set(predicted_matches)

    if len(true_matches) == 0:
        return 0.0

    correct_matches = len(
        true_matches.intersection(predicted_matches)
    )

    return correct_matches / len(true_matches)


def calculate_f05(precision, recall):
    """
    Calculate F0.5 score.

    F0.5 = (1.25 * Precision * Recall)
           / (0.25 * Precision + Recall)
    """

    denominator = (0.25 * precision) + recall

    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def evaluate_predictions(true_matches, predicted_matches):
    """
    Calculate precision, recall and F0.5.
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