import re
from pathlib import Path

import pandas as pd


MATCHING_COLUMNS = {
    "source1_entity_id",
    "matched_entity_ids",
}

CANDIDATE_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_ids",
}

ENTITY_ID_PATTERN = re.compile(r"^S[23]-\d+$")


def validate_submission_outputs(
    matching_results,
    candidate_pairs,
    source1_ids,
):
    """
    Perform local pre-submission validation.

    This is a safety check before running the official
    hackathon validator.

    Parameters
    ----------
    matching_results : pandas.DataFrame
        Final matching_results.tsv data.

    candidate_pairs : pandas.DataFrame
        Final candidate_pairs.tsv data.

    source1_ids : iterable
        Expected Source 1 entity IDs.

    Returns
    -------
    True

    Raises
    ------
    ValueError
        If a submission rule is violated.
    """

    source1_ids = [
        str(entity_id).strip()
        for entity_id in source1_ids
    ]

    expected_source1_ids = set(source1_ids)

    # ---------------------------------------------------------
    # 1. Check columns
    # ---------------------------------------------------------

    missing_matching = (
        MATCHING_COLUMNS
        - set(matching_results.columns)
    )

    if missing_matching:
        raise ValueError(
            "Missing matching result columns: "
            f"{sorted(missing_matching)}"
        )

    missing_candidates = (
        CANDIDATE_COLUMNS
        - set(candidate_pairs.columns)
    )

    if missing_candidates:
        raise ValueError(
            "Missing candidate pair columns: "
            f"{sorted(missing_candidates)}"
        )

    # ---------------------------------------------------------
    # 2. Check matching Source 1 IDs
    # ---------------------------------------------------------

    matching_source1 = (
        matching_results["source1_entity_id"]
        .astype(str)
        .str.strip()
    )

    if matching_source1.duplicated().any():
        raise ValueError(
            "Duplicate Source 1 entity IDs found "
            "in matching_results."
        )

    if set(matching_source1) != expected_source1_ids:
        raise ValueError(
            "matching_results must contain exactly "
            "one row for every test Source 1 entity."
        )

    # ---------------------------------------------------------
    # 3. Check candidate Source 1 IDs
    # ---------------------------------------------------------

    candidate_source1 = (
        candidate_pairs["source1_entity_id"]
        .astype(str)
        .str.strip()
    )

    if candidate_source1.duplicated().any():
        raise ValueError(
            "Duplicate Source 1 entity IDs found "
            "in candidate_pairs."
        )

    if set(candidate_source1) != expected_source1_ids:
        raise ValueError(
            "candidate_pairs must contain exactly "
            "one row for every test Source 1 entity."
        )

    # ---------------------------------------------------------
    # 4. Build candidate lookup
    # ---------------------------------------------------------

    candidate_lookup = {}

    for _, row in candidate_pairs.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        raw_candidates = row["candidate_entity_ids"]

        if pd.isna(raw_candidates):
            raw_candidates = ""

        candidate_text = str(
            raw_candidates
        ).strip()

        if not candidate_text:
            candidate_ids = []
        else:
            candidate_ids = [
                item.strip()
                for item in candidate_text.split(",")
                if item.strip()
            ]

        # Check duplicate candidates.
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(
                f"Duplicate candidate IDs found for "
                f"{source1_id}."
            )

        # Check S2/S3 format.
        for candidate_id in candidate_ids:

            if not ENTITY_ID_PATTERN.match(
                candidate_id
            ):
                raise ValueError(
                    f"Invalid candidate entity ID: "
                    f"{candidate_id}"
                )

        candidate_lookup[source1_id] = set(
            candidate_ids
        )

    # ---------------------------------------------------------
    # 5. Check final matches
    # ---------------------------------------------------------

    for _, row in matching_results.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        raw_matches = row["matched_entity_ids"]

        if pd.isna(raw_matches):
            raw_matches = ""

        match_text = str(
            raw_matches
        ).strip()

        if not match_text:
            match_ids = []
        else:
            match_ids = [
                item.strip()
                for item in match_text.split(",")
                if item.strip()
            ]

        # Duplicate final matches.
        if len(match_ids) != len(set(match_ids)):
            raise ValueError(
                f"Duplicate matched IDs found for "
                f"{source1_id}."
            )

        # Check S2/S3 format.
        for match_id in match_ids:

            if not ENTITY_ID_PATTERN.match(
                match_id
            ):
                raise ValueError(
                    f"Invalid matched entity ID: "
                    f"{match_id}"
                )

        # Every final match must be a candidate.
        valid_candidates = candidate_lookup.get(
            source1_id,
            set(),
        )

        invalid_matches = (
            set(match_ids) - valid_candidates
        )

        if invalid_matches:
            raise ValueError(
                f"Final matches for {source1_id} "
                f"are not present in candidate_pairs: "
                f"{sorted(invalid_matches)}"
            )

    return True


def validate_submission_files(
    matching_path,
    candidate_path,
    source1_ids,
):
    """
    Load the two TSV files and run local validation.
    """

    matching_path = Path(matching_path)
    candidate_path = Path(candidate_path)

    if not matching_path.exists():
        raise FileNotFoundError(
            f"Matching file not found: {matching_path}"
        )

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {candidate_path}"
        )

    matching_results = pd.read_csv(
        matching_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    candidate_pairs = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    return validate_submission_outputs(
        matching_results,
        candidate_pairs,
        source1_ids,
    )