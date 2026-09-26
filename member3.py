import pandas as pd
import re
import joblib

from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ============================================================
# 1. LOAD DATA
# ============================================================

source1 = pd.read_csv("train_source1.tsv", sep="\t")
source2 = pd.read_csv("train_source2.tsv", sep="\t")
source3 = pd.read_csv("train_source3.tsv", sep="\t")
ground_truth = pd.read_csv("ground_truth.tsv", sep="\t")


print("Data loaded successfully!")

print("Source 1 rows:", len(source1))
print("Source 2 rows:", len(source2))
print("Source 3 rows:", len(source3))
print("Ground truth rows:", len(ground_truth))


# ============================================================
# 2. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Normalize Source 1
source1["name_clean"] = source1["name"].apply(normalize_text)
source1["address_clean"] = source1["address"].apply(normalize_text)


# Normalize Source 2
source2["name_clean"] = source2["name"].apply(normalize_text)
source2["address_clean"] = source2["address"].apply(normalize_text)


# ============================================================
# 3. SIMILARITY FUNCTIONS
# ============================================================

def name_similarity(text1, text2):

    return SequenceMatcher(
        None,
        str(text1),
        str(text2)
    ).ratio()


def address_similarity(text1, text2):

    return SequenceMatcher(
        None,
        str(text1),
        str(text2)
    ).ratio()


def edit_similarity(text1, text2):

    return SequenceMatcher(
        None,
        str(text1),
        str(text2)
    ).ratio()


def jaccard_similarity(text1, text2):

    words1 = set(str(text1).split())
    words2 = set(str(text2).split())

    # Both empty
    if len(words1) == 0 and len(words2) == 0:
        return 1.0

    # One empty
    if len(words1) == 0 or len(words2) == 0:
        return 0.0

    intersection = words1.intersection(words2)

    union = words1.union(words2)

    return len(intersection) / len(union)


# ============================================================
# 4. CANDIDATE GENERATION
# ============================================================
#
# NOTE:
# In the final team pipeline, Member 2 should provide
# the candidate pairs.
#
# This country-based candidate generation is kept here
# for the current practice dataset.
# ============================================================

candidate_pairs = []


for _, row1 in source1.iterrows():

    for _, row2 in source2.iterrows():

        # Compare only records from the same country
        if row1["country"] == row2["country"]:

            candidate_pairs.append({

                "source1_id": row1["source1_id"],

                "source2_id": row2["source2_id"],

                "name1": row1["name_clean"],

                "name2": row2["name_clean"],

                "address1": row1["address_clean"],

                "address2": row2["address_clean"],

                "country1": row1["country"],

                "country2": row2["country"]

            })


candidate_df = pd.DataFrame(candidate_pairs)


print("Candidate pairs:", len(candidate_df))


# ============================================================
# 5. NAME SIMILARITY
# ============================================================

candidate_df["name_similarity"] = candidate_df.apply(

    lambda row: name_similarity(
        row["name1"],
        row["name2"]
    ),

    axis=1
)


# ============================================================
# 6. ADDRESS SIMILARITY
# ============================================================

candidate_df["address_similarity"] = candidate_df.apply(

    lambda row: address_similarity(
        row["address1"],
        row["address2"]
    ),

    axis=1
)


# ============================================================
# 7. NAME JACCARD SIMILARITY
# ============================================================

candidate_df["name_jaccard"] = candidate_df.apply(

    lambda row: jaccard_similarity(
        row["name1"],
        row["name2"]
    ),

    axis=1
)


# ============================================================
# 8. ADDRESS JACCARD SIMILARITY
# ============================================================

candidate_df["address_jaccard"] = candidate_df.apply(

    lambda row: jaccard_similarity(
        row["address1"],
        row["address2"]
    ),

    axis=1
)


# ============================================================
# 9. EDIT SIMILARITY
# ============================================================

candidate_df["edit_similarity"] = candidate_df.apply(

    lambda row: edit_similarity(
        row["name1"],
        row["name2"]
    ),

    axis=1
)


# ============================================================
# 10. TF-IDF + COSINE SIMILARITY
# ============================================================

print("Calculating TF-IDF similarity...")


# Combine all Source 1 and Source 2 names
all_names = pd.concat(

    [
        source1["name_clean"],
        source2["name_clean"]
    ]

).fillna("")


# Character-level TF-IDF
vectorizer = TfidfVectorizer(

    analyzer="char",

    ngram_range=(2, 5)

)


# Fit TF-IDF once
tfidf_matrix = vectorizer.fit_transform(all_names)


# Separate Source 1 and Source 2 vectors
source1_count = len(source1)

source1_vectors = tfidf_matrix[:source1_count]

source2_vectors = tfidf_matrix[source1_count:]


# Create ID → vector index mappings
source1_vector_index = dict(

    zip(
        source1["source1_id"],
        range(len(source1))
    )

)


source2_vector_index = dict(

    zip(
        source2["source2_id"],
        range(len(source2))
    )

)


# Calculate cosine similarity
cosine_scores = []


for _, row in candidate_df.iterrows():

    i = source1_vector_index[
        row["source1_id"]
    ]

    j = source2_vector_index[
        row["source2_id"]
    ]

    score = cosine_similarity(

        source1_vectors[i],

        source2_vectors[j]

    )[0][0]


    cosine_scores.append(score)


candidate_df["tfidf_cosine"] = cosine_scores


print("TF-IDF similarity completed!")


# ============================================================
# 11. CREATE TRAINING LABELS
# ============================================================

true_matches = dict(

    zip(
        ground_truth["source1_id"],
        ground_truth["source2_id"]
    )

)


candidate_df["label"] = candidate_df.apply(

    lambda row:

        1

        if true_matches.get(
            row["source1_id"]
        ) == row["source2_id"]

        else 0,

    axis=1

)


print("Labels created!")


# ============================================================
# 12. SAVE FEATURE DATA AS TSV
# ============================================================
#
# IMPORTANT:
# sep="\t" means the output is a TAB-SEPARATED FILE.
# ============================================================

candidate_df.to_csv(

    "member3_features.tsv",

    sep="\t",

    index=False

)


print("Feature file saved as member3_features.tsv")


# ============================================================
# 13. SELECT ML FEATURES
# ============================================================

features = [

    "name_similarity",

    "address_similarity",

    "name_jaccard",

    "address_jaccard",

    "edit_similarity",

    "tfidf_cosine"

]


X = candidate_df[features]

y = candidate_df["label"]


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


print("Training data:", len(X_train))
print("Testing data:", len(X_test))


# ============================================================
# 15. LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(

    max_iter=1000

)


model.fit(

    X_train,

    y_train

)


print("Logistic Regression model trained successfully!")


# ============================================================
# 16. SAVE MODEL
# ============================================================

joblib.dump(

    model,

    "member3_model.pkl"

)


print("Model saved as member3_model.pkl")


# ============================================================
# 17. TEST MODEL
# ============================================================

y_pred = model.predict(

    X_test

)


print("\nModel Evaluation:")
print("----------------------------------------")


print(

    classification_report(

        y_test,

        y_pred,

        zero_division=0

    )

)


# ============================================================
# 18. GENERATE MATCH PROBABILITY
# ============================================================

candidate_df["match_probability"] = model.predict_proba(

    candidate_df[features]

)[:, 1]


# ============================================================
# 19. GENERATE BINARY PREDICTION
# ============================================================
#
# Current practice threshold = 0.5
#
# NOTE:
# Final threshold should be tuned by Member 4
# using validation data.
# ============================================================

candidate_df["prediction"] = (

    candidate_df["match_probability"] >= 0.5

).astype(int)


# ============================================================
# 20. SAVE FINAL MEMBER 3 PREDICTIONS AS TSV
# ============================================================

candidate_df.to_csv(

    "member3_predictions.tsv",

    sep="\t",

    index=False

)


print("Final predictions saved as member3_predictions.tsv")


# ============================================================
# 21. COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("Member 3 processing completed!")
print("========================================")

print("\nFiles generated:")

print("1. member3_features.tsv")
print("2. member3_predictions.tsv")
print("3. member3_model.pkl")