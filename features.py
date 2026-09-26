from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# EDIT SIMILARITY
# ============================================================

def edit_similarity(text1, text2):
    return SequenceMatcher(
        None,
        str(text1),
        str(text2)
    ).ratio()


# ============================================================
# TF-IDF + COSINE SIMILARITY
# ============================================================

def tfidf_cosine_similarity(text1, text2):

    text1 = str(text1)
    text2 = str(text2)

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform([
        text1,
        text2
    ])

    score = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return score


# ============================================================
# TEST
# ============================================================

edit_score = edit_similarity(
    "ABC Restaurant",
    "ABC Resturant"
)

print("Edit similarity:", edit_score)


cosine_score = tfidf_cosine_similarity(
    "ABC Restaurant Kanpur",
    "ABC Restaurant Kanpur Pvt Ltd"
)

print("TF-IDF cosine similarity:", cosine_score)