from sklearn.feature_extraction.text import TfidfVectorizer


def make_vectorizer():
    """
    Classic lexical baseline:
    word-level TF-IDF with light regularization.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        stop_words="english",
    )
