import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MOVIES_PATH = os.path.join(DATA_DIR, "movies_clean.csv")
SENTIMENT_PATH = os.path.join(DATA_DIR, "sentiment_scores.csv")


def get_vader_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        raise ImportError(
            "vaderSentiment is not installed. Run: pip install vaderSentiment"
        )

    return SentimentIntensityAnalyzer()


def get_sentiment_score(text):
    """
    Returns sentiment score between 0 and 1.
    VADER compound score is from -1 to +1.
    We convert it to 0 to 1.
    """
    analyzer = get_vader_analyzer()

    text = "" if pd.isna(text) else str(text)

    compound = analyzer.polarity_scores(text)["compound"]

    normalized_score = (compound + 1) / 2

    return normalized_score


def create_basic_sentiment_scores():
    """
    Creates a simple sentiment score file using movie title + genres.
    This is a fallback/demo version.

    If you later use IMDB review data, replace this with actual review sentiment.
    """
    movies = pd.read_csv(MOVIES_PATH)

    movies = movies.copy()

    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)

    movies["sentiment_text"] = movies["title"] + " " + movies["genres"]

    movies["sentiment_score"] = movies["sentiment_text"].apply(
        get_sentiment_score
    )

    output = movies[["movieId", "sentiment_score"]]

    output.to_csv(SENTIMENT_PATH, index=False)

    print("Saved sentiment scores at:", SENTIMENT_PATH)

    return output


def load_sentiment_scores():
    if not os.path.exists(SENTIMENT_PATH):
        return create_basic_sentiment_scores()

    return pd.read_csv(SENTIMENT_PATH)


if __name__ == "__main__":
    sentiment_scores = create_basic_sentiment_scores()
    print(sentiment_scores.head())