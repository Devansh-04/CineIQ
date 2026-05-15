import os
import re
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

MOVIES_RAW_PATH = os.path.join(RAW_DIR, "movies.csv")
RATINGS_RAW_PATH = os.path.join(RAW_DIR, "ratings.csv")

MOVIES_CLEAN_PATH = os.path.join(PROCESSED_DIR, "movies_clean.csv")
RATINGS_CLEAN_PATH = os.path.join(PROCESSED_DIR, "ratings_clean.csv")


def extract_year(title):
    """
    Extracts year from movie title.
    Example: Toy Story (1995) -> 1995
    """
    match = re.search(r"\((\d{4})\)", str(title))

    if match:
        return int(match.group(1))

    return None


def clean_genres(genres):
    """
    Converts MovieLens genre format into cleaner text.
    Example: Action|Adventure|Sci-Fi -> action adventure sci fi
    """
    if pd.isna(genres):
        return ""

    genres = str(genres)
    genres = genres.replace("|", " ")
    genres = genres.replace("-", " ")
    genres = genres.lower()

    return genres


def load_raw_data():
    movies = pd.read_csv(MOVIES_RAW_PATH)
    ratings = pd.read_csv(RATINGS_RAW_PATH)

    return movies, ratings


def preprocess_movies(movies):
    movies = movies.copy()

    movies["title"] = movies["title"].fillna("")
    movies["genres"] = movies["genres"].fillna("(no genres listed)")

    movies["year"] = movies["title"].apply(extract_year)
    movies["genres_clean"] = movies["genres"].apply(clean_genres)

    return movies


def preprocess_ratings(ratings):
    ratings = ratings.copy()

    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    return ratings


def save_processed_data(movies_clean, ratings_clean):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    movies_clean.to_csv(MOVIES_CLEAN_PATH, index=False)
    ratings_clean.to_csv(RATINGS_CLEAN_PATH, index=False)

    print("Saved processed files:")
    print(MOVIES_CLEAN_PATH)
    print(RATINGS_CLEAN_PATH)


def run_preprocessing():
    movies, ratings = load_raw_data()

    movies_clean = preprocess_movies(movies)
    ratings_clean = preprocess_ratings(ratings)

    save_processed_data(movies_clean, ratings_clean)

    return movies_clean, ratings_clean


if __name__ == "__main__":
    run_preprocessing()