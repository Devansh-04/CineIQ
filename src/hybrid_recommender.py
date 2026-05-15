import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from explainability import explain_recommendation_row


# =====================================================
# PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MOVIES_PATH = os.path.join(DATA_DIR, "movies_clean.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings_clean.csv")
SENTIMENT_PATH = os.path.join(DATA_DIR, "sentiment_scores.csv")


# =====================================================
# LOAD DATA
# =====================================================

def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    return movies, ratings


# =====================================================
# CONTENT FEATURES
# =====================================================

def prepare_content_features(movies):
    movies = movies.copy()

    if "genres_clean" not in movies.columns:
        movies["genres_clean"] = ""

    if "genres" not in movies.columns:
        movies["genres"] = ""

    if "title" not in movies.columns:
        movies["title"] = ""

    movies["genres_clean"] = movies["genres_clean"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)
    movies["title"] = movies["title"].fillna("").astype(str)

    # Better content text:
    # title + cleaned genre words
    movies["content_text"] = (
        movies["title"] + " " + movies["genres_clean"]
    )

    return movies


# =====================================================
# USER GENRE PREFERENCE
# =====================================================

def get_user_genre_preferences(user_id, movies, ratings):
    user_ratings = ratings[ratings["userId"] == user_id]

    if user_ratings.empty:
        return {}

    user_movies = user_ratings.merge(
        movies,
        on="movieId",
        how="left"
    )

    genre_scores = {}

    for _, row in user_movies.iterrows():
        raw_genres = str(row["genres"])

        genres = raw_genres.split("|")

        for genre in genres:
            genre = genre.strip()

            if genre == "(no genres listed)" or genre == "nan" or genre == "":
                continue

            if genre not in genre_scores:
                genre_scores[genre] = []

            genre_scores[genre].append(row["rating"])

    genre_preferences = {
        genre: np.mean(scores)
        for genre, scores in genre_scores.items()
    }

    return genre_preferences


def calculate_genre_preference_score(movie_genres, genre_preferences):
    genres = str(movie_genres).split("|")

    scores = []

    for genre in genres:
        genre = genre.strip()

        if genre in genre_preferences:
            scores.append(genre_preferences[genre])

    if len(scores) == 0:
        return 0

    return np.mean(scores) / 5.0


# =====================================================
# POPULARITY SCORE
# =====================================================

def get_popularity_scores(ratings):
    movie_stats = (
        ratings.groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    min_count = movie_stats["rating_count"].min()
    max_count = movie_stats["rating_count"].max()

    if max_count == min_count:
        movie_stats["popularity_score"] = 0.5
    else:
        movie_stats["popularity_score"] = (
            movie_stats["rating_count"] - min_count
        ) / (
            max_count - min_count
        )

    movie_stats["avg_rating_score"] = movie_stats["avg_rating"] / 5.0

    return movie_stats


# =====================================================
# SENTIMENT SCORE
# =====================================================

def load_sentiment_scores():
    if os.path.exists(SENTIMENT_PATH):
        sentiment = pd.read_csv(SENTIMENT_PATH)

        if "movieId" in sentiment.columns and "sentiment_score" in sentiment.columns:
            return sentiment[["movieId", "sentiment_score"]]

    return None


# =====================================================
# EXPLANATION
# =====================================================

def make_explanation(row, movie_title):
    explanation = (
        f"Recommended because it is content-wise similar to '{movie_title}', "
        f"matches the user's preferred genres, and has strong audience rating signals. "
        f"Content similarity = {row['content_score']:.2f}, "
        f"user genre preference = {row['user_preference_score']:.2f}, "
        f"average rating signal = {row['avg_rating_score']:.2f}, "
        f"sentiment signal = {row['sentiment_score']:.2f}."
    )

    return explanation


# =====================================================
# MAIN HYBRID RECOMMENDER FUNCTION
# =====================================================

def hybrid_recommend(movie_title, user_id, top_n=10):
    movies, ratings = load_data()

    movies = prepare_content_features(movies)

    if movie_title not in movies["title"].values:
        raise ValueError(f"Movie '{movie_title}' not found in movies_clean.csv")

    # -----------------------------
    # TF-IDF using genres_clean
    # -----------------------------
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = tfidf.fit_transform(movies["content_text"])

    movie_index = movies[movies["title"] == movie_title].index[0]

    cosine_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    movies["content_score"] = cosine_scores

    # -----------------------------
    # User genre preference
    # -----------------------------
    genre_preferences = get_user_genre_preferences(
        user_id=user_id,
        movies=movies,
        ratings=ratings
    )

    movies["user_preference_score"] = movies["genres"].apply(
        lambda x: calculate_genre_preference_score(
            x,
            genre_preferences
        )
    )

    # -----------------------------
    # Popularity and average rating
    # -----------------------------
    popularity = get_popularity_scores(ratings)

    movies = movies.merge(
        popularity,
        on="movieId",
        how="left"
    )

    movies["popularity_score"] = movies["popularity_score"].fillna(0)
    movies["avg_rating_score"] = movies["avg_rating_score"].fillna(0)

    # -----------------------------
    # Sentiment
    # -----------------------------
    sentiment = load_sentiment_scores()

    if sentiment is not None:
        movies = movies.merge(
            sentiment,
            on="movieId",
            how="left"
        )
        movies["sentiment_score"] = movies["sentiment_score"].fillna(0.5)
    else:
        movies["sentiment_score"] = 0.5

    # -----------------------------
    # Remove already-rated movies
    # -----------------------------
    already_rated = ratings[ratings["userId"] == user_id]["movieId"].unique()

    candidates = movies[
        ~movies["movieId"].isin(already_rated)
    ].copy()

    selected_movie_id = movies.loc[movie_index, "movieId"]

    candidates = candidates[
        candidates["movieId"] != selected_movie_id
    ]

    # -----------------------------
    # Hybrid score
    # -----------------------------
    candidates["score"] = (
        0.40 * candidates["content_score"]
        + 0.25 * candidates["user_preference_score"]
        + 0.20 * candidates["avg_rating_score"]
        + 0.10 * candidates["popularity_score"]
        + 0.05 * candidates["sentiment_score"]
    )

    # -----------------------------
    # Explanation
    # -----------------------------
    candidates["explanation"] = candidates.apply(
    lambda row: explain_recommendation_row(row, movie_title),
    axis=1
)

    recommendations = candidates.sort_values(
        by="score",
        ascending=False
    ).head(top_n)

    output_columns = [
        "movieId",
        "title",
        "genres",
        "genres_clean",
        "year",
        "content_score",
        "user_preference_score",
        "avg_rating_score",
        "popularity_score",
        "sentiment_score",
        "score",
        "explanation"
    ]

    available_columns = [
        col for col in output_columns
        if col in recommendations.columns
    ]

    return recommendations[available_columns]