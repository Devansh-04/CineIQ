import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MOVIES_PATH = os.path.join(DATA_DIR, "movies_clean.csv")


def load_movies():
    movies = pd.read_csv(MOVIES_PATH)
    return movies


def prepare_content_text(movies):
    movies = movies.copy()

    if "genres_clean" not in movies.columns:
        movies["genres_clean"] = ""

    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres_clean"] = movies["genres_clean"].fillna("").astype(str)

    movies["content_text"] = movies["title"] + " " + movies["genres_clean"]

    return movies


def build_tfidf_matrix(movies):
    movies = prepare_content_text(movies)

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = tfidf.fit_transform(movies["content_text"])

    return movies, tfidf, tfidf_matrix


def recommend_similar_movies(movie_title, top_n=10):
    movies = load_movies()
    movies, tfidf, tfidf_matrix = build_tfidf_matrix(movies)

    if movie_title not in movies["title"].values:
        raise ValueError(f"Movie '{movie_title}' not found.")

    movie_index = movies[movies["title"] == movie_title].index[0]

    cosine_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    movies["content_score"] = cosine_scores

    recommendations = movies[movies["title"] != movie_title].sort_values(
        by="content_score",
        ascending=False
    ).head(top_n)

    output_columns = [
        "movieId",
        "title",
        "genres",
        "genres_clean",
        "year",
        "content_score"
    ]

    available_columns = [
        col for col in output_columns
        if col in recommendations.columns
    ]

    return recommendations[available_columns]


if __name__ == "__main__":
    recs = recommend_similar_movies("Toy Story (1995)", top_n=10)
    print(recs)