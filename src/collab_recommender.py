import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MOVIES_PATH = os.path.join(DATA_DIR, "movies_clean.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings_clean.csv")


def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    return movies, ratings


def get_user_rated_movies(user_id):
    movies, ratings = load_data()

    user_ratings = ratings[ratings["userId"] == user_id]

    user_movies = user_ratings.merge(
        movies,
        on="movieId",
        how="left"
    )

    return user_movies.sort_values(by="rating", ascending=False)


def recommend_popular_movies(user_id=None, top_n=10, min_ratings=50):
    """
    Simple collaborative baseline:
    recommends movies with high average rating and enough number of ratings.

    If user_id is given, already-rated movies are removed.
    """
    movies, ratings = load_data()

    movie_stats = (
        ratings.groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    movie_stats = movie_stats[movie_stats["rating_count"] >= min_ratings]

    recommendations = movie_stats.merge(
        movies,
        on="movieId",
        how="left"
    )

    if user_id is not None:
        already_rated = ratings[ratings["userId"] == user_id]["movieId"].unique()
        recommendations = recommendations[
            ~recommendations["movieId"].isin(already_rated)
        ]

    recommendations = recommendations.sort_values(
        by=["avg_rating", "rating_count"],
        ascending=False
    ).head(top_n)

    output_columns = [
        "movieId",
        "title",
        "genres",
        "year",
        "avg_rating",
        "rating_count"
    ]

    available_columns = [
        col for col in output_columns
        if col in recommendations.columns
    ]

    return recommendations[available_columns]


def recommend_by_user_genres(user_id, top_n=10):
    """
    Recommends movies from genres that the user rated highly.
    """
    movies, ratings = load_data()

    user_movies = get_user_rated_movies(user_id)

    if user_movies.empty:
        return recommend_popular_movies(top_n=top_n)

    liked_movies = user_movies[user_movies["rating"] >= 4.0]

    if liked_movies.empty:
        return recommend_popular_movies(user_id=user_id, top_n=top_n)

    liked_genres = []

    for _, row in liked_movies.iterrows():
        genres = str(row["genres"]).split("|")

        for genre in genres:
            if genre != "(no genres listed)" and genre != "nan":
                liked_genres.append(genre)

    if len(liked_genres) == 0:
        return recommend_popular_movies(user_id=user_id, top_n=top_n)

    top_genres = pd.Series(liked_genres).value_counts().head(5).index.tolist()

    movie_stats = (
        ratings.groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    candidate_movies = movies.copy()

    candidate_movies["genre_match"] = candidate_movies["genres"].apply(
        lambda x: sum(genre in str(x).split("|") for genre in top_genres)
    )

    candidate_movies = candidate_movies[candidate_movies["genre_match"] > 0]

    candidate_movies = candidate_movies.merge(
        movie_stats,
        on="movieId",
        how="left"
    )

    already_rated = ratings[ratings["userId"] == user_id]["movieId"].unique()

    candidate_movies = candidate_movies[
        ~candidate_movies["movieId"].isin(already_rated)
    ]

    candidate_movies["avg_rating"] = candidate_movies["avg_rating"].fillna(0)
    candidate_movies["rating_count"] = candidate_movies["rating_count"].fillna(0)

    recommendations = candidate_movies.sort_values(
        by=["genre_match", "avg_rating", "rating_count"],
        ascending=False
    ).head(top_n)

    output_columns = [
        "movieId",
        "title",
        "genres",
        "year",
        "genre_match",
        "avg_rating",
        "rating_count"
    ]

    available_columns = [
        col for col in output_columns
        if col in recommendations.columns
    ]

    return recommendations[available_columns]


if __name__ == "__main__":
    recs = recommend_popular_movies(user_id=1, top_n=10)
    print(recs)