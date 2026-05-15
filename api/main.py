from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import sys

# =====================================================
# PATH SETUP
# =====================================================

# main.py is inside cineiq/api/
# BASE_DIR becomes cineiq/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
SRC_DIR = os.path.join(BASE_DIR, "src")

MOVIES_PATH = os.path.join(DATA_DIR, "movies_clean.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings_clean.csv")

sys.path.append(SRC_DIR)

from hybrid_recommender import hybrid_recommend

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="CINEIQ API",
    description="Explainable Hybrid Movie Recommendation API",
    version="1.0.0"
)

# Allows frontend/dashboard to call API later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD DATA
# =====================================================

def load_movies():
    if not os.path.exists(MOVIES_PATH):
        raise FileNotFoundError("movies_clean.csv not found")
    return pd.read_csv(MOVIES_PATH)


def load_ratings():
    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError("ratings_clean.csv not found")
    return pd.read_csv(RATINGS_PATH)


# =====================================================
# BASIC ROUTES
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to CINEIQ API",
        "available_endpoints": [
            "/health",
            "/movies",
            "/users",
            "/recommend",
            "/similar",
            "/user-profile"
        ]
    }


@app.get("/health")
def health_check():
    movies_exists = os.path.exists(MOVIES_PATH)
    ratings_exists = os.path.exists(RATINGS_PATH)

    return {
        "status": "running",
        "movies_clean_found": movies_exists,
        "ratings_clean_found": ratings_exists,
        "movies_path": MOVIES_PATH,
        "ratings_path": RATINGS_PATH
    }


# =====================================================
# MOVIE AND USER ENDPOINTS
# =====================================================

@app.get("/movies")
def get_movies(limit: int = Query(20, ge=1, le=100)):
    try:
        movies = load_movies()

        result = movies[["movieId", "title", "genres", "year"]].head(limit)

        return {
            "count": len(result),
            "movies": result.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users")
def get_users(limit: int = Query(20, ge=1, le=100)):
    try:
        ratings = load_ratings()

        user_ids = sorted(ratings["userId"].unique())[:limit]

        return {
            "count": len(user_ids),
            "users": [int(user_id) for user_id in user_ids]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# RECOMMENDATION ENDPOINT
# =====================================================

@app.get("/recommend")
def recommend_movies(
    user_id: int = Query(..., description="User ID"),
    movie_title: str = Query(..., description="Movie title liked by user"),
    top_n: int = Query(10, ge=1, le=50)
):
    try:
        recommendations = hybrid_recommend(
            movie_title=movie_title,
            user_id=user_id,
            top_n=top_n
        )

        return {
            "user_id": user_id,
            "input_movie": movie_title,
            "top_n": top_n,
            "recommendations": recommendations.to_dict(orient="records")
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# SIMILAR MOVIES ENDPOINT
# =====================================================

@app.get("/similar")
def similar_movies(
    movie_title: str = Query(..., description="Movie title"),
    top_n: int = Query(10, ge=1, le=50)
):
    """
    Similar movies using the same hybrid_recommend function.
    Uses the first available user as a default user.
    """

    try:
        ratings = load_ratings()

        default_user = int(ratings["userId"].iloc[0])

        recommendations = hybrid_recommend(
            movie_title=movie_title,
            user_id=default_user,
            top_n=top_n
        )

        cols_to_keep = [
            "movieId",
            "title",
            "genres",
            "year",
            "content_score",
            "score",
            "explanation"
        ]

        available_cols = [
            col for col in cols_to_keep
            if col in recommendations.columns
        ]

        similar = recommendations[available_cols]

        return {
            "input_movie": movie_title,
            "top_n": top_n,
            "similar_movies": similar.to_dict(orient="records")
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# USER PROFILE ENDPOINT
# =====================================================

@app.get("/user-profile")
def user_profile(
    user_id: int = Query(..., description="User ID")
):
    try:
        movies = load_movies()
        ratings = load_ratings()

        user_ratings = ratings[ratings["userId"] == user_id]

        if user_ratings.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No ratings found for user {user_id}"
            )

        user_movies = user_ratings.merge(
            movies,
            on="movieId",
            how="left"
        )

        # Genre profile
        genre_rows = []

        for _, row in user_movies.iterrows():
            genres = str(row["genres"]).split("|")

            for genre in genres:
                genre = genre.strip()

                if genre != "(no genres listed)" and genre != "nan" and genre != "":
                    genre_rows.append({
                        "genre": genre,
                        "rating": row["rating"]
                    })

        genre_df = pd.DataFrame(genre_rows)

        if not genre_df.empty:
            genre_profile = (
                genre_df.groupby("genre")["rating"]
                .agg(["mean", "count"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )

            genre_data = genre_profile.to_dict(orient="records")
        else:
            genre_data = []

        # Decade profile
        decade_data = []

        if "year" in user_movies.columns:
            decade_df = user_movies.dropna(subset=["year"]).copy()

            if not decade_df.empty:
                decade_df["decade"] = (decade_df["year"] // 10 * 10).astype(int)

                decade_profile = (
                    decade_df.groupby("decade")["rating"]
                    .mean()
                    .reset_index()
                    .sort_values("decade")
                )

                decade_data = decade_profile.to_dict(orient="records")

        return {
            "user_id": user_id,
            "movies_rated": int(len(user_ratings)),
            "average_rating": float(round(user_ratings["rating"].mean(), 2)),
            "highest_rating": float(user_ratings["rating"].max()),
            "lowest_rating": float(user_ratings["rating"].min()),
            "genre_profile": genre_data,
            "decade_profile": decade_data
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))