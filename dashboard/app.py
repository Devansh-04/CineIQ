import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# =====================================================
# PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
SRC_DIR = os.path.join(BASE_DIR, "src")

MOVIES_PATH = os.path.join(DATA_DIR, "movies_clean.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings_clean.csv")

sys.path.append(SRC_DIR)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="CINEIQ Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 CINEIQ")
st.caption("Explainable Hybrid Movie Recommendation Dashboard")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    return movies, ratings


if not os.path.exists(MOVIES_PATH) or not os.path.exists(RATINGS_PATH):
    st.error("Input files missing.")
    st.write("Expected files:")
    st.code(MOVIES_PATH)
    st.code(RATINGS_PATH)
    st.stop()

movies, ratings = load_data()

# =====================================================
# COLUMN CHECK
# =====================================================

required_movie_cols = ["movieId", "title", "genres", "year"]
required_rating_cols = ["userId", "movieId", "rating"]

missing_movie_cols = [c for c in required_movie_cols if c not in movies.columns]
missing_rating_cols = [c for c in required_rating_cols if c not in ratings.columns]

if missing_movie_cols:
    st.error(f"movies_clean.csv is missing columns: {missing_movie_cols}")
    st.stop()

if missing_rating_cols:
    st.error(f"ratings_clean.csv is missing columns: {missing_rating_cols}")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🎛️ CINEIQ Controls")

user_ids = sorted(ratings["userId"].unique())

selected_user = st.sidebar.selectbox(
    "Select User ID",
    user_ids
)

num_recs = st.sidebar.slider(
    "Number of Recommendations",
    5,
    20,
    10
)

st.sidebar.markdown("---")

show_profile = st.sidebar.checkbox("User Taste Profile", True)
show_history = st.sidebar.checkbox("Rating History", True)
show_recommender = st.sidebar.checkbox("Hybrid Recommender", True)

# =====================================================
# USER DATA
# =====================================================

user_ratings = ratings[ratings["userId"] == selected_user].copy()

user_movies = user_ratings.merge(
    movies,
    on="movieId",
    how="left"
)

# =====================================================
# TOP METRICS
# =====================================================

st.markdown("## 📊 User Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Movies Rated", len(user_ratings))

with col2:
    st.metric("Average Rating", round(user_ratings["rating"].mean(), 2))

with col3:
    st.metric("Highest Rating", user_ratings["rating"].max())

with col4:
    st.metric("Lowest Rating", user_ratings["rating"].min())

st.markdown("---")

# =====================================================
# USER TASTE PROFILE
# =====================================================

if show_profile:
    st.markdown("## 👤 User Taste Profile")

    genre_rows = []

    for _, row in user_movies.iterrows():
        genres = str(row["genres"]).split("|")

        for genre in genres:
            if genre != "(no genres listed)" and genre != "nan":
                genre_rows.append({
                    "genre": genre,
                    "rating": row["rating"]
                })

    genre_df = pd.DataFrame(genre_rows)

    col1, col2 = st.columns(2)

    # -----------------------------
    # Radar chart
    # -----------------------------
    with col1:
        st.subheader("🎯 Genre Radar Chart")

        if not genre_df.empty:
            genre_pref = (
                genre_df.groupby("genre")["rating"]
                .mean()
                .reset_index()
                .sort_values("rating", ascending=False)
                .head(8)
            )

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=genre_pref["rating"],
                theta=genre_pref["genre"],
                fill="toself",
                name="Average Rating"
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )
                ),
                showlegend=False,
                title="Top Genre Preferences"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No genre data available for this user.")

    # -----------------------------
    # Most watched genre pie
    # -----------------------------
    with col2:
        st.subheader("🍿 Most Watched Genres")

        if not genre_df.empty:
            genre_count = genre_df["genre"].value_counts().reset_index()
            genre_count.columns = ["genre", "count"]

            fig = px.pie(
                genre_count.head(8),
                names="genre",
                values="count",
                title="Genre Watch Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No genre data available.")

    col3, col4 = st.columns(2)

    # -----------------------------
    # Genre average rating bar chart
    # -----------------------------
    with col3:
        st.subheader("⭐ Average Rating by Genre")

        if not genre_df.empty:
            genre_avg = (
                genre_df.groupby("genre")["rating"]
                .mean()
                .reset_index()
                .sort_values("rating", ascending=False)
            )

            fig = px.bar(
                genre_avg,
                x="genre",
                y="rating",
                title="Average Rating by Genre"
            )

            fig.update_layout(xaxis_tickangle=-45)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No genre rating data available.")

    # -----------------------------
    # Decade preference
    # -----------------------------
    with col4:
        st.subheader("📅 Decade Preference")

        decade_df = user_movies.dropna(subset=["year"]).copy()

        if not decade_df.empty:
            decade_df["decade"] = (decade_df["year"] // 10 * 10).astype(int)

            decade_pref = (
                decade_df.groupby("decade")["rating"]
                .mean()
                .reset_index()
                .sort_values("decade")
            )

            fig = px.line(
                decade_pref,
                x="decade",
                y="rating",
                markers=True,
                title="Average Rating by Decade"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No year data available.")

st.markdown("---")

# =====================================================
# RATING HISTORY
# =====================================================

if show_history:
    st.markdown("## 📜 User Rating History")

    col1, col2 = st.columns([2, 1])

    with col1:
        rated_movies = user_movies[
            ["title", "genres", "year", "rating"]
        ].sort_values(
            by="rating",
            ascending=False
        )

        st.dataframe(rated_movies, use_container_width=True)

    with col2:
        fig = px.histogram(
            user_ratings,
            x="rating",
            nbins=10,
            title="Rating Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================================
# HYBRID RECOMMENDER
# =====================================================

if show_recommender:
    st.markdown("## ⭐ Hybrid Movie Recommendations")

    try:
        from hybrid_recommender import hybrid_recommend
    except Exception as e:
        st.error("Could not import hybrid_recommend from src/hybrid_recommender.py")
        st.code(str(e))
        st.stop()

    movie_list = sorted(movies["title"].dropna().unique())

    selected_movie = st.selectbox(
        "Choose a movie you like",
        movie_list
    )

    if st.button("Generate Recommendations"):
        try:
            recommendations = hybrid_recommend(
                movie_title=selected_movie,
                user_id=selected_user,
                top_n=num_recs
            )

            st.success("Recommendations generated successfully!")

            if isinstance(recommendations, pd.DataFrame):
                st.markdown("### 🎬 Recommended Movies")

                for i, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"#### {row['title']}")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Hybrid Score", round(row["score"], 3))

                        with col2:
                            st.metric("Content Score", round(row["content_score"], 3))

                        with col3:
                            st.metric("User Preference", round(row["user_preference_score"], 3))

                        with col4:
                            st.metric("Sentiment", round(row["sentiment_score"], 3))

                        st.write(f"**Genres:** {row['genres']}")

                        if "explanation" in recommendations.columns:
                            st.info(row["explanation"])

                        st.markdown("---")

                if "score" in recommendations.columns:
                    fig = px.bar(
                        recommendations,
                        x="title",
                        y="score",
                        title="Hybrid Recommendation Scores"
                    )

                    fig.update_layout(xaxis_tickangle=-45)

                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.write(recommendations)

        except TypeError as e:
            st.error("Your hybrid_recommend function has different parameters.")
            st.write("Expected:")
            st.code("hybrid_recommend(movie_title, user_id, top_n)")
            st.code(str(e))

        except Exception as e:
            st.error("Error while generating recommendations.")
            st.code(str(e))