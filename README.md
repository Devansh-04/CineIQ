# 🎬 CINEIQ — Explainable Hybrid Movie Recommendation System

CINEIQ is a hybrid movie recommendation system that recommends movies using content similarity, user preference patterns, rating signals, popularity, and sentiment — and explains the reason behind every recommendation.

---

## 📁 Project Structure

```
## 2. Project Structure

```text
cineiq/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_content_based.ipynb
│   ├── 03_collaborative_filtering.ipynb
│   ├── 04_svd_model.ipynb
│   ├── 05_hybrid_recommender.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── content_recommender.py
│   ├── collaborative_recommender.py
│   ├── svd_recommender.py
│   ├── hybrid_recommender.py
│   ├── sentiment.py
│   ├── explainability.py
│
├── api/
│   ├── main.py
│
├── dashboard/
│   ├── app.py
│
├── models/
│
├── requirements.txt
└── README.md
```

---

## ✨ Features

### 3.1 Data Loading and Cleaning
Raw MovieLens data is loaded, cleaned, and saved to:
- `data/processed/movies_clean.csv` — contains `movieId`, `title`, `genres`, `genres_clean`, `year`
- `data/processed/ratings_clean.csv`

### 3.2 Content-Based Recommendation
Finds movies similar to a selected title using:
- TF-IDF vectorization on genre and title text
- Cosine similarity

### 3.3 Collaborative Filtering
Recommends movies based on shared rating patterns across users — users who rated similar movies in the past may like similar movies in the future.

### 3.4 SVD Recommendation
Uses matrix factorization (SVD) to learn hidden patterns between users and movies from ratings data.

### 3.5 Hybrid Recommendation
Combines multiple signals using weighted scoring:

| Signal | Weight |
|--------|--------|
| Content Similarity Score | 40% |
| User Genre Preference Score | 25% |
| Average Rating Score | 20% |
| Popularity Score | 10% |
| Sentiment Score | 5% |

**Hybrid Score** = `0.40 × Content + 0.25 × Preference + 0.20 × Rating + 0.10 × Popularity + 0.05 × Sentiment`

### 3.6 Sentiment and Explainability
Every recommendation includes a generated explanation, for example:

> *"Recommended because it is content-wise similar to the selected movie, matches the user's preferred genres, and has strong audience rating signals."*

---

## 📊 Dashboard

Run the Streamlit dashboard to interactively explore recommendations.

**Features:**
- User summary (movies rated, avg/highest/lowest rating)
- Genre radar chart
- Most watched genres
- Average rating by genre
- Decade preference
- Rating history
- Hybrid movie recommendations with explanations

---

## ⚡ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check API status |
| GET | `/movies?limit=10` | List movies |
| GET | `/users?limit=10` | List user IDs |
| GET | `/recommend?user_id=1&movie_title=Toy%20Story%20(1995)&top_n=10` | Hybrid recommendations |
| GET | `/similar?movie_title=Toy%20Story%20(1995)&top_n=10` | Content-similar movies |
| GET | `/user-profile?user_id=1` | User taste profile |

Interactive API docs available at: `http://127.0.0.1:8000/docs`

---

## 🛠️ Installation

```bash
# 1. Clone or download the project and navigate to the folder
cd cineiq/

# 2. Create and activate a conda environment
conda create -n cineiq python=3.10
conda activate cineiq

# 3. Install dependencies
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, install manually:

```bash
pip install pandas numpy scikit-learn streamlit plotly fastapi uvicorn vaderSentiment scikit-surprise
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
scikit-surprise
streamlit
plotly
fastapi
uvicorn
vaderSentiment
```

---

## 🗂️ Dataset Setup

1. Download the [MovieLens dataset](https://grouplens.org/datasets/movielens/) and place the raw files in `data/raw/`:
   - `movies.csv`
   - `ratings.csv`
   - `links.csv`
   - `tags.csv`

2. Run the data cleaning notebook:
   ```
   notebooks/01_data_cleaning.ipynb
   ```

   This generates:
   - `data/processed/movies_clean.csv`
   - `data/processed/ratings_clean.csv`

---

## 🚀 Running the Project

> ⚠️ Always run commands from the main project folder: `cineiq/`

### Start the Dashboard
```bash
streamlit run dashboard/app.py
```
Opens at: `http://localhost:8501`

### Start the API
```bash
uvicorn api.main:app --reload
```
Runs at: `http://127.0.0.1:8000`  
API docs: `http://127.0.0.1:8000/docs`

---

## 🧰 Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Scikit-Surprise` · `Streamlit` · `Plotly` · `FastAPI` · `Uvicorn` · `VADER Sentiment`

---

## 🔮 Future Improvements

- [ ] Add movie posters via TMDB / IMDb IDs
- [ ] Improve sentiment scoring using DistilBERT
- [ ] Persist TF-IDF and SVD models in `models/` folder
- [ ] Add MLflow experiment tracking
- [ ] Add user login system and personalized watchlist
- [ ] Improve dashboard UI
- [ ] Deploy Streamlit dashboard online
- [ ] Deploy FastAPI backend online
