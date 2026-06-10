import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender System")
st.markdown("Hybrid recommender using **Content-Based + SVD Collaborative Filtering**")

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                          names=['user_id','movie_id','rating','timestamp'])
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                         usecols=[0,1], names=['movie_id','title'])
    movies_full = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                              names=['movie_id','title','release_date','video_date','url',
                                     'unknown','Action','Adventure','Animation','Childrens',
                                     'Comedy','Crime','Documentary','Drama','Fantasy',
                                     'FilmNoir','Horror','Musical','Mystery','Romance',
                                     'SciFi','Thriller','War','Western'])
    return ratings, movies, movies_full

@st.cache_data
def clean_data(ratings, movies):
    df = ratings.merge(movies, on='movie_id')
    movie_counts = df.groupby('movie_id')['rating'].count()
    df = df[df['movie_id'].isin(movie_counts[movie_counts >= 10].index)]
    user_counts = df.groupby('user_id')['rating'].count()
    df = df[df['user_id'].isin(user_counts[user_counts >= 10].index)]
    return df

@st.cache_resource
def build_content_model(movies_full):
    genre_cols = ['Action','Adventure','Animation','Childrens','Comedy','Crime',
                  'Documentary','Drama','Fantasy','FilmNoir','Horror','Musical',
                  'Mystery','Romance','SciFi','Thriller','War','Western']
    movies_full['genres'] = movies_full.apply(
        lambda row: ' '.join([g for g in genre_cols if row[g] == 1]), axis=1)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies_full['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies_full.index, index=movies_full['title'])
    return cosine_sim, indices, movies_full

@st.cache_resource
def build_svd_model(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id','movie_id','rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    svd = SVD(n_factors=100, random_state=42)
    svd.fit(trainset)
    return svd

# ── Build everything ─────────────────────────────────────────
with st.spinner("Loading data and training models... (first run takes ~30 seconds)"):
    ratings, movies, movies_full = load_data()
    df = clean_data(ratings, movies)
    cosine_sim, indices, movies_full = build_content_model(movies_full)
    svd_model = build_svd_model(df)

st.success("✅ Models ready!")

# ── Recommendation functions ──────────────────────────────────
def get_similar_movies(title, n=20):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    titles = movies_full['title'].iloc[[i[0] for i in sim_scores]].tolist()
    seen = set()
    unique_titles = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            unique_titles.append(t)
    return unique_titles
def hybrid_recommendations(user_id, title, n=10):
    content_recs = get_similar_movies(title, n=20)
    scored = []
    for movie_title in content_recs:
        movie_row = movies_full[movies_full['title'] == movie_title]
        if movie_row.empty:
            continue
        movie_id = movie_row.iloc[0]['movie_id']
        predicted_rating = svd_model.predict(user_id, movie_id).est
        scored.append((movie_title, round(predicted_rating, 2)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]

# ── UI ────────────────────────────────────────────────────────
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input("👤 Enter User ID", min_value=1, max_value=943, value=1)
with col2:
    movie_list = sorted(movies_full['title'].tolist())
    selected_movie = st.selectbox("🎥 Select a Movie you like", movie_list)

if st.button("🚀 Get Recommendations", use_container_width=True):
    results = hybrid_recommendations(user_id, selected_movie)
    if results:
        st.markdown(f"### 🍿 Top Picks for User {user_id}")
        for i, (title, score) in enumerate(results, 1):
            stars = "⭐" * round(score)
            st.markdown(f"**{i}. {title}** — {score} {stars}")
    else:
        st.warning("No recommendations found. Try a different movie.")