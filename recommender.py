import pandas as pd

# Load ratings data
ratings = pd.read_csv('ml-100k/u.data', 
                      sep='\t', 
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Load movies data
movies = pd.read_csv('ml-100k/u.item',
                     sep='|',
                     encoding='latin-1',
                     usecols=[0, 1],
                     names=['movie_id', 'title'])

# Explore the data
print("=== RATINGS DATA ===")
print(ratings.head())

print("\n=== MOVIES DATA ===")
print(movies.head())

print("\n=== BASIC STATS ===")
print("Total ratings:", len(ratings))
print("Total movies:", len(movies))
print("Total users:", ratings['user_id'].nunique())
print("Rating scale:", ratings['rating'].min(), "to", ratings['rating'].max())
# ============================================================
# STEP 2 — CLEAN THE DATA
# ============================================================

# Merge ratings with movie titles
df = ratings.merge(movies, on='movie_id')

# Check for missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Filter out movies with very few ratings (keep movies rated by at least 10 users)
movie_counts = df.groupby('movie_id')['rating'].count()
popular_movies = movie_counts[movie_counts >= 10].index
df = df[df['movie_id'].isin(popular_movies)]

# Filter out users who rated very few movies (keep users who rated at least 10 movies)
user_counts = df.groupby('user_id')['rating'].count()
active_users = user_counts[user_counts >= 10].index
df = df[df['user_id'].isin(active_users)]

# Final stats after cleaning
print("\n=== AFTER CLEANING ===")
print("Remaining ratings:", len(df))
print("Remaining movies:", df['movie_id'].nunique())
print("Remaining users:", df['user_id'].nunique())

# Save cleaned dataframe for later steps
print("\n✅ Data cleaned and ready!")
# ============================================================
# STEP 3 — CONTENT-BASED FILTERING
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load full movies file with genres
movies_full = pd.read_csv('ml-100k/u.item',
                          sep='|',
                          encoding='latin-1',
                          names=['movie_id','title','release_date','video_date','url',
                                 'unknown','Action','Adventure','Animation','Childrens',
                                 'Comedy','Crime','Documentary','Drama','Fantasy',
                                 'FilmNoir','Horror','Musical','Mystery','Romance',
                                 'SciFi','Thriller','War','Western'])

# Combine genre columns into a single string per movie
genre_cols = ['Action','Adventure','Animation','Childrens','Comedy','Crime',
              'Documentary','Drama','Fantasy','FilmNoir','Horror','Musical',
              'Mystery','Romance','SciFi','Thriller','War','Western']

def get_genres(row):
    return ' '.join([genre for genre in genre_cols if row[genre] == 1])

movies_full['genres'] = movies_full.apply(get_genres, axis=1)

# Build TF-IDF matrix
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_full['genres'])

# Compute cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie title to index
indices = pd.Series(movies_full.index, index=movies_full['title'])

# Function to get similar movies
def get_similar_movies(title, n=10):
    if title not in indices:
        return f"Movie '{title}' not found."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # skip the movie itself
    movie_indices = [i[0] for i in sim_scores]
    return movies_full['title'].iloc[movie_indices].tolist()

# Test it
print("\n=== CONTENT-BASED RECOMMENDATIONS ===")
print("Movies similar to 'Toy Story (1995)':")
print(get_similar_movies('Toy Story (1995)'))
# ============================================================
# STEP 4 — COLLABORATIVE FILTERING (SVD)
# ============================================================

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Prepare data for Surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)

# Split into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD model
print("\nTraining SVD model... (may take 30 seconds)")
svd_model = SVD(n_factors=100, random_state=42)
svd_model.fit(trainset)

# Test accuracy
predictions = svd_model.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# Function to get movie recommendations for a user
def get_svd_recommendations(user_id, n=10):
    rated_movies = df[df['user_id'] == user_id]['movie_id'].tolist()
    all_movies = df['movie_id'].unique()
    unrated_movies = [m for m in all_movies if m not in rated_movies]
    
    predictions_list = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
    predictions_list.sort(key=lambda x: x.est, reverse=True)
    
    top_movie_ids = [pred.iid for pred in predictions_list[:n]]
    top_movies = movies[movies['movie_id'].isin(top_movie_ids)]['title'].tolist()
    return top_movies

# Test it
print("\n=== SVD RECOMMENDATIONS FOR USER 1 ===")
print(get_svd_recommendations(user_id=1))
# ============================================================
# STEP 5 — COMBINE BOTH MODELS (HYBRID RECOMMENDER)
# ============================================================

def hybrid_recommendations(user_id, title, n=10):
    print(f"\n🎬 Hybrid Recommendations for User {user_id} based on '{title}'")
    
    # --- Content-Based: movies similar to given title ---
    content_recs = get_similar_movies(title, n=20)
    
    # --- SVD: score those similar movies using SVD ---
    scored = []
    for movie_title in content_recs:
        movie_row = movies_full[movies_full['title'] == movie_title]
        if movie_row.empty:
            continue
        movie_id = movie_row.iloc[0]['movie_id']
        predicted_rating = svd_model.predict(user_id, movie_id).est
        scored.append((movie_title, predicted_rating))
    
    # Sort by predicted rating
    scored.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Movie':<50} {'Predicted Rating'}")
    print("-" * 65)
    for movie, score in scored[:n]:
        print(f"{movie:<50} {score:.2f}")

# Test it
hybrid_recommendations(user_id=1, title='Toy Story (1995)')
hybrid_recommendations(user_id=1, title='Star Wars (1977)')