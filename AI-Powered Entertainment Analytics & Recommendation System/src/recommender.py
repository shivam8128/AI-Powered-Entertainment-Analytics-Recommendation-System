import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import os

def build_recommender():
    try:
        movies = pd.read_csv('data/ml-latest-small/movies.csv')
    except Exception as e:
        print("Required CSV files not found. Run data/download_data.py first.")
        return

    # Use TF-IDF on genre for content based filtering
    movies['genre_clean'] = movies['genres'].str.replace('|', ' ')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genre_clean'].fillna(''))

    os.makedirs('models', exist_ok=True)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('models/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open('models/movies_df.pkl', 'wb') as f:
        pickle.dump(movies, f)
    
    print("Recommendation engine components saved to models/")

if __name__ == "__main__":
    build_recommender()
