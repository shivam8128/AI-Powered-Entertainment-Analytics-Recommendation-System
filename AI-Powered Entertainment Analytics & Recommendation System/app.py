import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Analytics UI", page_icon="🎬", layout="wide")

@st.cache_data
def load_movies():
    try:
        with open('models/movies_df.pkl', 'rb') as f:
            movies = pickle.load(f)
        return movies
    except:
        return None

@st.cache_resource
def load_models():
    try:
        with open('models/tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open('models/sentiment_model.pkl', 'rb') as f:
            sent_model = pickle.load(f)
        with open('models/sentiment_vectorizer.pkl', 'rb') as f:
            sent_vec = pickle.load(f)
        return tfidf_matrix, sent_model, sent_vec
    except:
        return None, None, None

movies = load_movies()
tfidf_matrix, sent_model, sent_vec = load_models()

st.title("🎬 Entertainment Analytics & Recommendation System")
st.markdown("Powered by **Real World Datasets** (MovieLens, IMDb, Spotify)")

tab1, tab2, tab3 = st.tabs(["Recommendation System", "Sentiment Analysis", "Dashboard Overview"])

with tab1:
    st.header("Movie Recommendation System (MovieLens)")
    if movies is not None and tfidf_matrix is not None:
        movie_titles = movies['title'].tolist()
        selected_movie = st.selectbox("Select a Movie:", movie_titles)
        if st.button("Get Recommendations"):
            idx = movies.index[movies['title'] == selected_movie].tolist()[0]
            cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
            sim_scores = list(enumerate(cosine_sim))
            sim_scores.sort(key=lambda x: float(x[1]), reverse=True)
            sim_scores = sim_scores[1:6]
            recs = movies.iloc[[i[0] for i in sim_scores]][['title', 'genres']]
            st.table(recs)
    else:
        st.warning("Models not found. Run download_data and run_all.bat first.")

with tab2:
    st.header("Analyze Review Sentiment (IMDb)")
    review_text = st.text_area("Enter a movie review snippet:")
    if st.button("Analyze Sentiment") and sent_model:
        vec = sent_vec.transform([review_text])
        prediction = sent_model.predict(vec)[0]
        if prediction == 1:
            st.success("Positive Sentiment 😊")
        else:
            st.error("Negative Sentiment 😞")

with tab3:
    st.header("Analytics Overview")
    if movies is not None:
        st.write("Genres Distribution:")
        genres_expanded = movies['genres'].str.split('|', expand=True).stack()
        st.bar_chart(genres_expanded.value_counts())
