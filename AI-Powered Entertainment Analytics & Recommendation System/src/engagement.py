import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import os

def train_engagement_models():
    try:
        spotify_df = pd.read_csv('data/spotify_songs.csv')
    except Exception as e:
        print("Data files not found. Generate them first.")
        return

    spotify_df.dropna(inplace=True)

    print("Training Popularity Predictor using Real Spotify Data...")
    features_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    X = spotify_df[features_cols]
    y = spotify_df['track_popularity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    preds = rf_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"Popularity Prediction R^2 Score: {r2:.4f}")
    
    print("Training Anomaly Detector for Fake/Outlier Engagements...")
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X)
    
    os.makedirs('models', exist_ok=True)
    with open('models/rf_popularity_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('models/isolation_forest.pkl', 'wb') as f:
        pickle.dump(iso_forest, f)

if __name__ == "__main__":
    train_engagement_models()
