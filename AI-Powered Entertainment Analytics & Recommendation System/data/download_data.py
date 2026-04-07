import os
import requests
import zipfile
import pandas as pd

def download_file(url, local_filename):
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return local_filename

def setup_datasets():
    os.makedirs('data', exist_ok=True)
    
    # 1. MovieLens
    ml_url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    ml_zip = "data/ml-latest-small.zip"
    if not os.path.exists("data/ml-latest-small/movies.csv"):
        download_file(ml_url, ml_zip)
        print("Extracting MovieLens...")
        with zipfile.ZipFile(ml_zip, 'r') as zip_ref:
            zip_ref.extractall("data")
        os.remove(ml_zip)
        print("MovieLens ready.")
    else:
        print("MovieLens already exists.")

    # 2. Spotify Popularity Dataset
    spotify_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
    spotify_csv = "data/spotify_songs.csv"
    if not os.path.exists(spotify_csv):
        print("Downloading Spotify dataset...")
        df = pd.read_csv(spotify_url)
        df.to_csv(spotify_csv, index=False)
        print("Spotify dataset ready.")
    else:
        print("Spotify dataset already exists.")
        
    print("All file-based real datasets downloaded! (IMDB will be fetched via HuggingFace Datasets array)")

if __name__ == "__main__":
    setup_datasets()
