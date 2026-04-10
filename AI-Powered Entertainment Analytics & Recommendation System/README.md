# Entertainment Analytics & Recommendation System

## Overview
This is a comprehensive Machine Learning project performing collaborative/content-based recommendations, NLP sentiment analysis, popularity prediction, and outlier engagement detection on Movie and Media datasets. Everything is bundled into a Streamlit User Interface.

## Setup Requirements

Ensure Python is installed on your system. 

### 1. Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### 2. Download Data & Train Models
You can easily execute the pipeline using `run_all.bat` (Windows) or `run_all.ps1` (PowerShell). Alternatively, do it manually:
```bash
py data/download_data.py
py src/recommender.py
py src/sentiment.py
py src/engagement.py
```

### 3. Run the Streamlit Application
```bash
py -m streamlit run app.py
```

## Features
- **Data Integrations**: Leverages MovieLens, IMDB, and Spotify datasets.
- **Content Based Recommendation**: using TF-IDF mapping on contextual features.
- **Sentiment Analysis**: using Logistic Regression NLP processing.
- **Popularity Prediction**: Regressions models trained on user interaction metadata.
- **Anomaly Detection**: finding Fake Engagement bottlenecks by applying Isolation Forests over engagement ratios.
