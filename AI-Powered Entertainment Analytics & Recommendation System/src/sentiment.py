import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
from datasets import load_dataset

def train_sentiment_model():
    print("Loading Real IMDB Dataset via HuggingFace...")
    dataset = load_dataset("imdb")
    # Take a 10000 sample subset strictly to keep runtimes extremely fast yet real
    train_data = dataset['train'].shuffle(seed=42).select(range(10000))
    test_data = dataset['test'].shuffle(seed=42).select(range(2000))
    
    X_train = train_data['text']
    y_train = train_data['label']  # 1: pos, 0: neg
    X_test = test_data['text']
    y_test = test_data['label']
    
    print("Vectorizing Text Data...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Sentiment Analysis Model Accuracy on Real IMDB test data: {acc * 100:.2f}%")
    
    os.makedirs('models', exist_ok=True)
    with open('models/sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/sentiment_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer

if __name__ == "__main__":
    train_sentiment_model()
