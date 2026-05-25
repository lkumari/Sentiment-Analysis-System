"""
Model Training Module for Sentiment Analysis
Handles training, saving, and loading of sentiment classification models
"""

import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from src.preprocessor import batch_preprocess


class SentimentClassifier:
    """
    Sentiment Classification Model using Naive Bayes
    """
    
    def __init__(self, model_type='multinomial', vectorizer_type='count'):
        """
        Initialize the sentiment classifier
        
        Args:
            model_type (str): 'gaussian' or 'multinomial' (default: 'multinomial')
            vectorizer_type (str): 'count' or 'tfidf' (default: 'count')
        """
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        
        # Initialize vectorizer
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 3))
        else:  # tfidf
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        
        # Initialize classifier
        if model_type == 'gaussian':
            self.classifier = GaussianNB()
        else:  # multinomial
            self.classifier = MultinomialNB()
    
    def train(self, reviews, labels, test_size=0.2, random_state=42):
        """
        Train the sentiment classifier
        
        Args:
            reviews (list): List of preprocessed review texts
            labels (list): List of sentiment labels
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Training metrics and results
        """
        # Vectorize the reviews
        X = self.vectorizer.fit_transform(reviews)
        
        # Convert to dense array if using Gaussian NB
        if self.model_type == 'gaussian':
            X = X.toarray()
        
        # Convert string labels to numeric (positive=2, neutral=1, negative=0)
        label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        y = np.array([label_map.get(label.lower(), 1) for label in labels])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate metrics
        reverse_label_map = {v: k for k, v in label_map.items()}
        target_names = [reverse_label_map[i] for i in sorted(label_map.values())]
        
        print(f"\n{'='*60}")
        print(f"Model Training Results ({self.model_type.upper()} + {self.vectorizer_type.upper()})")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:\n")
        print(classification_report(y_test, y_pred, target_names=target_names))
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(f"{'='*60}\n")
        
        return {
            'accuracy': accuracy,
            'test_size': test_size,
            'model_type': self.model_type,
            'vectorizer_type': self.vectorizer_type
        }
    
    def predict(self, text):
        """
        Predict sentiment of a single review
        
        Args:
            text (str): Review text
            
        Returns:
            str: Predicted sentiment ('positive', 'neutral', or 'negative')
        """
        X = self.vectorizer.transform([text])
        if self.model_type == 'gaussian':
            X = X.toarray()
        
        prediction = self.classifier.predict(X)[0]
        
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return sentiment_map.get(prediction, 'neutral')
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple reviews
        
        Args:
            texts (list): List of review texts
            
        Returns:
            list: List of predicted sentiments
        """
        return [self.predict(text) for text in texts]
    
    def save_model(self, model_path='models/sentiment_model.pkl', 
                   vectorizer_path='models/vectorizer.pkl'):
        """
        Save the trained model and vectorizer
        
        Args:
            model_path (str): Path to save the model
            vectorizer_path (str): Path to save the vectorizer
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path='models/sentiment_model.pkl',
                   vectorizer_path='models/vectorizer.pkl'):
        """
        Load a trained model and vectorizer
        
        Args:
            model_path (str): Path to the saved model
            vectorizer_path (str): Path to the saved vectorizer
        """
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"Model loaded from {model_path}")
        print(f"Vectorizer loaded from {vectorizer_path}")


def train_sentiment_model(csv_path='data/customer_reviews.csv', 
                         model_path='models/sentiment_model.pkl',
                         vectorizer_path='models/vectorizer.pkl'):
    """
    Complete pipeline to train and save a sentiment model
    
    Args:
        csv_path (str): Path to the CSV file with reviews
        model_path (str): Path to save the model
        vectorizer_path (str): Path to save the vectorizer
    """
    # Load data
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    
    print("Loading and preprocessing reviews...")
    reviews = batch_preprocess(df['Detailed Review'].tolist())
    labels = df['Sentiment'].tolist()
    
    # Initialize and train classifier
    classifier = SentimentClassifier(model_type='multinomial', vectorizer_type='count')
    classifier.train(reviews, labels)
    
    # Save the model
    classifier.save_model(model_path, vectorizer_path)
    
    return classifier
