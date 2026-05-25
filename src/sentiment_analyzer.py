"""
Sentiment Analyzer Module
Main interface for sentiment analysis
"""

import pickle
import os
from src.preprocessor import preprocess_review


class SentimentAnalyzer:
    """
    Main sentiment analyzer that loads pre-trained model and vectorizer
    """
    
    def __init__(self, model_path='models/sentiment_model.pkl',
                 vectorizer_path='models/vectorizer.pkl'):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_path (str): Path to the saved model
            vectorizer_path (str): Path to the saved vectorizer
        """
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.emoji_map = {
            'positive': '😊',
            'neutral': '😐',
            'negative': '😞'
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model and vectorizer"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"Please train the model first using model_trainer.py"
            )
        
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer file not found at {self.vectorizer_path}. "
                f"Please train the model first using model_trainer.py"
            )
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def analyze(self, text):
        """
        Analyze the sentiment of a text
        
        Args:
            text (str): Review text to analyze
            
        Returns:
            dict: Contains sentiment, emoji, and confidence info
        """
        if not text or not isinstance(text, str):
            return {
                'sentiment': 'neutral',
                'emoji': self.emoji_map['neutral'],
                'confidence': 0.0
            }
        
        # Preprocess the text
        processed = preprocess_review(text)
        
        if not processed:
            return {
                'sentiment': 'neutral',
                'emoji': self.emoji_map['neutral'],
                'confidence': 0.0
            }
        
        # Vectorize
        X = self.vectorizer.transform([processed])
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        
        # Get probability scores if available
        try:
            probabilities = self.model.predict_proba(X)[0]
            confidence = max(probabilities)
        except:
            confidence = 0.0
        
        sentiment = self.label_map.get(prediction, 'neutral')
        
        return {
            'sentiment': sentiment,
            'emoji': self.emoji_map.get(sentiment, self.emoji_map['neutral']),
            'confidence': float(confidence)
        }
    
    def analyze_batch(self, texts):
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts (list): List of review texts
            
        Returns:
            list: List of analysis results
        """
        return [self.analyze(text) for text in texts]
