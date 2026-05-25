"""
Train and save the sentiment analysis model
Run this script to train the model on customer_reviews.csv
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import train_sentiment_model

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("="*60)
    
    # Train the model
    model = train_sentiment_model(
        csv_path='data/customer_reviews.csv',
        model_path='models/sentiment_model.pkl',
        vectorizer_path='models/vectorizer.pkl'
    )
    
    print("\n✓ Model training completed successfully!")
    print("✓ Model saved to: models/sentiment_model.pkl")
    print("✓ Vectorizer saved to: models/vectorizer.pkl")
    print("\nYou can now run the GUI application with: python run_gui.py")
    print("="*60 + "\n")
