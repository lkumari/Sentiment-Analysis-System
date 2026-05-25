"""
Automation Module for Processing Incoming Reviews
Handles automatic detection and processing of new reviews
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_analyzer import SentimentAnalyzer


class ReviewProcessor:
    """
    Processes incoming reviews and updates them with sentiment analysis
    """
    
    def __init__(self, analyzer=None):
        """
        Initialize the review processor
        
        Args:
            analyzer: SentimentAnalyzer instance (creates new one if None)
        """
        self.analyzer = analyzer or SentimentAnalyzer()
        self.processed_reviews = []
    
    def process_csv_file(self, csv_path, output_path=None):
        """
        Process all reviews in a CSV file
        
        Args:
            csv_path (str): Path to input CSV file
            output_path (str): Path to save output (if None, overwrites input)
            
        Returns:
            pd.DataFrame: DataFrame with added sentiment and timestamp columns
        """
        # Load CSV
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        
        print(f"\nProcessing {len(df)} reviews from {csv_path}...")
        
        # Add columns for sentiment analysis
        sentiments = []
        confidences = []
        emojis = []
        timestamps = []
        
        for idx, review in enumerate(df['Detailed Review'], 1):
            result = self.analyzer.analyze(review)
            sentiments.append(result['sentiment'])
            confidences.append(result['confidence'])
            emojis.append(result['emoji'])
            timestamps.append(datetime.now().isoformat())
            
            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(df)} reviews...")
        
        # Add columns to dataframe
        df['Predicted_Sentiment'] = sentiments
        df['Confidence'] = confidences
        df['Emoji'] = emojis
        df['Processed_Timestamp'] = timestamps
        
        # Save results
        save_path = output_path or csv_path
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"✓ Results saved to {save_path}")
        
        return df
    
    def detect_negative_reviews(self, df, threshold=0.7):
        """
        Detect reviews with negative sentiment
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            threshold (float): Confidence threshold for detection
            
        Returns:
            pd.DataFrame: DataFrame of negative reviews above threshold
        """
        negative = df[
            (df['Predicted_Sentiment'] == 'negative') & 
            (df['Confidence'] >= threshold)
        ]
        return negative
    
    def generate_alert_notification(self, negative_reviews, threshold=0.7):
        """
        Generate notification alert for negative reviews
        
        Args:
            negative_reviews (pd.DataFrame): DataFrame of negative reviews
            threshold (float): Confidence threshold
            
        Returns:
            str: Alert message
        """
        count = len(negative_reviews)
        
        if count == 0:
            return "✓ No negative reviews detected."
        
        alert = f"""
        ⚠️  ALERT: Negative Reviews Detected
        {'='*50}
        Total negative reviews: {count}
        Confidence threshold: {threshold:.0%}
        
        Negative Reviews:
        """
        
        for idx, row in negative_reviews.iterrows():
            alert += f"\n  • {row['Detailed Review'][:50]}..."
            alert += f"\n    Confidence: {row['Confidence']:.2%}"
        
        return alert
    
    def generate_summary_report(self, df):
        """
        Generate a summary report of sentiment analysis
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            
        Returns:
            str: Summary report
        """
        total = len(df)
        positive = len(df[df['Predicted_Sentiment'] == 'positive'])
        neutral = len(df[df['Predicted_Sentiment'] == 'neutral'])
        negative = len(df[df['Predicted_Sentiment'] == 'negative'])
        
        avg_confidence = df['Confidence'].mean()
        
        report = f"""
        📊 SENTIMENT ANALYSIS SUMMARY REPORT
        {'='*50}
        Total Reviews Analyzed: {total}
        
        Sentiment Distribution:
        ├─ Positive: {positive} ({positive/total*100:.1f}%)
        ├─ Neutral:  {neutral} ({neutral/total*100:.1f}%)
        └─ Negative: {negative} ({negative/total*100:.1f}%)
        
        Model Confidence:
        └─ Average: {avg_confidence:.2%}
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        {'='*50}
        """
        
        return report


def process_incoming_reviews(input_csv, output_csv=None):
    """
    Main function to process incoming reviews
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to save output (optional)
    """
    print("\n" + "="*60)
    print("AUTOMATED REVIEW PROCESSOR")
    print("="*60)
    
    # Initialize processor
    processor = ReviewProcessor()
    
    # Process reviews
    df = processor.process_csv_file(input_csv, output_csv)
    
    # Generate reports
    print("\n" + processor.generate_summary_report(df))
    
    # Check for negative reviews
    negative_reviews = processor.detect_negative_reviews(df, threshold=0.6)
    if len(negative_reviews) > 0:
        print(processor.generate_alert_notification(negative_reviews, threshold=0.6))
    
    return df


if __name__ == "__main__":
    # Example usage
    process_incoming_reviews(
        'data/customer_reviews.csv',
        'data/customer_reviews_processed.csv'
    )
