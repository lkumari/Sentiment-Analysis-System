"""
Text Preprocessing Module for Sentiment Analysis
Handles cleaning and preprocessing of customer reviews
"""

import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


def preprocess_review(text):
    """
    Preprocess a review by:
    - Removing HTML tags
    - Removing special characters
    - Converting to lowercase
    - Tokenizing
    - Lemmatizing
    - Removing stopwords
    
    Args:
        text (str): Raw review text
        
    Returns:
        str: Preprocessed review
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    
    # Remove special characters (keep only alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = text.split()
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word, 'v') for word in tokens 
              if word not in stop_words and len(word) > 2]
    
    return " ".join(tokens)


def batch_preprocess(reviews):
    """
    Preprocess a batch of reviews
    
    Args:
        reviews (list): List of review texts
        
    Returns:
        list: List of preprocessed reviews
    """
    return [preprocess_review(review) for review in reviews]
