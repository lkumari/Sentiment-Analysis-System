"""
Run the Sentiment Analysis GUI Application
Make sure to run scripts/train_model.py first to train the model
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gui_app import SentimentAnalysisApp

def main():
    """Main entry point for the GUI application"""
    # Check if model files exist
    if not os.path.exists('models/sentiment_model.pkl') or not os.path.exists('models/vectorizer.pkl'):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Error",
            "Model files not found!\n\n"
            "Please run the following command first:\n"
            "python scripts/train_model.py"
        )
        root.destroy()
        return
    
    # Create and run the app
    root = tk.Tk()
    app = SentimentAnalysisApp(root, csv_path='data/customer_reviews.csv')
    root.mainloop()

if __name__ == "__main__":
    main()
