"""
Sentiment Analysis GUI Application using Tkinter
Displays reviews with sentiment analysis and visual feedback
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from src.sentiment_analyzer import SentimentAnalyzer
from src.preprocessor import preprocess_review
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter


class SentimentAnalysisApp:
    """Main GUI Application for Sentiment Analysis"""
    
    def __init__(self, root, csv_path='data/customer_reviews.csv'):
        """
        Initialize the application
        
        Args:
            root: Tkinter root window
            csv_path (str): Path to CSV file with reviews
        """
        self.root = root
        self.root.title("Sentiment Analysis Dashboard")
        self.root.geometry("1200x800")
        
        # Initialize analyzer
        try:
            self.analyzer = SentimentAnalyzer()
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            return
        
        # Load data
        self.csv_path = csv_path
        self.reviews_df = None
        self.current_index = 0
        self.load_data()
        
        # Create UI
        self.setup_ui()
        self.update_display()
    
    def load_data(self):
        """Load reviews from CSV"""
        try:
            self.reviews_df = pd.read_csv(self.csv_path, encoding='ISO-8859-1')
            if self.reviews_df.empty:
                messagebox.showerror("Error", "CSV file is empty")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="📊 Sentiment Analysis Dashboard",
            font=("Helvetica", 18, "bold")
        )
        title_label.pack(pady=10)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Review Analysis
        self.review_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.review_tab, text="Review Analysis")
        self.setup_review_tab()
        
        # Tab 2: Statistics
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Statistics")
        self.setup_stats_tab()
        
        # Tab 3: Batch Analysis
        self.batch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_tab, text="Batch Analysis")
        self.setup_batch_tab()
    
    def setup_review_tab(self):
        """Setup the review analysis tab"""
        # Review display area
        review_frame = ttk.LabelFrame(self.review_tab, text="Current Review", padding=10)
        review_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Review text
        self.review_text = tk.Text(review_frame, height=6, width=80, wrap=tk.WORD)
        self.review_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sentiment result frame
        result_frame = ttk.Frame(review_frame)
        result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(result_frame, text="Sentiment:", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT)
        self.sentiment_label = ttk.Label(result_frame, text="", font=("Helvetica", 14))
        self.sentiment_label.pack(side=tk.LEFT, padx=10)
        
        self.emoji_label = ttk.Label(result_frame, text="", font=("Helvetica", 20))
        self.emoji_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(result_frame, text="Confidence:", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, padx=(20, 0))
        self.confidence_label = ttk.Label(result_frame, text="", font=("Helvetica", 12))
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        
        # Navigation frame
        nav_frame = ttk.Frame(review_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.index_label = ttk.Label(nav_frame, text="", font=("Helvetica", 10))
        self.index_label.pack(side=tk.LEFT)
        
        ttk.Button(nav_frame, text="◀ Previous", command=self.prev_review).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_review).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Go to Review...", command=self.goto_review).pack(side=tk.LEFT, padx=5)
    
    def setup_stats_tab(self):
        """Setup the statistics tab"""
        self.stats_frame = ttk.Frame(self.stats_tab)
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure for charts
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.stats_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update statistics
        self.update_statistics()
    
    def setup_batch_tab(self):
        """Setup the batch analysis tab"""
        # Instructions
        instructions = ttk.Label(
            self.batch_tab,
            text="Upload a CSV file to analyze multiple reviews at once",
            font=("Helvetica", 10)
        )
        instructions.pack(padx=10, pady=10)
        
        # Button frame
        button_frame = ttk.Frame(self.batch_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="📁 Load CSV File", command=self.load_batch_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📊 Analyze All Reviews", command=self.analyze_all_reviews).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💾 Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(self.batch_tab, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview for results
        columns = ('Review', 'Sentiment', 'Confidence')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, height=15)
        self.results_tree.column("#0", width=50)
        self.results_tree.column("Review", width=400)
        self.results_tree.column("Sentiment", width=100)
        self.results_tree.column("Confidence", width=100)
        
        self.results_tree.heading("#0", text="ID")
        self.results_tree.heading("Review", text="Review Text")
        self.results_tree.heading("Sentiment", text="Sentiment")
        self.results_tree.heading("Confidence", text="Confidence")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.batch_results = None
    
    def update_display(self):
        """Update the review display"""
        if self.reviews_df is None or self.reviews_df.empty:
            return
        
        review = self.reviews_df.iloc[self.current_index]['Detailed Review']
        
        # Display review
        self.review_text.config(state=tk.NORMAL)
        self.review_text.delete("1.0", tk.END)
        self.review_text.insert("1.0", review)
        self.review_text.config(state=tk.DISABLED)
        
        # Analyze sentiment
        result = self.analyzer.analyze(review)
        self.sentiment_label.config(text=result['sentiment'].upper())
        self.emoji_label.config(text=result['emoji'])
        self.confidence_label.config(text=f"{result['confidence']:.2%}")
        
        # Update index label
        total = len(self.reviews_df)
        self.index_label.config(text=f"Review {self.current_index + 1} of {total}")
    
    def next_review(self):
        """Go to next review"""
        if self.reviews_df is not None and not self.reviews_df.empty:
            self.current_index = (self.current_index + 1) % len(self.reviews_df)
            self.update_display()
    
    def prev_review(self):
        """Go to previous review"""
        if self.reviews_df is not None and not self.reviews_df.empty:
            self.current_index = (self.current_index - 1) % len(self.reviews_df)
            self.update_display()
    
    def goto_review(self):
        """Jump to a specific review"""
        if self.reviews_df is None:
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Go to Review")
        dialog.geometry("300x100")
        
        ttk.Label(dialog, text=f"Enter review number (1-{len(self.reviews_df)}):").pack(padx=10, pady=10)
        
        entry = ttk.Entry(dialog)
        entry.pack(padx=10, pady=5)
        
        def goto():
            try:
                num = int(entry.get())
                if 1 <= num <= len(self.reviews_df):
                    self.current_index = num - 1
                    self.update_display()
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", f"Please enter a number between 1 and {len(self.reviews_df)}")
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number")
        
        ttk.Button(dialog, text="Go", command=goto).pack(padx=10, pady=5)
    
    def update_statistics(self):
        """Update statistics and charts"""
        if self.reviews_df is None or self.reviews_df.empty:
            return
        
        # Analyze all reviews if not already done
        sentiments = []
        for review in self.reviews_df['Detailed Review']:
            result = self.analyzer.analyze(review)
            sentiments.append(result['sentiment'])
        
        # Count sentiments
        sentiment_counts = Counter(sentiments)
        sentiments_list = ['positive', 'neutral', 'negative']
        counts = [sentiment_counts.get(s, 0) for s in sentiments_list]
        
        # Pie chart
        self.ax1.clear()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        self.ax1.pie(counts, labels=sentiments_list, autopct='%1.1f%%', colors=colors, startangle=90)
        self.ax1.set_title("Sentiment Distribution")
        
        # Bar chart
        self.ax2.clear()
        self.ax2.bar(sentiments_list, counts, color=colors)
        self.ax2.set_title("Sentiment Counts")
        self.ax2.set_ylabel("Number of Reviews")
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def load_batch_file(self):
        """Load a CSV file for batch analysis"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.batch_reviews = pd.read_csv(file_path, encoding='ISO-8859-1')
                messagebox.showinfo("Success", f"Loaded {len(self.batch_reviews)} reviews")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def analyze_all_reviews(self):
        """Analyze all reviews in the current dataset"""
        if not hasattr(self, 'batch_reviews') or self.batch_reviews is None:
            self.batch_reviews = self.reviews_df
        
        if self.batch_reviews is None or self.batch_reviews.empty:
            messagebox.showwarning("Warning", "No reviews to analyze")
            return
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Analyze each review
        results = []
        for idx, review in enumerate(self.batch_reviews['Detailed Review'], 1):
            result = self.analyzer.analyze(review)
            results.append(result)
            
            # Add to treeview
            self.results_tree.insert(
                "",
                tk.END,
                text=str(idx),
                values=(
                    review[:50] + "..." if len(str(review)) > 50 else review,
                    result['sentiment'].upper(),
                    f"{result['confidence']:.2%}"
                )
            )
        
        self.batch_results = results
        messagebox.showinfo("Success", f"Analyzed {len(results)} reviews")
    
    def export_results(self):
        """Export analysis results to CSV"""
        if self.batch_results is None:
            messagebox.showwarning("Warning", "No analysis results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                export_df = pd.DataFrame(self.batch_results)
                export_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
