"""
Financial Sentiment Analysis AI
Professional-grade financial sentiment analysis system

A production-ready, backend/NLP-focused financial sentiment analysis system 
designed for professional deployment with real-time news analysis, 
dual-model sentiment scoring, and comprehensive monitoring.
"""

__version__ = "1.0.0"
__author__ = "Financial Sentiment AI Team"
__description__ = "Professional financial sentiment analysis with VADER + FinBERT"

# Main components
from .core import (
    FinancialNewsFetcher, 
    FinancialTextCleaner,
    FinancialVaderAnalyzer,
    FinBERTAnalyzer
)

from .pipeline import (
    FinancialSentimentPipeline,
    FinancialDataLogger
)

# Convenience functions
from .core.news_fetcher import fetch_financial_news
from .core.text_cleaner import clean_financial_text
from .core.sentiment_vader import analyze_vader_sentiment
from .core.sentiment_finbert_clean import analyze_finbert
from .pipeline.inference import run_live_analysis, analyze_texts

__all__ = [
    # Core classes
    "FinancialNewsFetcher",
    "FinancialTextCleaner", 
    "FinancialVaderAnalyzer",
    "FinBERTAnalyzer",
    
    # Pipeline classes
    "FinancialSentimentPipeline",
    "FinancialDataLogger",
    
    # Convenience functions
    "fetch_financial_news",
    "clean_financial_text",
    "analyze_vader_sentiment", 
    "analyze_finbert",
    "run_live_analysis",
    "analyze_texts"
]

# Package metadata
__title__ = "financial-sentiment-ai"
__license__ = "MIT"
__copyright__ = "2024 Financial Sentiment AI Team"
