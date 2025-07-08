"""
Financial Sentiment Analysis AI - Pipeline Module
Orchestration and data management components
"""

__version__ = "1.0.0"

# Pipeline components
from .inference import FinancialSentimentPipeline, run_live_analysis, analyze_texts
from .logger import FinancialDataLogger

__all__ = [
    "FinancialSentimentPipeline",
    "FinancialDataLogger",
    "run_live_analysis", 
    "analyze_texts"
]
