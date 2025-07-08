"""
Clean FinBERT-based sentiment analysis for financial news
Professional implementation with robust error handling and entity extraction
"""

import logging
import re
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Optional dependencies with graceful fallbacks
try:
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification, 
        pipeline,
        logging as transformers_logging
    )
    transformers_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    np = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentLabel(Enum):
    """Enumeration for sentiment labels"""
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"

@dataclass
class FinBERTResult:
    """Clean data structure for FinBERT analysis results"""
    text: str
    sentiment: SentimentLabel
    confidence: float
    scores: Dict[str, float]
    entities: Optional[Dict[str, List[str]]] = None
    processing_time: Optional[float] = None
    model_info: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'text': self.text,
            'sentiment': self.sentiment.value,
            'confidence': self.confidence,
            'scores': self.scores,
            'entities': self.entities or {},
            'processing_time': self.processing_time,
            'model_info': self.model_info or {}
        }

class CleanFinBERTAnalyzer:
    """
    Clean, professional FinBERT sentiment analyzer with robust error handling
    """
    
    def __init__(self, 
                 model_name: str = "ProsusAI/finbert",
                 use_cuda: bool = True,
                 max_length: int = 512):
        """
        Initialize FinBERT analyzer with clean configuration
        
        Args:
            model_name: HuggingFace model identifier for FinBERT
            use_cuda: Whether to use CUDA if available
            max_length: Maximum input length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._setup_device(use_cuda)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.nlp = None
        
        # Model status
        self.is_initialized = False
        self.initialization_error = None
        
        # Financial patterns for entity extraction
        self.financial_patterns = self._setup_financial_patterns()
        
        # Initialize components
        self._initialize_models()
    
    def _setup_device(self, use_cuda: bool) -> Union[int, str]:
        """Setup computation device"""
        if use_cuda and TRANSFORMERS_AVAILABLE and torch and torch.cuda.is_available():
            return 0  # Use first GPU
        return -1  # Use CPU
    
    def _setup_financial_patterns(self) -> Dict[str, re.Pattern]:
        """Setup regex patterns for financial entity extraction"""
        return {
            'tickers': re.compile(r'\b[A-Z]{1,5}\b'),
            'currency': re.compile(r'\$[\d,]+(?:\.\d+)?[KMB]?'),
            'percentage': re.compile(r'\d+(?:\.\d+)?%'),
            'financial_terms': re.compile(
                r'\b(?:earnings|revenue|profit|loss|dividend|merger|acquisition|IPO|SEC|FDA|NYSE|NASDAQ)\b',
                re.IGNORECASE
            )
        }
    
    def _initialize_models(self) -> None:
        """Initialize FinBERT and spaCy models with error handling"""
        try:
            self._initialize_finbert()
            self._initialize_spacy()
            self.is_initialized = True
            logger.info("✅ CleanFinBERT analyzer initialized successfully")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Failed to initialize FinBERT: {e}")
            self.is_initialized = False
    
    def _initialize_finbert(self) -> None:
        """Initialize FinBERT model components"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers torch")
        
        logger.info(f"Loading FinBERT model: {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Create sentiment pipeline
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            return_all_scores=True,
            max_length=self.max_length,
            truncation=True
        )
        
        device_info = "CUDA" if self.device >= 0 else "CPU"
        logger.info(f"FinBERT model loaded on {device_info}")
    
    def _initialize_spacy(self) -> None:
        """Initialize spaCy for entity extraction"""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Entity extraction will be limited.")
            return
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded for entity extraction")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def analyze_sentiment(self, text: str) -> FinBERTResult:
        """
        Analyze sentiment with clean, robust implementation
        
        Args:
            text: Input text to analyze
            
        Returns:
            FinBERTResult object with analysis results
        """
        start_time = time.time()
        
        # Validate input
        if not text or not text.strip():
            return self._create_empty_result(text, "Empty input text")
        
        # Check if model is initialized
        if not self.is_initialized:
            return self._create_fallback_result(text, self.initialization_error)
        
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Run FinBERT analysis
            predictions = self.pipeline(cleaned_text)
            
            # Process results
            sentiment, confidence, scores = self._process_predictions(predictions[0])
            
            # Extract entities
            entities = self._extract_entities(text) if self.nlp else {}
            
            processing_time = time.time() - start_time
            
            # Update performance statistics
            self._update_stats(processing_time, success=True)
            
            return FinBERTResult(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                scores=scores,
                entities=entities,
                processing_time=processing_time,
                model_info={
                    'model_name': self.model_name,
                    'device': 'CUDA' if self.device >= 0 else 'CPU'
                }
            )
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            self._update_stats(time.time() - start_time, success=False)
            return self._create_fallback_result(text, str(e))
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for FinBERT"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Truncate if too long (leave room for special tokens)
        if len(text) > self.max_length - 10:
            text = text[:self.max_length - 10]
        
        return text
    
    def _process_predictions(self, predictions: List[Dict]) -> tuple:
        """Process FinBERT predictions into clean format"""
        # Convert predictions to standardized format
        scores = {}
        for pred in predictions:
            label = pred['label'].upper()
            score = float(pred['score'])
            
            # Map FinBERT labels to standard format
            if label in ['POSITIVE', 'POS']:
                scores['Positive'] = score
            elif label in ['NEGATIVE', 'NEG']:
                scores['Negative'] = score
            elif label in ['NEUTRAL', 'NEU']:
                scores['Neutral'] = score
        
        # Ensure all sentiment categories are present
        for sentiment_type in ['Positive', 'Negative', 'Neutral']:
            if sentiment_type not in scores:
                scores[sentiment_type] = 0.0
        
        # Find primary sentiment
        primary_sentiment_str = max(scores, key=scores.get)
        primary_sentiment = SentimentLabel(primary_sentiment_str)
        confidence = scores[primary_sentiment_str]
        
        return primary_sentiment, confidence, scores
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities from text"""
        entities = {
            'organizations': [],
            'persons': [],
            'money': [],
            'tickers': [],
            'financial_terms': []
        }
        
        try:
            # spaCy NER
            if self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        entities['organizations'].append(ent.text)
                    elif ent.label_ == "PERSON":
                        entities['persons'].append(ent.text)
                    elif ent.label_ == "MONEY":
                        entities['money'].append(ent.text)
            
            # Pattern-based extraction
            for pattern_name, pattern in self.financial_patterns.items():
                matches = pattern.findall(text)
                if pattern_name == 'tickers':
                    # Filter out common words
                    common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAS'}
                    entities['tickers'] = [t for t in matches if t not in common_words]
                elif pattern_name == 'financial_terms':
                    entities['financial_terms'] = matches
            
            # Remove duplicates and clean
            for key in entities:
                entities[key] = list(set(entities[key]))
                
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
        
        return entities
    
    def _create_empty_result(self, text: str, error: str) -> FinBERTResult:
        """Create result for empty input"""
        return FinBERTResult(
            text=text,
            sentiment=SentimentLabel.NEUTRAL,
            confidence=0.0,
            scores={'Positive': 0.33, 'Negative': 0.33, 'Neutral': 0.34},
            entities={},
            model_info={'error': f'Empty input: {error}'}
        )
    
    def _create_fallback_result(self, text: str, error: str) -> FinBERTResult:
        """Create fallback result when model is unavailable"""
        return FinBERTResult(
            text=text,
            sentiment=SentimentLabel.NEUTRAL,
            confidence=0.33,
            scores={'Positive': 0.33, 'Negative': 0.33, 'Neutral': 0.34},
            entities={},
            model_info={'error': f'FinBERT unavailable: {error}'}
        )
    
    def analyze_batch(self, texts: List[str]) -> List[FinBERTResult]:
        """
        Analyze multiple texts efficiently
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of FinBERTResult objects
        """
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Analyzing text {i+1}/{len(texts)}")
            result = self.analyze_sentiment(text)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model setup"""
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'device': 'CUDA' if self.device >= 0 else 'CPU',
            'is_initialized': self.is_initialized,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'spacy_available': SPACY_AVAILABLE,
            'initialization_error': self.initialization_error
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get detailed model status information"""
        return {
            'is_available': self.is_available(),
            'is_initialized': self.is_initialized,
            'model_name': self.model_name,
            'device': self.device,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'spacy_available': SPACY_AVAILABLE,
            'initialization_error': self.initialization_error,
            'max_length': self.max_length
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_analyses': getattr(self, '_total_analyses', 0),
            'successful_analyses': getattr(self, '_successful_analyses', 0),
            'failed_analyses': getattr(self, '_failed_analyses', 0),
            'average_processing_time': getattr(self, '_avg_processing_time', 0.0),
            'last_analysis_time': getattr(self, '_last_analysis_time', None)
        }
    
    def _update_stats(self, processing_time: float, success: bool = True) -> None:
        """Update performance statistics"""
        if not hasattr(self, '_total_analyses'):
            self._total_analyses = 0
            self._successful_analyses = 0
            self._failed_analyses = 0
            self._avg_processing_time = 0.0
        
        self._total_analyses += 1
        self._last_analysis_time = processing_time
        
        if success:
            self._successful_analyses += 1
            # Update running average
            current_avg = self._avg_processing_time
            count = self._successful_analyses
            self._avg_processing_time = ((current_avg * (count - 1)) + processing_time) / count
        else:
            self._failed_analyses += 1
    
    def switch_model(self, new_model_name: str) -> bool:
        """
        Switch to a different FinBERT model
        
        Args:
            new_model_name: HuggingFace model identifier
            
        Returns:
            bool: True if switch successful, False otherwise
        """
        try:
            old_model = self.model_name
            self.model_name = new_model_name
            
            # Reset initialization status
            self.is_initialized = False
            self.initialization_error = None
            
            # Clear existing models
            self.tokenizer = None
            self.model = None
            self.pipeline = None
            
            # Reinitialize with new model
            self._initialize_finbert()
            self.is_initialized = True
            
            logger.info(f"✅ Successfully switched from {old_model} to {new_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to switch model: {e}")
            # Revert to old model
            self.model_name = old_model
            self._initialize_finbert()
            return False
    
    def clear_cache(self) -> None:
        """Clear any cached results and reset statistics"""
        # Reset performance statistics
        if hasattr(self, '_total_analyses'):
            self._total_analyses = 0
            self._successful_analyses = 0
            self._failed_analyses = 0
            self._avg_processing_time = 0.0
            self._last_analysis_time = None
        
        logger.info("🧹 Cache and statistics cleared")
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Comprehensive validation of the FinBERT setup
        
        Returns:
            Dict with validation results
        """
        validation_results = {
            'status': 'unknown',
            'details': {},
            'recommendations': []
        }
        
        try:
            # Check dependencies
            validation_results['details']['transformers_available'] = TRANSFORMERS_AVAILABLE
            validation_results['details']['spacy_available'] = SPACY_AVAILABLE
            validation_results['details']['torch_available'] = torch is not None
            
            if TRANSFORMERS_AVAILABLE and torch:
                validation_results['details']['cuda_available'] = torch.cuda.is_available()
                validation_results['details']['device'] = 'CUDA' if self.device >= 0 else 'CPU'
            
            # Test model initialization
            validation_results['details']['model_initialized'] = self.is_initialized
            validation_results['details']['initialization_error'] = self.initialization_error
            
            # Test analysis capability
            if self.is_available():
                test_result = self.analyze_sentiment("Test analysis for validation")
                validation_results['details']['analysis_working'] = True
                validation_results['details']['test_sentiment'] = test_result.sentiment.value
                validation_results['status'] = 'fully_functional'
            else:
                validation_results['details']['analysis_working'] = False
                validation_results['status'] = 'fallback_mode'
                validation_results['recommendations'].append(
                    "Install transformers and torch for full FinBERT functionality"
                )
            
            # Performance recommendations
            if not TRANSFORMERS_AVAILABLE:
                validation_results['recommendations'].append(
                    "pip install transformers torch for FinBERT support"
                )
            
            if self.device == -1 and torch and torch.cuda.is_available():
                validation_results['recommendations'].append(
                    "CUDA detected but not being used. Set use_cuda=True for better performance"
                )
            
            return validation_results
            
        except Exception as e:
            validation_results['status'] = 'error'
            validation_results['details']['validation_error'] = str(e)
            return validation_results

    def is_available(self) -> bool:
        """Check if FinBERT model is available and working"""
        return self.is_initialized and self.pipeline is not None
    
    def analyze_single(self, text: str) -> FinBERTResult:
        """Alias for analyze_sentiment for backward compatibility"""
        return self.analyze_sentiment(text)

# Convenience functions for backward compatibility and ease of use
def analyze_finbert(text: str, model_name: str = "ProsusAI/finbert") -> Dict[str, Any]:
    """
    Quick FinBERT analysis function
    
    Args:
        text: Text to analyze
        model_name: FinBERT model to use
        
    Returns:
        Analysis result as dictionary
    """
    analyzer = CleanFinBERTAnalyzer(model_name=model_name)
    result = analyzer.analyze_sentiment(text)
    return result.to_dict()

def analyze_finbert_batch(texts: List[str], model_name: str = "ProsusAI/finbert") -> List[Dict[str, Any]]:
    """
    Quick batch FinBERT analysis function
    
    Args:
        texts: List of texts to analyze
        model_name: FinBERT model to use
        
    Returns:
        List of analysis results as dictionaries
    """
    analyzer = CleanFinBERTAnalyzer(model_name=model_name)
    results = analyzer.analyze_batch(texts)
    return [result.to_dict() for result in results]

# Maintain backward compatibility
FinBERTAnalyzer = CleanFinBERTAnalyzer

# Test harness
def test_clean_finbert_analyzer():
    """Test the clean FinBERT analyzer"""
    print("🧪 Testing Clean FinBERT Sentiment Analyzer")
    print("=" * 60)
    
    # Test headlines
    test_headlines = [
        "Apple stock surges 15% after record quarterly earnings beat expectations",
        "Federal Reserve signals potential interest rate cuts amid economic uncertainty",
        "Tesla shares plummet following disappointing delivery numbers and production delays",
        "S&P 500 maintains steady growth as investors await inflation data",
        "Bitcoin crashes below $20,000 as crypto winter continues"
    ]
    
    analyzer = CleanFinBERTAnalyzer()
    
    # Show model info
    model_info = analyzer.get_model_info()
    print(f"📊 Model Info:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    print()
    
    if not analyzer.is_initialized:
        print("⚠️ FinBERT model not available. Install dependencies:")
        print("   pip install transformers torch")
        print("   python -m spacy download en_core_web_sm")
        print("\n🔄 Running in fallback mode...")
        print()
    
    # Test each headline
    for i, headline in enumerate(test_headlines, 1):
        print(f"📰 Test {i}: {headline}")
        print("-" * 80)
        
        result = analyzer.analyze_sentiment(headline)
        
        print(f"🎯 Sentiment: {result.sentiment.value}")
        print(f"📊 Confidence: {result.confidence:.3f}")
        print(f"📈 Scores: {result.scores}")
        
        if result.entities and any(result.entities.values()):
            print(f"🏷️ Entities: {result.entities}")
        
        if result.processing_time:
            print(f"⏱️ Processing Time: {result.processing_time:.3f}s")
        
        if result.model_info and 'error' in result.model_info:
            print(f"⚠️ Note: {result.model_info['error']}")
        
        print()
    
    # Test batch analysis
    print(f"🔄 Testing Batch Analysis...")
    batch_results = analyzer.analyze_batch(test_headlines[:2])
    print(f"✅ Processed {len(batch_results)} texts in batch")
    
    print(f"\n✅ Clean FinBERT analyzer test completed!")

if __name__ == "__main__":
    test_clean_finbert_analyzer()
