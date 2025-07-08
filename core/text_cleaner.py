"""
Advanced Financial Text Preprocessing and Cleaning
Specialized for financial news with NER, ticker extraction, and domain-specific cleaning
"""

import re
import spacy
import string
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessedText:
    """Container for processed text with extracted entities"""
    original: str
    cleaned: str
    tokens: List[str]
    entities: Dict[str, List[str]]
    tickers: List[str]
    financial_terms: List[str]
    sentiment_keywords: List[str]

class FinancialTextCleaner:
    """
    Elite text preprocessing for financial sentiment analysis
    """
    
    def __init__(self, load_spacy: bool = True):
        self.load_spacy = load_spacy
        self.nlp = None
        
        if load_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model loaded successfully")
            except OSError:
                logger.warning("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.load_spacy = False
        
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize regex patterns and financial vocabularies"""
        
        # Ticker patterns
        self.ticker_patterns = [
            r'\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b',  # Standard tickers (AAPL, BRK.A)
            r'\$[A-Z]{1,5}\b',                    # Dollar-prefixed ($TSLA)
            r'\b[A-Z]{1,4}-USD\b',               # Crypto pairs (BTC-USD)
        ]
        
        # Financial entity patterns
        self.financial_patterns = {
            'currency': r'\$[\d,]+(?:\.\d{2})?[BMK]?|\$\d+(?:\.\d+)?(?:\s+(?:million|billion|trillion))?',
            'percentage': r'\d+(?:\.\d+)?%',
            'price_change': r'(?:up|down|gained|lost|fell|rose)\s+(?:\$?[\d,]+(?:\.\d{2})?|\d+(?:\.\d+)?%)',
            'market_cap': r'\$[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion)',
            'volume': r'(?:\d+(?:,\d{3})*|\d+(?:\.\d+)?[BMK])\s*(?:shares?|volume)',
        }
        
        # Financial terminology
        self.financial_terms = {
            'market_movements': [
                'surge', 'rally', 'soar', 'spike', 'jump', 'climb', 'rise', 'gain',
                'plunge', 'crash', 'plummet', 'tumble', 'drop', 'fall', 'decline', 'slide',
                'volatile', 'volatility', 'fluctuate', 'swing'
            ],
            'earnings_terms': [
                'earnings', 'revenue', 'profit', 'loss', 'eps', 'guidance', 'outlook',
                'beat', 'miss', 'estimate', 'forecast', 'quarterly', 'q1', 'q2', 'q3', 'q4'
            ],
            'market_terms': [
                'bull', 'bear', 'bullish', 'bearish', 'correction', 'recession',
                'recovery', 'expansion', 'bubble', 'catalyst', 'momentum'
            ],
            'financial_instruments': [
                'stock', 'share', 'equity', 'bond', 'option', 'future', 'etf',
                'index', 'commodity', 'currency', 'forex', 'crypto', 'bitcoin'
            ]
        }
        
        # Sentiment-bearing financial keywords
        self.positive_financial = [
            'profit', 'gain', 'increase', 'growth', 'strong', 'beat', 'exceed',
            'outperform', 'bullish', 'rally', 'surge', 'soar', 'breakthrough',
            'record', 'high', 'upgrade', 'buy', 'positive', 'optimistic'
        ]
        
        self.negative_financial = [
            'loss', 'decline', 'decrease', 'weak', 'miss', 'underperform',
            'bearish', 'crash', 'plunge', 'fall', 'low', 'downgrade',
            'sell', 'negative', 'pessimistic', 'concern', 'risk', 'warning'
        ]
        
        # Stopwords to preserve (unlike general NLP, these matter in finance)
        self.preserve_stopwords = {
            'up', 'down', 'in', 'out', 'over', 'under', 'above', 'below',
            'more', 'less', 'better', 'worse', 'higher', 'lower', 'top', 'bottom'
        }
        
        # Common financial abbreviations
        self.financial_abbreviations = {
            'fed': 'federal reserve',
            'sec': 'securities and exchange commission',
            'ipo': 'initial public offering',
            'ceo': 'chief executive officer',
            'cfo': 'chief financial officer',
            'gdp': 'gross domestic product',
            'cpi': 'consumer price index',
            'ppi': 'producer price index',
            'fomc': 'federal open market committee',
            'qe': 'quantitative easing',
            'etf': 'exchange traded fund',
            'reit': 'real estate investment trust',
            'ebitda': 'earnings before interest taxes depreciation amortization',
            'pe': 'price to earnings',
            'pb': 'price to book',
            'roe': 'return on equity',
            'roa': 'return on assets'
        }
    
    def extract_tickers(self, text: str) -> List[str]:
        """Advanced ticker extraction with validation"""
        tickers = set()
        
        for pattern in self.ticker_patterns:
            matches = re.findall(pattern, text)
            tickers.update(matches)
        
        # Clean and validate tickers
        cleaned_tickers = []
        for ticker in tickers:
            # Remove $ prefix if present
            clean_ticker = ticker.lstrip('$')
            
            # Skip common false positives
            false_positives = {
                'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
                'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET',
                'USE', 'MAN', 'NEW', 'NOW', 'WAY', 'MAY', 'SAY', 'CEO',
                'CFO', 'CTO', 'SEC', 'FDA', 'API', 'URL', 'USA', 'USD'
            }
            
            if (len(clean_ticker) >= 2 and 
                clean_ticker.upper() not in false_positives and
                not clean_ticker.isdigit()):
                cleaned_tickers.append(clean_ticker.upper())
        
        return sorted(list(set(cleaned_tickers)))
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities using regex patterns"""
        entities = {}
        
        for entity_type, pattern in self.financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches)) if matches else []
        
        return entities
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy if available"""
        if not self.nlp:
            return {}
        
        try:
            doc = self.nlp(text)
            entities = {
                'PERSON': [],
                'ORG': [],
                'GPE': [],      # Geopolitical entities
                'MONEY': [],
                'PERCENT': [],
                'DATE': []
            }
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text.strip())
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return entities
            
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            return {}
    
    def identify_financial_terms(self, text: str) -> List[str]:
        """Identify financial terminology in text"""
        text_lower = text.lower()
        found_terms = []
        
        for category, terms in self.financial_terms.items():
            for term in terms:
                if term in text_lower:
                    found_terms.append(term)
        
        return list(set(found_terms))
    
    def extract_sentiment_keywords(self, text: str) -> List[str]:
        """Extract sentiment-bearing financial keywords"""
        text_lower = text.lower()
        sentiment_words = []
        
        # Check positive keywords
        for word in self.positive_financial:
            if word in text_lower:
                sentiment_words.append(f"positive:{word}")
        
        # Check negative keywords
        for word in self.negative_financial:
            if word in text_lower:
                sentiment_words.append(f"negative:{word}")
        
        return sentiment_words
    
    def clean_financial_text(self, text: str) -> str:
        """
        Advanced financial text cleaning
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Expand financial abbreviations
        for abbr, expansion in self.financial_abbreviations.items():
            text = re.sub(rf'\b{abbr}\b', expansion, text, flags=re.IGNORECASE)
        
        # Normalize financial expressions
        # Convert percentage representations
        text = re.sub(r'(\d+(?:\.\d+)?)\s*percent', r'\1%', text)
        
        # Normalize currency
        text = re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', r'$\1', text)
        
        # Remove excessive punctuation but preserve important financial punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_financial(self, text: str) -> List[str]:
        """Financial-aware tokenization"""
        if not text:
            return []
        
        # Use spaCy tokenization if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                tokens = []
                for token in doc:
                    if not token.is_space:
                        # Preserve important financial tokens
                        if (token.text.lower() in self.preserve_stopwords or
                            not token.is_stop or
                            token.like_num or
                            token.text.isupper()):
                            tokens.append(token.text)
                return tokens
            except Exception as e:
                logger.warning(f"spaCy tokenization failed: {e}")
        
        # Fallback to simple tokenization
        # Split on whitespace and punctuation, but preserve financial expressions
        tokens = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\w+|\$', text)
        return [token for token in tokens if len(token) > 1 or token in '$%']
    
    def process_comprehensive(self, text: str) -> ProcessedText:
        """
        Comprehensive text processing for financial sentiment analysis
        """
        if not text:
            return ProcessedText("", "", [], {}, [], [], [])
        
        # Clean the text
        cleaned_text = self.clean_financial_text(text)
        
        # Tokenize
        tokens = self.tokenize_financial(cleaned_text)
        
        # Extract various types of information
        tickers = self.extract_tickers(text)
        financial_entities = self.extract_financial_entities(text)
        named_entities = self.extract_named_entities(text)
        financial_terms = self.identify_financial_terms(text)
        sentiment_keywords = self.extract_sentiment_keywords(text)
        
        # Combine entities
        all_entities = {**financial_entities, **named_entities}
        
        return ProcessedText(
            original=text,
            cleaned=cleaned_text,
            tokens=tokens,
            entities=all_entities,
            tickers=tickers,
            financial_terms=financial_terms,
            sentiment_keywords=sentiment_keywords
        )

# Convenience functions
def clean_financial_text(text: str) -> Dict[str, Any]:
    """
    Convenience function to clean financial text
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text result
    """
    cleaner = FinancialTextCleaner()
    return cleaner.clean_financial_text(text)

def test_text_cleaner():
    """Test the financial text cleaner"""
    
    cleaner = FinancialTextCleaner()
    
    # Test headlines
    test_texts = [
        "Apple (AAPL) stock surges 5% after beating Q3 earnings estimates by $0.15 per share",
        "Fed signals possible pause in rate hikes, sending markets up 2.3%",
        "Tesla plunges 8% on disappointing delivery numbers and CEO departure rumors",
        "$TSLA down 12% pre-market after missing revenue guidance by $500M",
        "Bitcoin (BTC-USD) rally continues, up 15% to $45,000 amid institutional buying"
    ]
    
    print("🧪 Testing Elite Financial Text Cleaner")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📰 Test {i}: {text}")
        print("-" * 50)
        
        processed = cleaner.process_comprehensive(text)
        
        print(f"🧹 Cleaned: {processed.cleaned}")
        if processed.tickers:
            print(f"💰 Tickers: {processed.tickers}")
        if processed.financial_terms:
            print(f"🏦 Financial Terms: {processed.financial_terms[:5]}")
        if processed.sentiment_keywords:
            print(f"😊😢 Sentiment Keywords: {processed.sentiment_keywords}")
        if processed.entities.get('currency'):
            print(f"💵 Currency: {processed.entities['currency']}")
        if processed.entities.get('percentage'):
            print(f"📊 Percentages: {processed.entities['percentage']}")

if __name__ == "__main__":
    test_text_cleaner()
