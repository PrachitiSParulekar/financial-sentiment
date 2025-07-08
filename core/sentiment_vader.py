"""
Advanced VADER Sentiment Analysis for Financial Text
Enhanced with financial domain knowledge and custom rules
"""

import re
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class VaderResult:
    """Enhanced VADER result with financial context"""
    sentiment_label: str
    confidence: float
    raw_scores: Dict[str, float]
    financial_modifiers: List[str]
    explanation: str

class FinancialVaderAnalyzer:
    """
    Enhanced VADER analyzer with financial domain knowledge
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self._initialize_financial_lexicon()
        self._initialize_modifiers()
    
    def _initialize_financial_lexicon(self):
        """Add financial-specific sentiment words to VADER lexicon"""
        
        # Financial positive terms with intensity scores
        financial_positive = {
            'surge': 2.5, 'rally': 2.0, 'soar': 2.8, 'spike': 2.3, 'boom': 2.5,
            'bull': 2.0, 'bullish': 2.2, 'outperform': 2.0, 'beat': 1.8,
            'exceed': 1.7, 'strong': 1.5, 'robust': 1.6, 'solid': 1.3,
            'breakthrough': 2.2, 'record': 1.8, 'high': 1.2, 'peak': 1.5,
            'upgrade': 1.9, 'overweight': 1.4, 'buy': 1.6, 'accumulate': 1.5,
            'momentum': 1.4, 'optimistic': 1.8, 'confident': 1.6
        }
        
        # Financial negative terms with intensity scores
        financial_negative = {
            'plunge': -2.8, 'crash': -3.0, 'plummet': -2.7, 'tumble': -2.3,
            'bear': -2.0, 'bearish': -2.2, 'underperform': -2.0, 'miss': -1.8,
            'disappoint': -2.1, 'weak': -1.5, 'fragile': -1.8, 'volatile': -1.3,
            'concern': -1.6, 'worry': -1.7, 'fear': -2.2, 'panic': -2.8,
            'downgrade': -2.1, 'underweight': -1.6, 'sell': -1.8, 'dump': -2.3,
            'pessimistic': -2.0, 'uncertain': -1.4, 'risk': -1.5, 'warning': -1.8
        }
        
        # Financial neutral terms (important context words)
        financial_neutral = {
            'maintain': 0.0, 'hold': 0.0, 'neutral': 0.0, 'unchanged': 0.0,
            'steady': 0.2, 'stable': 0.3, 'flat': 0.0, 'mixed': 0.0,
            'guidance': 0.0, 'forecast': 0.0, 'estimate': 0.0, 'target': 0.0
        }
        
        # Update VADER lexicon
        self.analyzer.lexicon.update(financial_positive)
        self.analyzer.lexicon.update(financial_negative)
        self.analyzer.lexicon.update(financial_neutral)
        
        # Store for reference
        self.financial_lexicon = {
            **financial_positive,
            **financial_negative,
            **financial_neutral
        }
    
    def _initialize_modifiers(self):
        """Initialize financial sentiment modifiers"""
        
        # Magnitude modifiers for financial terms
        self.magnitude_modifiers = {
            'massive': 1.8, 'huge': 1.6, 'major': 1.4, 'significant': 1.3,
            'substantial': 1.2, 'considerable': 1.2, 'notable': 1.1,
            'slight': 0.6, 'minor': 0.7, 'modest': 0.8, 'marginal': 0.5
        }
        
        # Direction modifiers
        self.direction_modifiers = {
            'sharply': 1.5, 'dramatically': 1.6, 'rapidly': 1.3,
            'gradually': 0.8, 'slowly': 0.7, 'slightly': 0.6
        }
        
        # Temporal modifiers (financial context)
        self.temporal_modifiers = {
            'suddenly': 1.2, 'unexpectedly': 1.3, 'surprisingly': 1.2,
            'as expected': 0.9, 'predictably': 0.8, 'inevitably': 1.0
        }
        
        # Financial context intensifiers
        self.financial_intensifiers = {
            'record-breaking': 1.7, 'unprecedented': 1.6, 'historic': 1.4,
            'best-ever': 1.6, 'worst-ever': -1.6, 'all-time': 1.3
        }
    
    def _calculate_financial_context_score(self, text: str) -> float:
        """Calculate additional sentiment score based on financial context"""
        text_lower = text.lower()
        context_score = 0.0
        
        # Check for percentage changes (strong indicators)
        percentage_patterns = [
            (r'up\s+(\d+(?:\.\d+)?)%', 1.0),      # "up 5%" -> positive
            (r'down\s+(\d+(?:\.\d+)?)%', -1.0),   # "down 3%" -> negative
            (r'gained?\s+(\d+(?:\.\d+)?)%', 1.0), # "gained 2%" -> positive
            (r'lost\s+(\d+(?:\.\d+)?)%', -1.0),   # "lost 4%" -> negative
            (r'fell\s+(\d+(?:\.\d+)?)%', -1.0),   # "fell 6%" -> negative
            (r'rose\s+(\d+(?:\.\d+)?)%', 1.0),    # "rose 3%" -> positive
        ]
        
        for pattern, sentiment_direction in percentage_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                percentage = float(match)
                # Scale sentiment by percentage magnitude
                magnitude = min(percentage / 10.0, 2.0)  # Cap at 2.0
                context_score += sentiment_direction * magnitude
        
        # Check for earnings beats/misses
        if 'beat' in text_lower and ('estimate' in text_lower or 'expectation' in text_lower):
            context_score += 1.5
        elif 'miss' in text_lower and ('estimate' in text_lower or 'expectation' in text_lower):
            context_score -= 1.5
        
        # Check for price movements with currency
        price_up_patterns = [
            r'\$[\d,]+.*(?:gain|rise|up|increase)',
            r'(?:gain|rise|up|increase).*\$[\d,]+',
            r'price.*(?:jump|surge|rally)'
        ]
        
        price_down_patterns = [
            r'\$[\d,]+.*(?:loss|fall|down|decrease|drop)',
            r'(?:loss|fall|down|decrease|drop).*\$[\d,]+',
            r'price.*(?:plunge|crash|tumble)'
        ]
        
        for pattern in price_up_patterns:
            if re.search(pattern, text_lower):
                context_score += 0.8
        
        for pattern in price_down_patterns:
            if re.search(pattern, text_lower):
                context_score -= 0.8
        
        return np.clip(context_score, -3.0, 3.0)  # Clip to reasonable range
    
    def _apply_financial_modifiers(self, base_scores: Dict[str, float], text: str) -> Dict[str, float]:
        """Apply financial domain modifiers to base VADER scores"""
        text_lower = text.lower()
        modified_scores = base_scores.copy()
        
        # Apply magnitude modifiers
        magnitude_multiplier = 1.0
        for modifier, multiplier in self.magnitude_modifiers.items():
            if modifier in text_lower:
                magnitude_multiplier *= multiplier
        
        # Apply direction modifiers
        direction_multiplier = 1.0
        for modifier, multiplier in self.direction_modifiers.items():
            if modifier in text_lower:
                direction_multiplier *= multiplier
        
        # Apply combined multipliers
        total_multiplier = magnitude_multiplier * direction_multiplier
        
        # Modify scores
        for key in ['pos', 'neg', 'neu']:
            if key != 'neu':  # Don't modify neutral scores as much
                modified_scores[key] *= total_multiplier
        
        # Recalculate compound score
        modified_scores['compound'] = self._calculate_compound_score(
            modified_scores['pos'], modified_scores['neg'], modified_scores['neu']
        )
        
        return modified_scores
    
    def _calculate_compound_score(self, pos: float, neg: float, neu: float) -> float:
        """Recalculate compound score using VADER's normalization"""
        # This approximates VADER's compound score calculation
        compound = pos - neg
        # Normalize
        if compound >= 0:
            compound = compound / (compound + 1)
        else:
            compound = compound / (abs(compound) + 1)
        return compound
    
    def _get_sentiment_label_and_confidence(self, scores: Dict[str, float]) -> Tuple[str, float]:
        """Determine sentiment label and confidence from scores"""
        compound = scores['compound']
        pos = scores['pos']
        neg = scores['neg']
        neu = scores['neu']
        
        # Enhanced thresholds for financial context
        if compound >= 0.25:
            sentiment = 'positive'
            confidence = min(compound + pos * 0.3, 1.0)
        elif compound <= -0.25:
            sentiment = 'negative'
            confidence = min(abs(compound) + neg * 0.3, 1.0)
        else:
            sentiment = 'neutral'
            confidence = neu + abs(compound) * 0.5
        
        return sentiment, confidence
    
    def _generate_explanation(self, text: str, scores: Dict[str, float], 
                           financial_modifiers: List[str]) -> str:
        """Generate human-readable explanation of sentiment analysis"""
        sentiment, confidence = self._get_sentiment_label_and_confidence(scores)
        
        explanation_parts = [f"Classified as {sentiment.upper()} (confidence: {confidence:.3f})"]
        
        # Add information about financial terms found
        found_financial_terms = []
        text_lower = text.lower()
        for term, score in self.financial_lexicon.items():
            if term in text_lower:
                found_financial_terms.append(f"{term}({score:+.1f})")
        
        if found_financial_terms:
            explanation_parts.append(f"Financial terms: {', '.join(found_financial_terms[:3])}")
        
        # Add modifier information
        if financial_modifiers:
            explanation_parts.append(f"Modifiers: {', '.join(financial_modifiers)}")
        
        # Add score breakdown
        explanation_parts.append(
            f"Scores - Positive: {scores['pos']:.3f}, "
            f"Negative: {scores['neg']:.3f}, "
            f"Neutral: {scores['neu']:.3f}, "
            f"Compound: {scores['compound']:.3f}"
        )
        
        return " | ".join(explanation_parts)
    
    def analyze(self, text: str, include_explanation: bool = True) -> VaderResult:
        """
        Perform enhanced VADER sentiment analysis on financial text
        """
        if not text:
            return VaderResult('neutral', 0.0, {}, [], "Empty text")
        
        # Get base VADER scores
        base_scores = self.analyzer.polarity_scores(text)
        
        # Calculate financial context score
        financial_context = self._calculate_financial_context_score(text)
        
        # Apply financial modifiers
        modified_scores = self._apply_financial_modifiers(base_scores, text)
        
        # Incorporate financial context into compound score
        final_compound = modified_scores['compound'] + (financial_context * 0.2)
        final_compound = np.clip(final_compound, -1.0, 1.0)
        
        final_scores = modified_scores.copy()
        final_scores['compound'] = final_compound
        
        # Determine final sentiment and confidence
        sentiment, confidence = self._get_sentiment_label_and_confidence(final_scores)
        
        # Identify applied modifiers
        financial_modifiers = []
        text_lower = text.lower()
        
        for modifier_dict in [self.magnitude_modifiers, self.direction_modifiers, 
                             self.temporal_modifiers, self.financial_intensifiers]:
            for modifier in modifier_dict:
                if modifier in text_lower:
                    financial_modifiers.append(modifier)
        
        # Generate explanation
        explanation = ""
        if include_explanation:
            explanation = self._generate_explanation(text, final_scores, financial_modifiers)
        
        return VaderResult(
            sentiment_label=sentiment,
            confidence=confidence,
            raw_scores=final_scores,
            financial_modifiers=financial_modifiers,
            explanation=explanation
        )
    
    def batch_analyze(self, texts: List[str], include_explanation: bool = False) -> List[VaderResult]:
        """Analyze multiple texts efficiently"""
        results = []
        for text in texts:
            result = self.analyze(text, include_explanation=include_explanation)
            results.append(result)
        return results
    
    def get_lexicon_stats(self) -> Dict[str, int]:
        """Get statistics about the enhanced lexicon"""
        total_terms = len(self.analyzer.lexicon)
        financial_terms = len(self.financial_lexicon)
        
        return {
            'total_lexicon_terms': total_terms,
            'financial_terms_added': financial_terms,
            'coverage_percentage': (financial_terms / total_terms) * 100
        }

# Convenience functions
def analyze_vader_sentiment(text: str) -> Dict[str, Any]:
    """
    Convenience function for VADER sentiment analysis
    
    Args:
        text: Input text to analyze
        
    Returns:
        Sentiment analysis result
    """
    analyzer = FinancialVaderAnalyzer()
    return analyzer.analyze_sentiment(text)

def test_financial_vader():
    """Test the enhanced financial VADER analyzer"""
    
    analyzer = FinancialVaderAnalyzer()
    
    # Test cases covering different financial scenarios
    test_cases = [
        "Apple stock surges 15% after beating earnings estimates by $0.20 per share",
        "Tesla shares plummet 8% on disappointing delivery numbers and production concerns",
        "Market remains flat as investors await Federal Reserve interest rate decision",
        "Bitcoin crashes dramatically, down 20% in massive sell-off amid regulatory fears",
        "Strong earnings beat sends NVIDIA rallying to record highs, up 12% in pre-market",
        "Oil prices fell slightly by 2% on modest demand concerns",
        "Fed maintains interest rates as expected, market shows mixed reaction",
        "Unprecedented surge in tech stocks drives NASDAQ to historic peaks",
        "Banking sector underperforms significantly amid rising credit concerns"
    ]
    
    print("🧪 Testing Enhanced Financial VADER Analyzer")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n📰 Test {i}: {text}")
        print("-" * 60)
        
        result = analyzer.analyze(text, include_explanation=True)
        
        # Display results
        emoji = "📈" if result.sentiment_label == 'positive' else "📉" if result.sentiment_label == 'negative' else "➡️"
        print(f"{emoji} Sentiment: {result.sentiment_label.upper()} (Confidence: {result.confidence:.3f})")
        
        if result.financial_modifiers:
            print(f"🔧 Financial Modifiers: {', '.join(result.financial_modifiers)}")
        
        print(f"📊 Raw Scores: {result.raw_scores}")
        
        if result.explanation:
            print(f"💡 Explanation: {result.explanation}")
    
    # Display lexicon statistics
    stats = analyzer.get_lexicon_stats()
    print(f"\n📊 Enhanced Lexicon Statistics:")
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")

# Test harness
def test_enhanced_vader_analyzer():
    """Test the enhanced VADER analyzer with various financial texts"""
    
    analyzer = FinancialVaderAnalyzer()
    
    # Sample texts for testing
    texts = [
        "The company's stock price jumped 10% after the positive earnings report.",
        "There are concerns about the declining sales figures and market share.",
        "Experts predict a volatile market ahead with potential for significant gains or losses.",
        "The merger between the two firms is expected to create substantial synergies.",
        "Regulatory changes could have a major impact on the industry's profitability."
    ]
    
    results = analyzer.batch_analyze(texts, include_explanation=True)
    
    for i, result in enumerate(results, 1):
        print(f"\nText {i}: {texts[i-1]}")
        print(f"Sentiment: {result.sentiment_label} (Confidence: {result.confidence:.2f})")
        print(f"Modifiers: {', '.join(result.financial_modifiers)}")
        print(f"Scores: {result.raw_scores}")
        print(f"Explanation: {result.explanation}")

if __name__ == "__main__":
    test_financial_vader()
    test_enhanced_vader_analyzer()
