"""
Main inference pipeline for financial sentiment analysis
Orchestrates news fetching, text cleaning, and dual-model sentiment analysis
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialSentimentPipeline:
    """
    Complete pipeline for financial news sentiment analysis
    Integrates news fetching, text cleaning, and dual sentiment models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the sentiment analysis pipeline
        
        Args:
            config: Configuration dictionary for pipeline components
        """
        self.config = config or self._default_config()
        self.news_fetcher = None
        self.text_cleaner = None
        self.vader_analyzer = None
        self.finbert_analyzer = None
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'processing_time': 0,
            'last_run': None
        }
        
        self._initialize_components()
    
    def _default_config(self) -> Dict:
        """Default configuration for the pipeline"""
        return {
            'news_sources': ['newsapi', 'rss'],
            'max_articles': 50,
            'financial_keywords': ['earnings', 'stock', 'market', 'revenue', 'profit'],
            'enable_vader': True,
            'enable_finbert': True,
            'enable_entities': True,
            'save_results': True,
            'output_format': 'json',
            'model_comparison': True,
            'batch_size': 10
        }
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Import and initialize news fetcher
            from core.news_fetcher import FinancialNewsFetcher
            self.news_fetcher = FinancialNewsFetcher()
            logger.info("✅ News fetcher initialized")
            
            # Import and initialize text cleaner
            from core.text_cleaner import FinancialTextCleaner
            self.text_cleaner = FinancialTextCleaner()
            logger.info("✅ Text cleaner initialized")
            
            # Initialize VADER analyzer if enabled
            if self.config.get('enable_vader', True):
                try:
                    from core.sentiment_vader import FinancialVaderAnalyzer
                    self.vader_analyzer = FinancialVaderAnalyzer()
                    logger.info("✅ VADER analyzer initialized")
                except Exception as e:
                    logger.warning(f"VADER analyzer failed to initialize: {e}")
            
            # Initialize FinBERT analyzer if enabled
            if self.config.get('enable_finbert', True):
                try:
                    from core.sentiment_finbert_clean import CleanFinBERTAnalyzer as FinBERTAnalyzer
                    self.finbert_analyzer = FinBERTAnalyzer()
                    logger.info("✅ FinBERT analyzer initialized")
                except Exception as e:
                    logger.warning(f"FinBERT analyzer failed to initialize: {e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def fetch_and_analyze_live(self, keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Fetch live news and perform sentiment analysis
        
        Args:
            keywords: Optional custom keywords to search for
            
        Returns:
            List of analyzed articles with sentiment scores
        """
        start_time = time.time()
        logger.info("🚀 Starting live news sentiment analysis pipeline")
        
        # Use default keywords if none provided
        if keywords is None:
            keywords = self.config.get('financial_keywords', ['stock', 'market'])
        
        try:
            # Step 1: Fetch news articles
            logger.info(f"📰 Fetching news with keywords: {keywords}")
            articles = self.news_fetcher.fetch_financial_news(
                keywords=keywords,
                max_articles=self.config.get('max_articles', 50)
            )
            
            if not articles:
                logger.warning("No articles fetched")
                return []
            
            logger.info(f"📊 Fetched {len(articles)} articles")
            
            # Step 2: Process articles through the pipeline
            results = self._process_articles(articles)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(articles), len(results), processing_time)
            
            logger.info(f"✅ Pipeline completed in {processing_time:.2f}s")
            logger.info(f"📈 Successfully analyzed {len(results)}/{len(articles)} articles")
            
            return results
            
        except Exception as e:
            logger.error(f"Live analysis pipeline failed: {e}")
            return []
    
    def analyze_custom_text(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze custom text inputs through the sentiment pipeline
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of sentiment analysis results
        """
        start_time = time.time()
        logger.info(f"🎯 Analyzing {len(texts)} custom texts")
        
        try:
            # Create article-like structure for custom texts
            articles = []
            for i, text in enumerate(texts):
                articles.append({
                    'title': f"Custom Text {i+1}",
                    'description': text,
                    'content': text,
                    'url': f"custom_{i+1}",
                    'published_at': datetime.now().isoformat(),
                    'source': 'custom_input'
                })
            
            # Process through pipeline
            results = self._process_articles(articles)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(texts), len(results), processing_time)
            
            logger.info(f"✅ Custom text analysis completed in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Custom text analysis failed: {e}")
            return []
    
    def _process_articles(self, articles: List[Dict]) -> List[Dict[str, Any]]:
        """
        Process articles through the complete sentiment analysis pipeline
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of processed results with sentiment analysis
        """
        results = []
        batch_size = self.config.get('batch_size', 10)
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            logger.info(f"🔄 Processing batch {i//batch_size + 1}/{(len(articles)-1)//batch_size + 1}")
            
            for article in batch:
                try:
                    result = self._process_single_article(article)
                    if result:
                        results.append(result)
                        self.stats['successful_analyses'] += 1
                    else:
                        self.stats['failed_analyses'] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process article: {e}")
                    self.stats['failed_analyses'] += 1
        
        return results
    
    def _process_single_article(self, article: Dict) -> Optional[Dict[str, Any]]:
        """
        Process a single article through the complete pipeline
        
        Args:
            article: Article dictionary
            
        Returns:
            Processed result with sentiment analysis
        """
        try:
            # Extract text content
            text_content = article.get('title', '') + ' ' + article.get('description', '')
            if not text_content.strip():
                return None
            
            # Step 1: Clean and preprocess text
            cleaned_result = self.text_cleaner.clean_financial_text(text_content)
            cleaned_text = cleaned_result['cleaned_text']
            
            if not cleaned_text:
                return None
            
            # Initialize result structure
            result = {
                'article': {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('published_at', ''),
                    'source': article.get('source', 'unknown')
                },
                'preprocessing': cleaned_result,
                'sentiment_analysis': {},
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': '1.0'
            }
            
            # Step 2: VADER sentiment analysis
            if self.vader_analyzer:
                try:
                    vader_result = self.vader_analyzer.analyze_sentiment(
                        cleaned_text, 
                        include_explanation=True
                    )
                    result['sentiment_analysis']['vader'] = vader_result
                except Exception as e:
                    logger.warning(f"VADER analysis failed: {e}")
            
            # Step 3: FinBERT sentiment analysis
            if self.finbert_analyzer:
                try:
                    finbert_result = self.finbert_analyzer.analyze_sentiment(
                        cleaned_text,
                        include_entities=self.config.get('enable_entities', True)
                    )
                    result['sentiment_analysis']['finbert'] = finbert_result
                except Exception as e:
                    logger.warning(f"FinBERT analysis failed: {e}")
            
            # Step 4: Model comparison and consensus
            if self.config.get('model_comparison', True):
                result['model_comparison'] = self._compare_models(result['sentiment_analysis'])
            
            return result
            
        except Exception as e:
            logger.error(f"Single article processing failed: {e}")
            return None
    
    def _compare_models(self, sentiment_results: Dict) -> Dict[str, Any]:
        """
        Compare results from different sentiment models
        
        Args:
            sentiment_results: Dictionary containing results from different models
            
        Returns:
            Model comparison analysis
        """
        comparison = {
            'models_used': list(sentiment_results.keys()),
            'agreement': False,
            'consensus_sentiment': None,
            'confidence_scores': {},
            'recommendation': 'manual_review'
        }
        
        try:
            # Extract sentiments and confidences
            sentiments = {}
            confidences = {}
            
            for model, result in sentiment_results.items():
                if 'sentiment' in result and 'confidence' in result:
                    sentiments[model] = result['sentiment']
                    confidences[model] = result['confidence']
            
            if len(sentiments) >= 2:
                # Check agreement
                sentiment_values = list(sentiments.values())
                comparison['agreement'] = len(set(sentiment_values)) == 1
                
                if comparison['agreement']:
                    # Models agree
                    comparison['consensus_sentiment'] = sentiment_values[0]
                    avg_confidence = sum(confidences.values()) / len(confidences)
                    comparison['consensus_confidence'] = avg_confidence
                    comparison['recommendation'] = 'high_confidence' if avg_confidence > 0.7 else 'moderate_confidence'
                else:
                    # Models disagree
                    comparison['consensus_sentiment'] = 'CONFLICT'
                    comparison['recommendation'] = 'manual_review'
                    comparison['conflict_details'] = sentiments
            
            comparison['confidence_scores'] = confidences
            
        except Exception as e:
            logger.warning(f"Model comparison failed: {e}")
            comparison['error'] = str(e)
        
        return comparison
    
    def _update_stats(self, total: int, successful: int, processing_time: float):
        """Update pipeline statistics"""
        self.stats['total_processed'] += total
        self.stats['processing_time'] += processing_time
        self.stats['last_run'] = datetime.now().isoformat()
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        success_rate = 0
        if self.stats['total_processed'] > 0:
            success_rate = self.stats['successful_analyses'] / self.stats['total_processed']
        
        avg_processing_time = 0
        if self.stats['total_processed'] > 0:
            avg_processing_time = self.stats['processing_time'] / self.stats['total_processed']
        
        return {
            'total_processed': self.stats['total_processed'],
            'successful_analyses': self.stats['successful_analyses'],
            'failed_analyses': self.stats['failed_analyses'],
            'success_rate': success_rate,
            'average_processing_time': avg_processing_time,
            'last_run': self.stats['last_run'],
            'components_status': {
                'news_fetcher': self.news_fetcher is not None,
                'text_cleaner': self.text_cleaner is not None,
                'vader_analyzer': self.vader_analyzer is not None,
                'finbert_analyzer': self.finbert_analyzer is not None
            }
        }
    
    def export_results(self, results: List[Dict], format: str = 'json', filename: Optional[str] = None) -> str:
        """
        Export analysis results to file
        
        Args:
            results: Analysis results to export
            format: Export format ('json', 'csv')
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_{timestamp}.{format}"
        
        output_path = Path("data") / "results" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                import pandas as pd
                # Flatten results for CSV export
                flattened_results = []
                for result in results:
                    flat_result = {
                        'title': result['article']['title'],
                        'url': result['article']['url'],
                        'published_at': result['article']['published_at'],
                        'source': result['article']['source']
                    }
                    
                    # Add sentiment results
                    for model, sentiment_data in result['sentiment_analysis'].items():
                        flat_result[f'{model}_sentiment'] = sentiment_data.get('sentiment', '')
                        flat_result[f'{model}_confidence'] = sentiment_data.get('confidence', 0)
                    
                    # Add consensus
                    if 'model_comparison' in result:
                        flat_result['consensus'] = result['model_comparison'].get('consensus_sentiment', '')
                        flat_result['agreement'] = result['model_comparison'].get('agreement', False)
                    
                    flattened_results.append(flat_result)
                
                df = pd.DataFrame(flattened_results)
                df.to_csv(output_path, index=False)
            
            logger.info(f"Results exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return ""

# Convenience functions
def run_live_analysis(keywords: Optional[List[str]] = None, config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Quick function to run live news sentiment analysis
    
    Args:
        keywords: Keywords to search for
        config: Pipeline configuration
        
    Returns:
        Analysis results
    """
    pipeline = FinancialSentimentPipeline(config)
    return pipeline.fetch_and_analyze_live(keywords)

def analyze_texts(texts: List[str], config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Quick function to analyze custom texts
    
    Args:
        texts: List of texts to analyze
        config: Pipeline configuration
        
    Returns:
        Analysis results
    """
    pipeline = FinancialSentimentPipeline(config)
    return pipeline.analyze_custom_text(texts)

# Test harness
def test_inference_pipeline():
    """Test the complete inference pipeline"""
    print("🧪 Testing Financial Sentiment Inference Pipeline")
    print("=" * 60)
    
    # Test with custom texts (safer than live API calls)
    test_texts = [
        "Apple stock surges 15% after record quarterly earnings beat expectations",
        "Federal Reserve signals potential interest rate cuts amid economic uncertainty",
        "Tesla shares plummet following disappointing delivery numbers and production delays"
    ]
    
    try:
        # Initialize pipeline
        pipeline = FinancialSentimentPipeline()
        
        # Test 1: Custom text analysis
        print("📝 Test 1: Custom Text Analysis")
        print("-" * 40)
        
        results = pipeline.analyze_custom_text(test_texts)
        
        for i, result in enumerate(results, 1):
            print(f"\n📰 Result {i}: {result['article']['title']}")
            
            # Display sentiment results
            for model, sentiment_data in result['sentiment_analysis'].items():
                sentiment = sentiment_data.get('sentiment', 'Unknown')
                confidence = sentiment_data.get('confidence', 0)
                print(f"  🤖 {model.upper()}: {sentiment} ({confidence:.3f})")
            
            # Display consensus
            if 'model_comparison' in result:
                consensus = result['model_comparison'].get('consensus_sentiment', 'Unknown')
                agreement = result['model_comparison'].get('agreement', False)
                print(f"  🤝 Consensus: {consensus} (Agreement: {agreement})")
        
        # Test 2: Pipeline statistics
        print(f"\n📊 Test 2: Pipeline Statistics")
        print("-" * 40)
        
        stats = pipeline.get_pipeline_stats()
        print(f"Total Processed: {stats['total_processed']}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        print(f"Avg Processing Time: {stats['average_processing_time']:.3f}s")
        print(f"Components Status: {stats['components_status']}")
        
        # Test 3: Export results
        print(f"\n💾 Test 3: Export Results")
        print("-" * 40)
        
        json_file = pipeline.export_results(results, format='json')
        if json_file:
            print(f"✅ Results exported to: {json_file}")
        
        print(f"\n✅ Pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference_pipeline()
