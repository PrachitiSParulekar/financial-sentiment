"""
Professional CLI interface for Financial Sentiment Analysis AI
Provides command-line access to live analysis, custom text analysis, and model comparison
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialSentimentCLI:
    """
    Professional command-line interface for financial sentiment analysis
    """
    
    def __init__(self):
        """Initialize the CLI application"""
        self.version = "1.0.0"
        self.pipeline = None
        self.data_logger = None
        
        # Initialize components lazily to avoid startup delays
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline and logger components"""
        try:
            from pipeline.inference import FinancialSentimentPipeline
            from pipeline.logger import FinancialDataLogger
            
            self.pipeline = FinancialSentimentPipeline()
            self.data_logger = FinancialDataLogger(storage_type='sqlite')
            
            logger.info("✅ Financial Sentiment AI components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            print(f"❌ Error: Failed to initialize system components: {e}")
            sys.exit(1)
    
    def run_live_analysis(self, keywords: List[str], max_articles: int = 50, 
                         save_results: bool = True, output_format: str = 'json') -> Dict[str, Any]:
        """
        Run live news sentiment analysis
        
        Args:
            keywords: Keywords to search for
            max_articles: Maximum number of articles to analyze
            save_results: Whether to save results to database
            output_format: Output format ('json', 'summary', 'detailed')
            
        Returns:
            Analysis results and metadata
        """
        print(f"🚀 Starting live financial sentiment analysis...")
        print(f"🔍 Keywords: {', '.join(keywords)}")
        print(f"📊 Max articles: {max_articles}")
        print("-" * 60)
        
        start_time = datetime.now()
        
        try:
            # Configure pipeline for live analysis
            config = {
                'max_articles': max_articles,
                'financial_keywords': keywords,
                'enable_vader': True,
                'enable_finbert': True,
                'model_comparison': True,
                'save_results': save_results
            }
            
            # Update pipeline config
            self.pipeline.config.update(config)
            
            # Run analysis
            results = self.pipeline.fetch_and_analyze_live(keywords)
            
            if not results:
                print("❌ No articles found or analysis failed")
                return {'success': False, 'results': [], 'message': 'No results'}
            
            # Save results if requested
            analysis_ids = []
            if save_results and self.data_logger:
                analysis_ids = self.data_logger.log_batch_results(results)
                print(f"💾 Saved {len(analysis_ids)} results to database")
            
            # Get pipeline statistics
            stats = self.pipeline.get_pipeline_stats()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Format output
            output_data = {
                'success': True,
                'timestamp': start_time.isoformat(),
                'processing_time': processing_time,
                'total_articles': len(results),
                'keywords': keywords,
                'results': results,
                'analysis_ids': analysis_ids,
                'pipeline_stats': stats
            }
            
            # Display summary
            self._display_analysis_summary(results, processing_time)
            
            return output_data
            
        except Exception as e:
            logger.error(f"Live analysis failed: {e}")
            print(f"❌ Analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_custom_analysis(self, texts: List[str], save_results: bool = True, 
                           output_format: str = 'json') -> Dict[str, Any]:
        """
        Analyze custom text inputs
        
        Args:
            texts: List of texts to analyze
            save_results: Whether to save results
            output_format: Output format
            
        Returns:
            Analysis results
        """
        print(f"🎯 Analyzing {len(texts)} custom text(s)...")
        print("-" * 60)
        
        start_time = datetime.now()
        
        try:
            results = self.pipeline.analyze_custom_text(texts)
            
            if not results:
                print("❌ Custom text analysis failed")
                return {'success': False, 'results': []}
            
            # Save results if requested
            analysis_ids = []
            if save_results and self.data_logger:
                analysis_ids = self.data_logger.log_batch_results(results)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            self._display_analysis_summary(results, processing_time)
            
            return {
                'success': True,
                'timestamp': start_time.isoformat(),
                'processing_time': processing_time,
                'total_texts': len(results),
                'results': results,
                'analysis_ids': analysis_ids
            }
            
        except Exception as e:
            logger.error(f"Custom analysis failed: {e}")
            print(f"❌ Analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def compare_models(self, text: str) -> Dict[str, Any]:
        """
        Compare VADER and FinBERT models on the same text
        
        Args:
            text: Text to analyze with both models
            
        Returns:
            Comparison results
        """
        print(f"🔄 Comparing models on text: {text[:100]}...")
        print("-" * 60)
        
        try:
            # Analyze with both models
            results = self.pipeline.analyze_custom_text([text])
            
            if not results:
                print("❌ Model comparison failed")
                return {'success': False}
            
            result = results[0]
            sentiment_analysis = result.get('sentiment_analysis', {})
            model_comparison = result.get('model_comparison', {})
            
            # Display comparison
            print(f"📰 Text: {text}")
            print()
            
            if 'vader' in sentiment_analysis:
                vader = sentiment_analysis['vader']
                print(f"⚡ VADER Analysis:")
                print(f"   Sentiment: {vader.get('sentiment', 'Unknown')}")
                print(f"   Confidence: {vader.get('confidence', 0):.3f}")
                print(f"   Scores: {vader.get('scores', {})}")
                print()
            
            if 'finbert' in sentiment_analysis:
                finbert = sentiment_analysis['finbert']
                print(f"🤖 FinBERT Analysis:")
                print(f"   Sentiment: {finbert.get('sentiment', 'Unknown')}")
                print(f"   Confidence: {finbert.get('confidence', 0):.3f}")
                print(f"   Scores: {finbert.get('scores', {})}")
                if 'entities' in finbert:
                    print(f"   Entities: {finbert['entities']}")
                print()
            
            # Display consensus
            print(f"🤝 Model Comparison:")
            print(f"   Agreement: {model_comparison.get('agreement', False)}")
            print(f"   Consensus: {model_comparison.get('consensus_sentiment', 'Unknown')}")
            
            if 'confidence_scores' in model_comparison:
                print(f"   Confidence Scores: {model_comparison['confidence_scores']}")
            
            return {
                'success': True,
                'text': text,
                'vader': sentiment_analysis.get('vader'),
                'finbert': sentiment_analysis.get('finbert'),
                'comparison': model_comparison
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            print(f"❌ Comparison failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def query_historical_data(self, days: int = 7, sentiment: Optional[str] = None, 
                             limit: int = 50) -> Dict[str, Any]:
        """
        Query historical analysis results
        
        Args:
            days: Number of days to look back
            sentiment: Filter by sentiment (optional)
            limit: Maximum results to return
            
        Returns:
            Historical data and statistics
        """
        print(f"📊 Querying historical data (last {days} days)...")
        if sentiment:
            print(f"🎯 Filtering by sentiment: {sentiment}")
        print("-" * 60)
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Query results
            results = self.data_logger.query_results(
                start_date=start_date,
                end_date=end_date,
                sentiment=sentiment,
                limit=limit
            )
            
            # Get statistics
            stats = self.data_logger.get_model_statistics(days=days)
            
            print(f"📈 Found {len(results)} results")
            print(f"📊 Total predictions in period: {stats.get('total_predictions', 0)}")
            
            if 'models' in stats:
                for model, model_stats in stats['models'].items():
                    print(f"   {model.upper()}: {model_stats.get('total_predictions', 0)} predictions")
                    if 'sentiment_distribution' in model_stats:
                        for sent, count in model_stats['sentiment_distribution'].items():
                            print(f"     {sent}: {count}")
            
            return {
                'success': True,
                'results': results,
                'statistics': stats,
                'query_params': {
                    'days': days,
                    'sentiment_filter': sentiment,
                    'limit': limit
                }
            }
            
        except Exception as e:
            logger.error(f"Historical query failed: {e}")
            print(f"❌ Query failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _display_analysis_summary(self, results: List[Dict], processing_time: float):
        """Display a summary of analysis results"""
        if not results:
            return
        
        print(f"✅ Analysis completed in {processing_time:.2f} seconds")
        print(f"📊 Processed {len(results)} articles")
        print()
        
        # Sentiment distribution
        vader_sentiments = []
        finbert_sentiments = []
        consensus_sentiments = []
        
        for result in results:
            sentiment_analysis = result.get('sentiment_analysis', {})
            
            if 'vader' in sentiment_analysis:
                vader_sentiments.append(sentiment_analysis['vader'].get('sentiment', ''))
            
            if 'finbert' in sentiment_analysis:
                finbert_sentiments.append(sentiment_analysis['finbert'].get('sentiment', ''))
            
            model_comparison = result.get('model_comparison', {})
            if 'consensus_sentiment' in model_comparison:
                consensus_sentiments.append(model_comparison['consensus_sentiment'])
        
        # Display distributions
        for name, sentiments in [('VADER', vader_sentiments), ('FinBERT', finbert_sentiments), ('Consensus', consensus_sentiments)]:
            if sentiments:
                sentiment_counts = {}
                for s in sentiments:
                    sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
                
                print(f"🎯 {name} Sentiment Distribution:")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / len(sentiments)) * 100
                    print(f"   {sentiment}: {count} ({percentage:.1f}%)")
                print()
        
        # Show sample results
        print("📰 Sample Results:")
        for i, result in enumerate(results[:3], 1):
            article = result.get('article', {})
            sentiment_analysis = result.get('sentiment_analysis', {})
            
            print(f"   {i}. {article.get('title', 'No title')[:80]}...")
            
            if 'vader' in sentiment_analysis:
                vader = sentiment_analysis['vader']
                print(f"      VADER: {vader.get('sentiment', 'Unknown')} ({vader.get('confidence', 0):.2f})")
            
            if 'finbert' in sentiment_analysis:
                finbert = sentiment_analysis['finbert']
                print(f"      FinBERT: {finbert.get('sentiment', 'Unknown')} ({finbert.get('confidence', 0):.2f})")
        
        if len(results) > 3:
            print(f"   ... and {len(results) - 3} more results")
        
        print()

def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Financial Sentiment Analysis AI - Professional CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --live --keywords "Apple earnings" "Tesla stock"
  %(prog)s --custom "Apple stock surges after earnings beat"
  %(prog)s --compare "Federal Reserve announces rate cut"
  %(prog)s --history --days 30 --sentiment Positive
  %(prog)s --stats
        """
    )
    
    parser.add_argument('--version', action='version', version='Financial Sentiment AI 1.0.0')
    
    # Main operation modes
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument('--live', action='store_true',
                                help='Run live news sentiment analysis')
    operation_group.add_argument('--custom', type=str, nargs='+',
                                help='Analyze custom text inputs')
    operation_group.add_argument('--compare', type=str,
                                help='Compare VADER and FinBERT models on text')
    operation_group.add_argument('--history', action='store_true',
                                help='Query historical analysis results')
    operation_group.add_argument('--stats', action='store_true',
                                help='Show pipeline statistics')
    
    # Live analysis options
    parser.add_argument('--keywords', type=str, nargs='+',
                       default=['earnings', 'stock', 'market'],
                       help='Keywords for live news search (default: earnings, stock, market)')
    parser.add_argument('--max-articles', type=int, default=50,
                       help='Maximum articles to analyze (default: 50)')
    
    # Historical query options
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to look back (default: 7)')
    parser.add_argument('--sentiment', type=str, choices=['Positive', 'Negative', 'Neutral'],
                       help='Filter by sentiment')
    parser.add_argument('--limit', type=int, default=50,
                       help='Maximum results to return (default: 50)')
    
    # Output options
    parser.add_argument('--output', type=str, choices=['json', 'summary', 'detailed'],
                       default='summary', help='Output format (default: summary)')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results to database (default: True)')
    parser.add_argument('--no-save', action='store_false', dest='save',
                       help='Do not save results to database')
    parser.add_argument('--export', type=str,
                       help='Export results to file (specify filename)')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress non-essential output')
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize CLI
    cli = FinancialSentimentCLI()
    
    # Execute requested operation
    result = None
    
    try:
        if args.live:
            result = cli.run_live_analysis(
                keywords=args.keywords,
                max_articles=args.max_articles,
                save_results=args.save,
                output_format=args.output
            )
        
        elif args.custom:
            result = cli.run_custom_analysis(
                texts=args.custom,
                save_results=args.save,
                output_format=args.output
            )
        
        elif args.compare:
            result = cli.compare_models(args.compare)
        
        elif args.history:
            result = cli.query_historical_data(
                days=args.days,
                sentiment=args.sentiment,
                limit=args.limit
            )
        
        elif args.stats:
            stats = cli.pipeline.get_pipeline_stats()
            print("📊 Pipeline Statistics:")
            print("-" * 40)
            print(f"Total Processed: {stats['total_processed']}")
            print(f"Success Rate: {stats['success_rate']:.2%}")
            print(f"Avg Processing Time: {stats['average_processing_time']:.3f}s")
            print(f"Last Run: {stats['last_run']}")
            print(f"Components: {stats['components_status']}")
            result = {'success': True, 'stats': stats}
        
        # Export results if requested
        if result and result.get('success') and args.export and 'results' in result:
            export_path = cli.pipeline.export_results(
                result['results'],
                format='json' if args.export.endswith('.json') else 'csv',
                filename=args.export
            )
            if export_path:
                print(f"💾 Results exported to: {export_path}")
        
        # Output JSON if requested
        if args.output == 'json' and result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Exit with appropriate code
        sys.exit(0 if result and result.get('success', False) else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"CLI operation failed: {e}")
        print(f"❌ Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
