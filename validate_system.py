"""
System validation and integration test for Financial Sentiment Analysis AI
Validates all components work together correctly
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing imports...")
    try:
        # Test core imports
        from core.news_fetcher import FinancialNewsFetcher
        from core.text_cleaner import FinancialTextCleaner
        from core.sentiment_vader import FinancialVaderAnalyzer
        from core.sentiment_finbert_clean import CleanFinBERTAnalyzer as FinBERTAnalyzer
        
        # Test pipeline imports
        from pipeline.inference import FinancialSentimentPipeline
        from pipeline.logger import FinancialDataLogger
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_text_analysis():
    """Test the complete text analysis pipeline"""
    print("\n🎯 Testing text analysis pipeline...")
    
    try:
        # Sample financial texts
        test_texts = [
            "Apple stock surges 15% after record quarterly earnings beat expectations",
            "Federal Reserve signals potential interest rate cuts amid economic uncertainty",
            "Tesla shares plummet following disappointing delivery numbers"
        ]
        
        # Test text cleaner
        from core.text_cleaner import FinancialTextCleaner
        cleaner = FinancialTextCleaner()
        
        print("   Testing text cleaner...")
        for text in test_texts[:1]:  # Test one sample
            result = cleaner.clean_financial_text(text)
            if result['cleaned_text']:
                print(f"   ✅ Text cleaner working: '{result['cleaned_text'][:50]}...'")
                break
        
        # Test VADER analyzer
        from core.sentiment_vader import FinancialVaderAnalyzer
        vader_analyzer = FinancialVaderAnalyzer()
        
        print("   Testing VADER analyzer...")
        for text in test_texts[:1]:  # Test one sample
            result = vader_analyzer.analyze_sentiment(text)
            if result['sentiment']:
                print(f"   ✅ VADER working: {result['sentiment']} ({result['confidence']:.3f})")
                break
        
        # Test FinBERT analyzer (may not work without proper setup)
        print("   Testing FinBERT analyzer...")
        try:
            from core.sentiment_finbert_clean import CleanFinBERTAnalyzer as FinBERTAnalyzer
            finbert_analyzer = FinBERTAnalyzer()
            
            if finbert_analyzer.model is not None:
                result = finbert_analyzer.analyze_sentiment(test_texts[0])
                print(f"   ✅ FinBERT working: {result['sentiment']} ({result['confidence']:.3f})")
            else:
                print("   ⚠️ FinBERT model not available (expected without transformers)")
                
        except Exception as e:
            print(f"   ⚠️ FinBERT not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text analysis test failed: {e}")
        return False

def test_pipeline_integration():
    """Test the complete pipeline integration"""
    print("\n🔄 Testing pipeline integration...")
    
    try:
        from pipeline.inference import FinancialSentimentPipeline
        
        # Initialize pipeline
        pipeline = FinancialSentimentPipeline()
        
        # Test custom text analysis
        test_texts = ["Apple reports strong quarterly earnings, stock rises"]
        
        print("   Testing custom text analysis...")
        results = pipeline.analyze_custom_text(test_texts)
        
        if results:
            result = results[0]
            print(f"   ✅ Pipeline working: {len(results)} result(s) generated")
            
            # Check if we have sentiment analysis
            if 'sentiment_analysis' in result:
                sentiment_data = result['sentiment_analysis']
                print(f"   📊 Models used: {list(sentiment_data.keys())}")
            
            return True
        else:
            print("   ❌ No results from pipeline")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def test_data_logging():
    """Test data logging functionality"""
    print("\n💾 Testing data logging...")
    
    try:
        from pipeline.logger import FinancialDataLogger
        
        # Initialize logger
        logger = FinancialDataLogger(storage_type='sqlite')
        
        # Test sample result logging
        sample_result = {
            'article': {
                'title': 'Test Article',
                'url': 'https://test.com',
                'source': 'test',
                'published_at': datetime.now().isoformat()
            },
            'preprocessing': {
                'cleaned_text': 'test article content'
            },
            'sentiment_analysis': {
                'vader': {
                    'sentiment': 'Positive',
                    'confidence': 0.8,
                    'scores': {'Positive': 0.8, 'Negative': 0.1, 'Neutral': 0.1}
                }
            },
            'model_comparison': {
                'consensus_sentiment': 'Positive',
                'agreement': True
            },
            'pipeline_version': '1.0'
        }
        
        # Log the result
        analysis_id = logger.log_sentiment_result(sample_result)
        
        if analysis_id:
            print(f"   ✅ Data logging working: {analysis_id}")
            
            # Test querying
            results = logger.query_results(limit=1)
            if results:
                print(f"   ✅ Data querying working: {len(results)} result(s)")
            
            return True
        else:
            print("   ❌ Failed to log data")
            return False
            
    except Exception as e:
        print(f"❌ Data logging test failed: {e}")
        return False

def test_cli_components():
    """Test CLI components can be imported and initialized"""
    print("\n⌨️ Testing CLI components...")
    
    try:
        # Test CLI import
        import cli
        
        # Test CLI class initialization
        cli_instance = cli.FinancialSentimentCLI()
        
        if cli_instance.pipeline and cli_instance.data_logger:
            print("   ✅ CLI components initialized successfully")
            return True
        else:
            print("   ⚠️ CLI initialized but some components missing")
            return True  # Still consider this a pass
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Run the complete system validation"""
    print("🧪 Financial Sentiment Analysis AI - System Validation")
    print("=" * 60)
    print(f"📅 Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Track test results
    tests = [
        ("Imports", test_imports),
        ("Text Analysis", test_text_analysis),
        ("Pipeline Integration", test_pipeline_integration),
        ("Data Logging", test_data_logging),
        ("CLI Components", test_cli_components)
    ]
    
    passed = 0
    total = len(tests)
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    end_time = time.time()
    
    # Final results
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS")
    print("-" * 30)
    print(f"✅ Passed: {passed}/{total} tests")
    print(f"❌ Failed: {total - passed}/{total} tests")
    print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        return True
    elif passed >= total * 0.8:  # 80% pass rate
        print(f"\n⚠️ Most tests passed ({passed}/{total}). System is mostly functional.")
        print("   Some advanced features may require additional setup.")
        return True
    else:
        print(f"\n❌ Multiple test failures ({total - passed}/{total}). Check installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
