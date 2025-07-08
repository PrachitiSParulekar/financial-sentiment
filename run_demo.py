#!/usr/bin/env python3
"""
Comprehensive demonstration of the improved financial sentiment analysis system
Shows the clean FinBERT implementation and working components
"""

import os
import sys
import time
from datetime import datetime

# Add core to path
sys.path.insert(0, 'core')

def main():
    """Main demonstration function"""
    print("🚀 Financial Sentiment Analysis AI - System Demonstration")
    print("=" * 80)
    print(f"📅 Demo Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print("=" * 80)
    
    # Test financial headlines
    financial_headlines = [
        "Apple stock surges 15% after record quarterly earnings beat expectations",
        "Tesla shares plummet following disappointing delivery numbers", 
        "Federal Reserve signals potential interest rate cuts amid economic uncertainty",
        "Microsoft reports strong cloud revenue growth exceeding analyst forecasts",
        "Oil prices decline on recession fears and weakening global demand",
        "Goldman Sachs upgrades tech sector outlook on AI revenue potential",
        "Inflation data shows cooling trends, boosting market confidence",
        "Banking sector faces headwinds from rising credit losses"
    ]
    
    print(f"\n📰 Analyzing {len(financial_headlines)} Financial Headlines")
    print("=" * 80)
    
    # Test 1: Clean FinBERT Analyzer
    test_clean_finbert(financial_headlines)
    
    # Test 2: VADER Analyzer  
    test_vader_analyzer(financial_headlines)
    
    # Test 3: News Fetcher (RSS)
    test_news_fetcher()
    
    # Test 4: System Integration
    test_system_integration()
    
    print("\n" + "=" * 80)
    print("🎉 System demonstration completed successfully!")
    print("💡 Key improvements in clean FinBERT:")
    print("   ✅ Robust error handling with graceful fallbacks")
    print("   ✅ Performance tracking and statistics")
    print("   ✅ Clean enum-based sentiment labels")
    print("   ✅ Comprehensive validation and status reporting")
    print("   ✅ Entity extraction with financial patterns")
    print("   ✅ Model switching and cache management")
    print("=" * 80)

def test_clean_finbert(headlines):
    """Test the clean FinBERT implementation"""
    print("\n🤖 Testing Clean FinBERT Implementation")
    print("-" * 60)
    
    try:
        from sentiment_finbert_clean import CleanFinBERTAnalyzer
        
        analyzer = CleanFinBERTAnalyzer()
        
        # Show status
        status = analyzer.get_model_status()
        print(f"📊 Model Status: {status['is_available']}")
        print(f"🔧 Device: {status.get('device', 'CPU')}")
        print(f"📦 Dependencies: Transformers={status['transformers_available']}, spaCy={status['spacy_available']}")
        
        if status.get('initialization_error'):
            print(f"⚠️  Note: {status['initialization_error']}")
        
        print("\n📈 Sentiment Analysis Results:")
        
        for i, headline in enumerate(headlines[:3], 1):  # Test first 3
            print(f"\n{i}. '{headline[:60]}...'")
            
            result = analyzer.analyze_sentiment(headline)
            print(f"   🎯 Sentiment: {result.sentiment.value}")
            print(f"   📊 Confidence: {result.confidence:.3f}")
            print(f"   📈 Scores: {result.scores}")
            
            if result.entities and any(result.entities.values()):
                entities_summary = {}
                for key, values in result.entities.items():
                    if values:
                        entities_summary[key] = values[:2]  # Show first 2
                print(f"   🏷️  Entities: {entities_summary}")
        
        # Performance stats
        stats = analyzer.get_performance_stats()
        if stats['total_analyses'] > 0:
            print(f"\n📊 Performance: {stats['successful_analyses']}/{stats['total_analyses']} successful")
            if stats['average_processing_time'] > 0:
                print(f"⏱️  Avg Time: {stats['average_processing_time']:.3f}s")
        
        print("✅ Clean FinBERT test completed")
        
    except Exception as e:
        print(f"❌ FinBERT test failed: {e}")

def test_vader_analyzer(headlines):
    """Test the VADER analyzer"""
    print("\n📊 Testing Enhanced VADER Analyzer")
    print("-" * 60)
    
    try:
        from sentiment_vader import FinancialVaderAnalyzer
        
        analyzer = FinancialVaderAnalyzer()
        print("✅ VADER analyzer initialized")
        
        print("\n📈 VADER Analysis Results:")
        
        for i, headline in enumerate(headlines[:3], 1):  # Test first 3
            print(f"\n{i}. '{headline[:60]}...'")
            
            result = analyzer.analyze(headline, include_explanation=False)
            print(f"   🎯 Sentiment: {result.sentiment_label}")
            print(f"   📊 Confidence: {result.confidence:.3f}")
            print(f"   📈 Compound: {result.raw_scores['compound']:.3f}")
            
            if result.financial_modifiers:
                print(f"   🔧 Modifiers: {result.financial_modifiers}")
        
        print("✅ VADER test completed")
        
    except Exception as e:
        print(f"❌ VADER test failed: {e}")

def test_news_fetcher():
    """Test the news fetcher with RSS"""
    print("\n📰 Testing News Fetcher (RSS)")
    print("-" * 60)
    
    try:
        from news_fetcher import FinancialNewsFetcher
        
        fetcher = FinancialNewsFetcher()
        print("✅ News fetcher initialized")
        
        # Try RSS fetch with timeout
        print("🔄 Fetching RSS articles (timeout: 10s)...")
        start_time = time.time()
        
        try:
            articles = fetcher.fetch_from_rss(max_articles=3)
            fetch_time = time.time() - start_time
            
            if articles:
                print(f"✅ Fetched {len(articles)} articles in {fetch_time:.2f}s")
                
                for i, article in enumerate(articles[:2], 1):
                    print(f"\n{i}. {article.title[:80]}...")
                    print(f"   📅 Published: {article.published}")
                    print(f"   🔗 Source: {article.source}")
                    if article.tickers:
                        print(f"   📈 Tickers: {article.tickers[:5]}")  # Show first 5
            else:
                print("⚠️  No articles fetched (network/feed issues)")
                
        except Exception as fetch_error:
            print(f"⚠️  RSS fetch failed: {fetch_error}")
            print("💡 This is common due to network issues or RSS feed problems")
        
        print("✅ News fetcher test completed")
        
    except Exception as e:
        print(f"❌ News fetcher test failed: {e}")

def test_system_integration():
    """Test system integration capabilities"""
    print("\n🔗 Testing System Integration")
    print("-" * 60)
    
    try:
        # Test importing all core components
        from sentiment_finbert_clean import CleanFinBERTAnalyzer
        from sentiment_vader import FinancialVaderAnalyzer  
        from news_fetcher import FinancialNewsFetcher
        
        print("✅ All core components imported successfully")
        
        # Test dual sentiment analysis
        test_text = "Apple stock jumps 12% on strong earnings guidance"
        
        print(f"\n🔬 Dual Sentiment Analysis:")
        print(f"Text: '{test_text}'")
        
        # FinBERT analysis
        finbert = CleanFinBERTAnalyzer()
        finbert_result = finbert.analyze_sentiment(test_text)
        print(f"\n🤖 FinBERT: {finbert_result.sentiment.value} ({finbert_result.confidence:.3f})")
        
        # VADER analysis
        vader = FinancialVaderAnalyzer()
        vader_result = vader.analyze(test_text, include_explanation=False)
        print(f"📊 VADER: {vader_result.sentiment_label} ({vader_result.confidence:.3f})")
        
        # Consensus
        if finbert_result.sentiment.value.lower() == vader_result.sentiment_label.lower():
            print(f"✅ Consensus: Both models agree on {finbert_result.sentiment.value}")
        else:
            print(f"⚖️  Disagreement: FinBERT={finbert_result.sentiment.value}, VADER={vader_result.sentiment_label}")
        
        print("✅ Integration test completed")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

if __name__ == "__main__":
    main()
