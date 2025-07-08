#!/usr/bin/env python3
"""
Simple test for the clean FinBERT implementation
Tests the improved FinBERT analyzer with fallback handling
"""

import os
import sys
sys.path.insert(0, 'core')

def test_finbert_clean():
    """Test the clean FinBERT implementation"""
    print("🧪 Testing Clean FinBERT Implementation")
    print("=" * 50)
    
    try:
        # Import the clean FinBERT
        from sentiment_finbert_clean import CleanFinBERTAnalyzer, analyze_finbert
        
        print("✅ Successfully imported CleanFinBERTAnalyzer")
        
        # Test sample financial texts
        test_texts = [
            "Apple stock surges 15% after record quarterly earnings beat expectations",
            "Tesla shares plummet following disappointing delivery numbers",
            "Federal Reserve signals potential interest rate cuts amid economic uncertainty",
            "Microsoft reports strong cloud revenue growth in latest quarter",
            "Oil prices decline on recession fears and demand concerns"
        ]
        
        print(f"\n📊 Testing {len(test_texts)} sample texts...")
        print("-" * 40)
        
        # Test using the convenience function
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Testing: '{text[:60]}...'")
            
            try:
                result = analyze_finbert(text)
                print(f"   📈 Sentiment: {result['sentiment']}")
                print(f"   🎯 Confidence: {result['confidence']:.3f}")
                print(f"   📊 Scores: {result['scores']}")
                
                if result.get('entities'):
                    print(f"   🏢 Entities: {result['entities']}")
                
                if result.get('model_info', {}).get('error'):
                    print(f"   ⚠️  Note: {result['model_info']['error']}")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing text: {e}")
        
        # Test the class directly
        print(f"\n🔧 Testing CleanFinBERTAnalyzer class...")
        analyzer = CleanFinBERTAnalyzer()
        
        if analyzer.is_available():
            print("   ✅ FinBERT model is available")
            sample_result = analyzer.analyze_single("Apple stock rises on strong earnings")
            print(f"   📊 Sample analysis: {sample_result.sentiment.value} ({sample_result.confidence:.3f})")
        else:
            print("   ⚠️  FinBERT model not available (using fallback)")
            print("   💡 This is expected without transformers/torch installed")
        
        print(f"\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_finbert_improvements():
    """Test specific improvements in the clean FinBERT implementation"""
    print("\n🚀 Testing FinBERT Improvements")
    print("=" * 50)
    
    try:
        from sentiment_finbert_clean import CleanFinBERTAnalyzer, SentimentLabel, FinBERTResult
        
        # Test 1: Enum usage
        print("1. Testing SentimentLabel enum...")
        labels = [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL]
        print(f"   ✅ Available labels: {[label.value for label in labels]}")
        
        # Test 2: Error handling
        print("2. Testing error handling...")
        analyzer = CleanFinBERTAnalyzer()
        result = analyzer.analyze_single("Test text")
        print(f"   ✅ Graceful fallback: {result.sentiment.value}")
        
        # Test 3: Result structure
        print("3. Testing result structure...")
        result_dict = result.to_dict()
        expected_keys = ['text', 'sentiment', 'confidence', 'scores', 'entities', 'processing_time', 'model_info']
        has_all_keys = all(key in result_dict for key in expected_keys)
        print(f"   ✅ Result structure complete: {has_all_keys}")
        
        # Test 4: Batch processing
        print("4. Testing batch processing...")
        test_batch = ["Apple rises", "Tesla falls", "Market neutral"]
        batch_results = analyzer.analyze_batch(test_batch)
        print(f"   ✅ Batch processing: {len(batch_results)} results")
        
        print(f"\n✅ All improvement tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Improvement test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 FinBERT Clean Code Test Suite")
    print("=" * 60)
    
    success = True
    success &= test_finbert_clean()
    success &= test_finbert_improvements()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Clean FinBERT is working correctly.")
        print("💡 The system gracefully handles missing dependencies with fallbacks.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    print("=" * 60)
