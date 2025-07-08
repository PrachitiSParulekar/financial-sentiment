"""
Advanced data logging and storage for financial sentiment analysis results
Supports multiple backends: SQLite, MongoDB, JSON with model versioning and drift monitoring
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataLogger:
    """
    Advanced data logger with multiple backend support and model tracking
    """
    
    def __init__(self, storage_type: str = 'sqlite', connection_string: Optional[str] = None):
        """
        Initialize data logger
        
        Args:
            storage_type: Type of storage backend ('sqlite', 'mongodb', 'json')
            connection_string: Connection string for database (optional)
        """
        self.storage_type = storage_type.lower()
        self.connection_string = connection_string
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Model versioning
        self.model_versions = {
            'vader': '1.0',
            'finbert': '1.0',
            'pipeline': '1.0'
        }
        
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize the storage backend"""
        if self.storage_type == 'sqlite':
            self._initialize_sqlite()
        elif self.storage_type == 'mongodb':
            self._initialize_mongodb()
        elif self.storage_type == 'json':
            self._initialize_json()
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    def _initialize_sqlite(self):
        """Initialize SQLite database with schema"""
        db_path = self.data_dir / "financial_sentiment.db"
        self.db_path = str(db_path)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create main results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    article_title TEXT,
                    article_url TEXT,
                    article_source TEXT,
                    published_at DATETIME,
                    cleaned_text TEXT,
                    vader_sentiment TEXT,
                    vader_confidence REAL,
                    vader_scores TEXT,
                    finbert_sentiment TEXT,
                    finbert_confidence REAL,
                    finbert_scores TEXT,
                    consensus_sentiment TEXT,
                    model_agreement BOOLEAN,
                    entities TEXT,
                    processing_time REAL,
                    pipeline_version TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create model performance tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    accuracy REAL,
                    precision_positive REAL,
                    precision_negative REAL,
                    precision_neutral REAL,
                    recall_positive REAL,
                    recall_negative REAL,
                    recall_neutral REAL,
                    f1_score REAL,
                    processing_speed REAL,
                    total_predictions INTEGER,
                    confidence_avg REAL,
                    drift_score REAL
                )
            ''')
            
            # Create model drift monitoring table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_drift (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    drift_metric TEXT NOT NULL,
                    drift_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    alert_triggered BOOLEAN DEFAULT FALSE,
                    window_start DATETIME,
                    window_end DATETIME,
                    sample_size INTEGER
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON sentiment_results(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_id ON sentiment_results(analysis_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_performance_timestamp ON model_performance(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_drift_timestamp ON model_drift(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ SQLite database initialized: {db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    def _initialize_mongodb(self):
        """Initialize MongoDB connection (placeholder for future implementation)"""
        try:
            # Try to import pymongo
            import pymongo
            
            # Connection would go here
            self.mongo_client = None
            self.mongo_db = None
            
            logger.info("MongoDB backend available but not implemented yet")
            
        except ImportError:
            logger.warning("MongoDB support requires pymongo: pip install pymongo")
            self.storage_type = 'json'  # Fallback to JSON
            self._initialize_json()
    
    def _initialize_json(self):
        """Initialize JSON file storage"""
        self.json_dir = self.data_dir / "json_logs"
        self.json_dir.mkdir(exist_ok=True)
        
        self.results_file = self.json_dir / "sentiment_results.jsonl"
        self.performance_file = self.json_dir / "model_performance.jsonl"
        self.drift_file = self.json_dir / "model_drift.jsonl"
        
        logger.info(f"✅ JSON storage initialized: {self.json_dir}")
    
    def log_sentiment_result(self, result: Dict[str, Any]) -> str:
        """
        Log a single sentiment analysis result
        
        Args:
            result: Complete sentiment analysis result
            
        Returns:
            Analysis ID for the logged result
        """
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Add metadata
        enriched_result = {
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'model_versions': self.model_versions.copy(),
            **result
        }
        
        try:
            if self.storage_type == 'sqlite':
                self._log_sqlite(enriched_result)
            elif self.storage_type == 'json':
                self._log_json(enriched_result)
            elif self.storage_type == 'mongodb':
                self._log_mongodb(enriched_result)
            
            logger.info(f"📝 Logged result with ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Failed to log result: {e}")
            return ""
    
    def _log_sqlite(self, result: Dict[str, Any]):
        """Log result to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Extract article info
            article = result.get('article', {})
            preprocessing = result.get('preprocessing', {})
            sentiment_analysis = result.get('sentiment_analysis', {})
            model_comparison = result.get('model_comparison', {})
            
            # Extract VADER results
            vader_result = sentiment_analysis.get('vader', {})
            vader_sentiment = vader_result.get('sentiment', '')
            vader_confidence = vader_result.get('confidence', 0.0)
            vader_scores = json.dumps(vader_result.get('scores', {}))
            
            # Extract FinBERT results
            finbert_result = sentiment_analysis.get('finbert', {})
            finbert_sentiment = finbert_result.get('sentiment', '')
            finbert_confidence = finbert_result.get('confidence', 0.0)
            finbert_scores = json.dumps(finbert_result.get('scores', {}))
            
            # Extract consensus
            consensus_sentiment = model_comparison.get('consensus_sentiment', '')
            model_agreement = model_comparison.get('agreement', False)
            
            # Extract entities
            entities = json.dumps(finbert_result.get('entities', {}))
            
            cursor.execute('''
                INSERT INTO sentiment_results (
                    analysis_id, article_title, article_url, article_source,
                    published_at, cleaned_text, vader_sentiment, vader_confidence,
                    vader_scores, finbert_sentiment, finbert_confidence, finbert_scores,
                    consensus_sentiment, model_agreement, entities, pipeline_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['analysis_id'],
                article.get('title', ''),
                article.get('url', ''),
                article.get('source', ''),
                article.get('published_at', ''),
                preprocessing.get('cleaned_text', ''),
                vader_sentiment,
                vader_confidence,
                vader_scores,
                finbert_sentiment,
                finbert_confidence,
                finbert_scores,
                consensus_sentiment,
                model_agreement,
                entities,
                result.get('pipeline_version', '1.0')
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"SQLite logging failed: {e}")
            raise
        finally:
            conn.close()
    
    def _log_json(self, result: Dict[str, Any]):
        """Log result to JSON file"""
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def _log_mongodb(self, result: Dict[str, Any]):
        """Log result to MongoDB (placeholder)"""
        # MongoDB implementation would go here
        pass
    
    def log_batch_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Log multiple sentiment analysis results
        
        Args:
            results: List of sentiment analysis results
            
        Returns:
            List of analysis IDs
        """
        analysis_ids = []
        
        for result in results:
            analysis_id = self.log_sentiment_result(result)
            if analysis_id:
                analysis_ids.append(analysis_id)
        
        logger.info(f"📝 Logged {len(analysis_ids)} results in batch")
        return analysis_ids
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """
        Log model performance metrics
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics dictionary
        """
        performance_record = {
            'model_name': model_name,
            'model_version': self.model_versions.get(model_name, '1.0'),
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        try:
            if self.storage_type == 'sqlite':
                self._log_performance_sqlite(performance_record)
            elif self.storage_type == 'json':
                with open(self.performance_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(performance_record, ensure_ascii=False) + '\n')
            
            logger.info(f"📊 Logged performance metrics for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log performance: {e}")
    
    def _log_performance_sqlite(self, record: Dict[str, Any]):
        """Log performance to SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO model_performance (
                    model_name, model_version, accuracy, precision_positive,
                    precision_negative, precision_neutral, recall_positive,
                    recall_negative, recall_neutral, f1_score, processing_speed,
                    total_predictions, confidence_avg, drift_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.get('model_name', ''),
                record.get('model_version', ''),
                record.get('accuracy', 0.0),
                record.get('precision_positive', 0.0),
                record.get('precision_negative', 0.0),
                record.get('precision_neutral', 0.0),
                record.get('recall_positive', 0.0),
                record.get('recall_negative', 0.0),
                record.get('recall_neutral', 0.0),
                record.get('f1_score', 0.0),
                record.get('processing_speed', 0.0),
                record.get('total_predictions', 0),
                record.get('confidence_avg', 0.0),
                record.get('drift_score', 0.0)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Performance logging to SQLite failed: {e}")
            raise
        finally:
            conn.close()
    
    def log_model_drift(self, model_name: str, drift_metric: str, drift_value: float, 
                       threshold: float, window_start: datetime, window_end: datetime, 
                       sample_size: int):
        """
        Log model drift detection results
        
        Args:
            model_name: Name of the model
            drift_metric: Type of drift metric (e.g., 'distribution_shift', 'confidence_drop')
            drift_value: Calculated drift value
            threshold: Threshold for alerting
            window_start: Start of analysis window
            window_end: End of analysis window
            sample_size: Number of samples in analysis
        """
        alert_triggered = drift_value > threshold
        
        drift_record = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'drift_metric': drift_metric,
            'drift_value': drift_value,
            'threshold': threshold,
            'alert_triggered': alert_triggered,
            'window_start': window_start.isoformat(),
            'window_end': window_end.isoformat(),
            'sample_size': sample_size
        }
        
        try:
            if self.storage_type == 'sqlite':
                self._log_drift_sqlite(drift_record)
            elif self.storage_type == 'json':
                with open(self.drift_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(drift_record, ensure_ascii=False) + '\n')
            
            if alert_triggered:
                logger.warning(f"🚨 Model drift alert for {model_name}: {drift_metric} = {drift_value:.4f} > {threshold}")
            else:
                logger.info(f"📈 Model drift logged for {model_name}: {drift_metric} = {drift_value:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to log drift: {e}")
    
    def _log_drift_sqlite(self, record: Dict[str, Any]):
        """Log drift to SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO model_drift (
                    model_name, drift_metric, drift_value, threshold,
                    alert_triggered, window_start, window_end, sample_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record['model_name'],
                record['drift_metric'],
                record['drift_value'],
                record['threshold'],
                record['alert_triggered'],
                record['window_start'],
                record['window_end'],
                record['sample_size']
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Drift logging to SQLite failed: {e}")
            raise
        finally:
            conn.close()
    
    def query_results(self, start_date: Optional[datetime] = None, 
                     end_date: Optional[datetime] = None,
                     model_name: Optional[str] = None,
                     sentiment: Optional[str] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query stored sentiment analysis results
        
        Args:
            start_date: Filter results from this date
            end_date: Filter results until this date
            model_name: Filter by specific model
            sentiment: Filter by sentiment ('Positive', 'Negative', 'Neutral')
            limit: Maximum number of results
            
        Returns:
            List of matching results
        """
        if self.storage_type == 'sqlite':
            return self._query_sqlite(start_date, end_date, model_name, sentiment, limit)
        elif self.storage_type == 'json':
            return self._query_json(start_date, end_date, model_name, sentiment, limit)
        else:
            return []
    
    def _query_sqlite(self, start_date, end_date, model_name, sentiment, limit):
        """Query SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        cursor = conn.cursor()
        
        try:
            # Build query
            query = "SELECT * FROM sentiment_results WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            if sentiment:
                query += " AND (vader_sentiment = ? OR finbert_sentiment = ?)"
                params.extend([sentiment, sentiment])
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            results = []
            for row in rows:
                result = dict(row)
                # Parse JSON fields
                if result['vader_scores']:
                    result['vader_scores'] = json.loads(result['vader_scores'])
                if result['finbert_scores']:
                    result['finbert_scores'] = json.loads(result['finbert_scores'])
                if result['entities']:
                    result['entities'] = json.loads(result['entities'])
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"SQLite query failed: {e}")
            return []
        finally:
            conn.close()
    
    def _query_json(self, start_date, end_date, model_name, sentiment, limit):
        """Query JSON files"""
        results = []
        
        try:
            if not self.results_file.exists():
                return results
            
            with open(self.results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(results) >= limit:
                        break
                    
                    try:
                        result = json.loads(line.strip())
                        
                        # Apply filters
                        result_timestamp = datetime.fromisoformat(result['timestamp'])
                        
                        if start_date and result_timestamp < start_date:
                            continue
                        if end_date and result_timestamp > end_date:
                            continue
                        
                        if sentiment:
                            sentiment_analysis = result.get('sentiment_analysis', {})
                            vader_sentiment = sentiment_analysis.get('vader', {}).get('sentiment', '')
                            finbert_sentiment = sentiment_analysis.get('finbert', {}).get('sentiment', '')
                            
                            if sentiment not in [vader_sentiment, finbert_sentiment]:
                                continue
                        
                        results.append(result)
                        
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"JSON query failed: {e}")
            return []
    
    def get_model_statistics(self, model_name: Optional[str] = None, 
                           days: int = 30) -> Dict[str, Any]:
        """
        Get model performance statistics
        
        Args:
            model_name: Specific model to analyze (None for all)
            days: Number of days to look back
            
        Returns:
            Statistics dictionary
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        results = self.query_results(start_date=start_date, end_date=end_date)
        
        if not results:
            return {'total_predictions': 0, 'models': {}}
        
        stats = {
            'total_predictions': len(results),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'models': {}
        }
        
        # Analyze by model
        for model in ['vader', 'finbert']:
            model_results = []
            
            for result in results:
                if self.storage_type == 'sqlite':
                    # SQLite format
                    if model == 'vader' and result.get('vader_sentiment'):
                        model_results.append({
                            'sentiment': result['vader_sentiment'],
                            'confidence': result['vader_confidence']
                        })
                    elif model == 'finbert' and result.get('finbert_sentiment'):
                        model_results.append({
                            'sentiment': result['finbert_sentiment'],
                            'confidence': result['finbert_confidence']
                        })
                else:
                    # JSON format
                    sentiment_analysis = result.get('sentiment_analysis', {})
                    if model in sentiment_analysis:
                        model_data = sentiment_analysis[model]
                        model_results.append({
                            'sentiment': model_data.get('sentiment', ''),
                            'confidence': model_data.get('confidence', 0)
                        })
            
            if model_results:
                # Calculate statistics
                sentiments = [r['sentiment'] for r in model_results]
                confidences = [r['confidence'] for r in model_results]
                
                sentiment_counts = {}
                for s in sentiments:
                    sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
                
                stats['models'][model] = {
                    'total_predictions': len(model_results),
                    'sentiment_distribution': sentiment_counts,
                    'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
                    'min_confidence': min(confidences) if confidences else 0,
                    'max_confidence': max(confidences) if confidences else 0
                }
        
        return stats

# Test harness
def test_data_logger():
    """Test the data logger with sample data"""
    print("🧪 Testing Financial Data Logger")
    print("=" * 50)
    
    # Initialize logger
    logger_instance = FinancialDataLogger(storage_type='sqlite')
    
    # Test data
    sample_result = {
        'article': {
            'title': 'Apple stock surges after earnings',
            'url': 'https://example.com/apple-earnings',
            'source': 'test_source',
            'published_at': datetime.now().isoformat()
        },
        'preprocessing': {
            'cleaned_text': 'apple stock surges earnings'
        },
        'sentiment_analysis': {
            'vader': {
                'sentiment': 'Positive',
                'confidence': 0.85,
                'scores': {'Positive': 0.85, 'Negative': 0.05, 'Neutral': 0.10}
            },
            'finbert': {
                'sentiment': 'Positive',
                'confidence': 0.92,
                'scores': {'Positive': 0.92, 'Negative': 0.03, 'Neutral': 0.05},
                'entities': {'organizations': ['Apple'], 'tickers': ['AAPL']}
            }
        },
        'model_comparison': {
            'consensus_sentiment': 'Positive',
            'agreement': True
        },
        'pipeline_version': '1.0'
    }
    
    # Test 1: Log single result
    print("📝 Test 1: Logging single result")
    analysis_id = logger_instance.log_sentiment_result(sample_result)
    print(f"✅ Logged with ID: {analysis_id}")
    
    # Test 2: Log model performance
    print("\n📊 Test 2: Logging model performance")
    performance_metrics = {
        'accuracy': 0.85,
        'f1_score': 0.82,
        'processing_speed': 0.15,
        'total_predictions': 100,
        'confidence_avg': 0.78
    }
    logger_instance.log_model_performance('vader', performance_metrics)
    print("✅ Performance metrics logged")
    
    # Test 3: Log model drift
    print("\n📈 Test 3: Logging model drift")
    now = datetime.now()
    window_start = now - timedelta(hours=24)
    logger_instance.log_model_drift(
        'vader', 
        'confidence_drop', 
        0.15, 
        0.20, 
        window_start, 
        now, 
        100
    )
    print("✅ Model drift logged")
    
    # Test 4: Query results
    print("\n🔍 Test 4: Querying results")
    results = logger_instance.query_results(limit=5)
    print(f"✅ Retrieved {len(results)} results")
    
    # Test 5: Get statistics
    print("\n📊 Test 5: Getting statistics")
    stats = logger_instance.get_model_statistics(days=7)
    print(f"✅ Stats: {stats['total_predictions']} predictions")
    print(f"   Models: {list(stats['models'].keys())}")
    
    print(f"\n✅ Data logger test completed!")

if __name__ == "__main__":
    test_data_logger()
