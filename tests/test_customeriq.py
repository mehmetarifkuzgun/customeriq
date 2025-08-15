import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processor import DataProcessor
from rfm_analyzer import RFMAnalyzer


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.data_processor = DataProcessor()
        
        self.sample_data = pd.DataFrame({
            'customer_id': [1, 1, 2, 2, 3],
            'order_date': [
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=10),
                datetime.now() - timedelta(days=60),
                datetime.now() - timedelta(days=5),
                datetime.now() - timedelta(days=100)
            ],
            'order_value': [100.0, 150.0, 200.0, 75.0, 300.0],
            'product_category': ['Electronics', 'Electronics', 'Clothing', 'Clothing', 'Books'],
            'quantity': [1, 2, 1, 1, 3]
        })
    
    def test_clean_transaction_data(self):
        """Test transaction data cleaning"""
        cleaned_data = self.data_processor.clean_transaction_data(self.sample_data)
        
        required_columns = ['customer_id', 'order_date', 'order_value']
        for col in required_columns:
            self.assertIn(col, cleaned_data.columns)
        
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_data['order_date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_data['order_value']))
        
        self.assertTrue((cleaned_data['order_value'] > 0).all())
    
    def test_create_customer_features(self):
        cleaned_data = self.data_processor.clean_transaction_data(self.sample_data)
        customer_features = self.data_processor.create_customer_features(cleaned_data)
        
        self.assertEqual(len(customer_features), self.sample_data['customer_id'].nunique())
        
        required_columns = [
            'first_purchase_date', 'last_purchase_date', 'total_orders',
            'total_spent', 'avg_order_value', 'days_since_last_purchase'
        ]
        for col in required_columns:
            self.assertIn(col, customer_features.columns)
        
        customer_1 = customer_features[customer_features['customer_id'] == 1]
        self.assertEqual(customer_1['total_orders'].iloc[0], 2)
        self.assertEqual(customer_1['total_spent'].iloc[0], 250.0)
        self.assertEqual(customer_1['avg_order_value'].iloc[0], 125.0)
    
    def test_create_sample_dataset(self):
        sample_data = self.data_processor.create_sample_dataset(n_customers=10)
        
        self.assertEqual(sample_data['customer_id'].nunique(), 10)
        self.assertTrue(len(sample_data) >= 10)
        
        self.assertTrue((sample_data['order_value'] > 0).all())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(sample_data['order_date']))


class TestRFMAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.rfm_analyzer = RFMAnalyzer()
        
        # Create test customer data
        self.customer_data = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'total_orders': [10, 5, 15, 2, 8],
            'total_spent': [1000.0, 250.0, 2000.0, 100.0, 600.0],
            'days_since_last_purchase': [10, 45, 5, 120, 30]
        })
    
    def test_calculate_rfm(self):
        rfm_data = self.rfm_analyzer.calculate_rfm(self.customer_data)
        
        required_columns = ['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score']
        for col in required_columns:
            self.assertIn(col, rfm_data.columns)
        
        for score_col in ['R_Score', 'F_Score', 'M_Score']:
            self.assertTrue((rfm_data[score_col] >= 1).all())
            self.assertTrue((rfm_data[score_col] <= 5).all())
        
        self.assertIn('RFM_Score', rfm_data.columns)
        self.assertIn('RFM_Score_Numeric', rfm_data.columns)
    
    def test_segment_customers(self):
        rfm_data = self.rfm_analyzer.calculate_rfm(self.customer_data)
        segmented_data, segment_stats = self.rfm_analyzer.segment_customers(rfm_data)
        
        self.assertIn('Segment', segmented_data.columns)
        self.assertIn('Segment_Description', segmented_data.columns)
        
        self.assertEqual(segmented_data['Segment'].isnull().sum(), 0)
        
        self.assertGreater(len(segment_stats), 0)
        required_stat_columns = ['customer_count', 'avg_monetary', 'customer_percentage']
        for col in required_stat_columns:
            self.assertIn(col, segment_stats.columns)
    
    def test_identify_high_value_customers(self):
        """Test high-value customer identification"""
        rfm_data = self.rfm_analyzer.calculate_rfm(self.customer_data)
        high_value = self.rfm_analyzer.identify_high_value_customers(rfm_data, top_percentage=0.4)
        
        expected_count = int(len(self.customer_data) * 0.4)
        self.assertEqual(len(high_value), expected_count)
        
        self.assertTrue(high_value['RFM_Score_Numeric'].is_monotonic_decreasing)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.data_processor = DataProcessor()
        self.rfm_analyzer = RFMAnalyzer()
    
    def test_end_to_end_workflow(self):
        raw_data = self.data_processor.create_sample_dataset(n_customers=50)
        
        cleaned_data = self.data_processor.clean_transaction_data(raw_data)
        self.assertGreater(len(cleaned_data), 0)
        
        customer_features = self.data_processor.create_customer_features(cleaned_data)
        self.assertEqual(len(customer_features), 50)
        
        rfm_data = self.rfm_analyzer.calculate_rfm(customer_features)
        self.assertEqual(len(rfm_data), 50)
        
        segmented_data, segment_stats = self.rfm_analyzer.segment_customers(rfm_data)
        self.assertEqual(len(segmented_data), 50)
        self.assertGreater(len(segment_stats), 0)
        
        segmented_with_actions, recommendations = self.rfm_analyzer.recommend_actions(segmented_data)
        self.assertGreater(len(recommendations), 0)
        
        self.assertIn('Recommended_Strategy', segmented_with_actions.columns)
        self.assertIn('Priority', segmented_with_actions.columns)


if __name__ == '__main__':
    test_suite = unittest.TestSuite()
    
    test_classes = [TestDataProcessor, TestRFMAnalyzer, TestIntegration]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
