import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from config.settings import PROCESSED_DATA_DIR
from utils.metrics import DataQualityMetrics

class DataProcessor:
    def __init__(self):
        self.setup_logging()
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_raw_data(self, file_path, file_type='csv'):
        try:
            file_path = Path(file_path)
            
            if file_type.lower() == 'csv' or file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path)
            elif file_type.lower() in ['excel', 'xlsx'] or file_path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def validate_required_columns(self, data, required_columns):
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.logger.info("All required columns are present")
        return True
    
    def clean_transaction_data(self, data):
        self.logger.info("Starting transaction data cleaning...")
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Required columns for transaction data
        required_columns = ['customer_id', 'order_date', 'order_value']
        self.validate_required_columns(cleaned_data, required_columns)
        
        # Convert date columns
        cleaned_data['order_date'] = pd.to_datetime(cleaned_data['order_date'], errors='coerce')
        
        # Remove rows with invalid dates
        invalid_dates = cleaned_data['order_date'].isnull()
        if invalid_dates.sum() > 0:
            self.logger.warning(f"Removing {invalid_dates.sum()} rows with invalid dates")
            cleaned_data = cleaned_data[~invalid_dates]
        
        # Remove negative or zero order values
        invalid_values = (cleaned_data['order_value'] <= 0)
        if invalid_values.sum() > 0:
            self.logger.warning(f"Removing {invalid_values.sum()} rows with invalid order values")
            cleaned_data = cleaned_data[~invalid_values]
        
        # Remove future dates
        future_dates = cleaned_data['order_date'] > datetime.now()
        if future_dates.sum() > 0:
            self.logger.warning(f"Removing {future_dates.sum()} rows with future dates")
            cleaned_data = cleaned_data[~future_dates]
        
        # Handle missing customer IDs
        missing_customer_ids = cleaned_data['customer_id'].isnull()
        if missing_customer_ids.sum() > 0:
            self.logger.warning(f"Removing {missing_customer_ids.sum()} rows with missing customer IDs")
            cleaned_data = cleaned_data[~missing_customer_ids]
        
        # Ensure customer_id is integer
        cleaned_data['customer_id'] = cleaned_data['customer_id'].astype(int)
        
        # Add derived fields
        cleaned_data['year'] = cleaned_data['order_date'].dt.year
        cleaned_data['month'] = cleaned_data['order_date'].dt.month
        cleaned_data['day_of_week'] = cleaned_data['order_date'].dt.dayofweek
        cleaned_data['quarter'] = cleaned_data['order_date'].dt.quarter
        
        # Handle product categories if present
        if 'product_category' in cleaned_data.columns:
            cleaned_data['product_category'] = cleaned_data['product_category'].fillna('Unknown')
        
        # Handle quantities if present
        if 'quantity' in cleaned_data.columns:
            cleaned_data['quantity'] = cleaned_data['quantity'].fillna(1)
            cleaned_data['quantity'] = np.maximum(cleaned_data['quantity'], 1)  # Ensure positive
        
        self.logger.info(f"Transaction data cleaning completed. Final shape: {cleaned_data.shape}")
        return cleaned_data
    
    def create_customer_features(self, transaction_data):
        self.logger.info("Creating customer features...")
        
        # Ensure data is sorted by date
        transaction_data = transaction_data.sort_values(['customer_id', 'order_date'])
        
        # Calculate reference date (latest date in dataset)
        reference_date = transaction_data['order_date'].max()
        
        # Customer aggregations
        customer_features = transaction_data.groupby('customer_id').agg({
            'order_date': ['min', 'max', 'count'],
            'order_value': ['sum', 'mean', 'std', 'min', 'max']
        }).round(2)
        
        # Flatten column names
        customer_features.columns = [
            'first_purchase_date', 'last_purchase_date', 'total_orders',
            'total_spent', 'avg_order_value', 'std_order_value', 
            'min_order_value', 'max_order_value'
        ]
        
        # Calculate days since last purchase
        customer_features['days_since_last_purchase'] = (
            reference_date - customer_features['last_purchase_date']
        ).dt.days
        
        # Calculate customer lifespan in days
        customer_features['customer_lifespan_days'] = (
            customer_features['last_purchase_date'] - customer_features['first_purchase_date']
        ).dt.days
        
        # Calculate purchase frequency (orders per day)
        customer_features['purchase_frequency'] = (
            customer_features['total_orders'] / 
            np.maximum(customer_features['customer_lifespan_days'], 1)
        ).round(4)
        
        # Handle missing std values (single purchase customers)
        customer_features['std_order_value'] = customer_features['std_order_value'].fillna(0)
        
        # Additional behavioral features
        if 'product_category' in transaction_data.columns:
            # Number of unique categories purchased
            category_diversity = transaction_data.groupby('customer_id')['product_category'].nunique()
            customer_features['category_diversity'] = category_diversity
        
        # Seasonal behavior features
        seasonal_features = self._calculate_seasonal_features(transaction_data)
        customer_features = customer_features.join(seasonal_features, how='left')
        
        # Reset index to make customer_id a column
        customer_features = customer_features.reset_index()
        
        self.logger.info(f"Created features for {len(customer_features)} customers")
        return customer_features
    
    def _calculate_seasonal_features(self, transaction_data):
        # Month-wise purchase analysis
        monthly_purchases = transaction_data.groupby(['customer_id', 'month']).size().unstack(fill_value=0)
        
        # Calculate seasonal preferences
        seasonal_features = pd.DataFrame(index=monthly_purchases.index)
        
        # Preferred shopping month
        seasonal_features['preferred_month'] = monthly_purchases.idxmax(axis=1)
        
        # Purchase consistency (std of monthly purchases)
        seasonal_features['monthly_consistency'] = monthly_purchases.std(axis=1).round(2)
        
        # Quarterly spending pattern
        quarterly_data = transaction_data.copy()
        quarterly_spending = quarterly_data.groupby(['customer_id', 'quarter'])['order_value'].sum().unstack(fill_value=0)
        seasonal_features['preferred_quarter'] = quarterly_spending.idxmax(axis=1)
        
        return seasonal_features
    
    def detect_and_handle_outliers(self, data, columns, method='iqr', threshold=1.5):
        self.logger.info(f"Detecting outliers using {method} method...")
        
        outlier_info = {}
        cleaned_data = data.copy()
        
        for column in columns:
            if column not in data.columns:
                continue
                
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                outliers = z_scores > threshold
            
            outlier_count = outliers.sum()
            outlier_info[column] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(data)) * 100
            }
            
            # Cap outliers instead of removing them
            if method == 'iqr' and outlier_count > 0:
                cleaned_data.loc[cleaned_data[column] < lower_bound, column] = lower_bound
                cleaned_data.loc[cleaned_data[column] > upper_bound, column] = upper_bound
        
        self.logger.info(f"Outlier detection completed: {outlier_info}")
        return cleaned_data, outlier_info
    
    def create_sample_dataset(self, n_customers=1000, n_transactions_range=(1, 50)):
        self.logger.info("Creating sample dataset...")
        
        np.random.seed(42)
        
        # Generate customer IDs
        customer_ids = range(1, n_customers + 1)
        
        transactions = []
        
        for customer_id in customer_ids:
            # Random number of transactions per customer
            n_transactions = np.random.randint(n_transactions_range[0], n_transactions_range[1] + 1)
            
            # Generate transaction dates over past 2 years
            start_date = datetime.now() - timedelta(days=730)
            end_date = datetime.now() - timedelta(days=1)
            
            # Create random dates for this customer
            random_dates = pd.date_range(start_date, end_date, periods=n_transactions)
            random_dates = np.random.choice(random_dates, n_transactions, replace=False)
            random_dates = sorted(random_dates)
            
            # Generate order values (log-normal distribution)
            base_value = np.random.lognormal(3.5, 0.8)  # Customer's base spending
            order_values = np.random.lognormal(np.log(base_value), 0.5, n_transactions)
            order_values = np.round(order_values, 2)
            
            # Product categories
            categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Beauty', 'Food']
            
            for i in range(n_transactions):
                transactions.append({
                    'customer_id': customer_id,
                    'order_date': random_dates[i],
                    'order_value': order_values[i],
                    'product_category': np.random.choice(categories),
                    'quantity': np.random.randint(1, 6)
                })
        
        sample_data = pd.DataFrame(transactions)
        
        # Save sample data
        sample_file = self.processed_data_dir / "sample_transactions.csv"
        sample_data.to_csv(sample_file, index=False)
        
        self.logger.info(f"Sample dataset created with {len(sample_data)} transactions for {n_customers} customers")
        return sample_data
    
    def generate_data_quality_report(self, data):
        self.logger.info("Generating data quality report...")
        
        report = {
            'dataset_info': {
                'shape': data.shape,
                'memory_usage': data.memory_usage(deep=True).sum(),
                'dtypes': data.dtypes.to_dict()
            },
            'completeness': DataQualityMetrics.data_completeness_report(data),
            'consistency_issues': DataQualityMetrics.data_consistency_check(data)
        }
        
        # Outlier detection for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for column in numeric_columns:
            outlier_info[column] = DataQualityMetrics.detect_outliers_iqr(data, column)
        
        report['outliers'] = outlier_info
        
        return report
    
    def save_processed_data(self, data, filename):
        filepath = self.processed_data_dir / filename
        
        if filename.endswith('.csv'):
            data.to_csv(filepath, index=False)
        elif filename.endswith('.xlsx'):
            data.to_excel(filepath, index=False)
        else:
            # Default to CSV
            data.to_csv(filepath.with_suffix('.csv'), index=False)
        
        self.logger.info(f"Processed data saved to {filepath}")
        return filepath
