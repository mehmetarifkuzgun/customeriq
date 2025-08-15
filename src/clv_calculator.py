
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from lifelines import BetaGeoFitter, GammaGammaFitter
except ImportError:
    print("Warning: lifelines not installed. CLV models will use simplified calculations.")
    BetaGeoFitter = None
    GammaGammaFitter = None
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from config.settings import CLV_PREDICTION_PERIOD, RANDOM_STATE, MODELS_DIR
from utils.metrics import ModelMetrics

class CLVCalculator:
    def __init__(self, prediction_period_days=CLV_PREDICTION_PERIOD):
        self.prediction_period_days = prediction_period_days
        self.random_state = RANDOM_STATE
        self.models_dir = Path(MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Initialize models
        self.bgf_model = BetaGeoFitter()  # BG/NBD model for purchase prediction
        self.ggf_model = GammaGammaFitter()  # Gamma-Gamma model for monetary prediction
        self.ml_model = None
        self.scaler = StandardScaler()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_clv_data(self, transaction_data, customer_data=None):
        self.logger.info("Preparing data for CLV calculation...")
        
        # Ensure date column is datetime
        transaction_data['order_date'] = pd.to_datetime(transaction_data['order_date'])
        
        # Calculate observation period
        max_date = transaction_data['order_date'].max()
        min_date = transaction_data['order_date'].min()
        observation_period = (max_date - min_date).days
        
        # Calculate RFM values for CLV
        current_date = max_date
        
        # Group by customer and calculate CLV features
        clv_data = transaction_data.groupby('customer_id').agg({
            'order_date': ['min', 'max', 'count'],
            'order_value': ['sum', 'mean']
        })
        
        # Flatten column names
        clv_data.columns = ['first_purchase', 'last_purchase', 'frequency', 'total_monetary', 'avg_monetary']
        
        # Calculate recency (time between first and last purchase)
        clv_data['recency'] = (clv_data['last_purchase'] - clv_data['first_purchase']).dt.days
        
        # Calculate T (age of customer at end of observation period)
        clv_data['T'] = (current_date - clv_data['first_purchase']).dt.days
        
        # Adjust frequency (number of repeat purchases)
        clv_data['frequency'] = clv_data['frequency'] - 1
        clv_data['frequency'] = np.maximum(clv_data['frequency'], 0)
        
        # Filter customers with valid data
        valid_customers = (clv_data['T'] > 0) & (clv_data['avg_monetary'] > 0)
        clv_data = clv_data[valid_customers].copy()
        
        # Add customer features if available
        if customer_data is not None:
            clv_data = clv_data.merge(
                customer_data.set_index('customer_id'), 
                left_index=True, 
                right_index=True, 
                how='left'
            )
        
        # Reset index to make customer_id a column
        clv_data = clv_data.reset_index()
        
        self.logger.info(f"CLV data prepared for {len(clv_data)} customers")
        return clv_data
    
    def calculate_historical_clv(self, clv_data, period_days=365):
        self.logger.info("Calculating historical CLV...")
        
        historical_clv = clv_data.copy()
        
        # Simple CLV calculation: average order value * frequency * predicted lifetime
        # Estimate lifetime based on purchase pattern
        avg_time_between_purchases = np.where(
            historical_clv['frequency'] > 0,
            historical_clv['recency'] / historical_clv['frequency'],
            historical_clv['T']
        )
        
        # Estimate remaining lifetime
        estimated_lifetime = np.where(
            avg_time_between_purchases > 0,
            period_days / avg_time_between_purchases,
            1
        )
        
        # Calculate CLV
        historical_clv['historical_clv'] = (
            historical_clv['avg_monetary'] * 
            historical_clv['frequency'] * 
            estimated_lifetime
        ).round(2)
        
        # Add CLV segments
        historical_clv['clv_segment'] = pd.qcut(
            historical_clv['historical_clv'], 
            q=5, 
            labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        )
        
        return historical_clv
    
    def train_bgf_model(self, clv_data):

        self.logger.info("Training BG/NBD model for purchase prediction...")
        
        # Filter customers with valid data for BG/NBD
        valid_data = clv_data[
            (clv_data['frequency'] >= 0) & 
            (clv_data['recency'] >= 0) & 
            (clv_data['T'] > 0)
        ].copy()
        
        # Fit BG/NBD model
        self.bgf_model.fit(
            valid_data['frequency'], 
            valid_data['recency'], 
            valid_data['T'],
            verbose=True
        )
        
        self.logger.info("BG/NBD model training completed")
        return self.bgf_model
    
    def train_ggf_model(self, clv_data):

        self.logger.info("Training Gamma-Gamma model for monetary prediction...")
        
        # Filter customers with repeat purchases for Gamma-Gamma model
        repeat_customers = clv_data[clv_data['frequency'] > 0].copy()
        
        if len(repeat_customers) == 0:
            self.logger.warning("No repeat customers found for Gamma-Gamma model")
            return None
        
        # Fit Gamma-Gamma model
        self.ggf_model.fit(
            repeat_customers['frequency'],
            repeat_customers['avg_monetary'],
            verbose=True
        )
        
        self.logger.info("Gamma-Gamma model training completed")
        return self.ggf_model
    
    def train_ml_model(self, clv_data, target_column='historical_clv'):

        self.logger.info("Training ML model for CLV prediction...")
        
        # Prepare features
        feature_columns = [
            'frequency', 'recency', 'T', 'avg_monetary', 'total_monetary'
        ]
        
        # Add additional features if available
        additional_features = [
            'total_orders', 'customer_lifespan_days', 'purchase_frequency',
            'category_diversity', 'std_order_value'
        ]
        
        for feature in additional_features:
            if feature in clv_data.columns:
                feature_columns.append(feature)
        
        # Filter available features
        available_features = [col for col in feature_columns if col in clv_data.columns]
        
        X = clv_data[available_features].fillna(0)
        y = clv_data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.ml_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=self.random_state
        )
        
        self.ml_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.ml_model.predict(X_test_scaled)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.ml_model, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_absolute_error'
        )
        metrics['cv_mae'] = -cv_scores.mean()
        
        self.logger.info(f"ML model training completed. R²: {metrics['r2']:.3f}")
        return self.ml_model, metrics
    
    def predict_clv(self, clv_data, method='all', period_days=None):
        if period_days is None:
            period_days = self.prediction_period_days
        
        self.logger.info(f"Predicting CLV for {period_days} days using {method} method...")
        
        predictions = clv_data[['customer_id']].copy()
        
        # BG/NBD + Gamma-Gamma prediction
        if method in ['bgf', 'combined', 'all']:
            predictions['predicted_purchases'] = self.bgf_model.predict(
                period_days / 365,  # Convert to years for BG/NBD
                clv_data['frequency'],
                clv_data['recency'],
                clv_data['T']
            )
            
            # Predict monetary value for repeat customers
            if self.ggf_model is not None:
                repeat_customers = clv_data['frequency'] > 0
                
                predictions['predicted_avg_order_value'] = clv_data['avg_monetary'].copy()
                
                if repeat_customers.sum() > 0:
                    predicted_monetary = self.ggf_model.conditional_expected_average_profit(
                        clv_data[repeat_customers]['frequency'],
                        clv_data[repeat_customers]['avg_monetary']
                    )
                    predictions.loc[repeat_customers, 'predicted_avg_order_value'] = predicted_monetary
            else:
                predictions['predicted_avg_order_value'] = clv_data['avg_monetary']
            
            # Calculate CLV
            predictions['clv_bgf'] = (
                predictions['predicted_purchases'] * 
                predictions['predicted_avg_order_value']
            ).round(2)
        
        # Machine Learning prediction
        if method in ['ml', 'combined', 'all'] and self.ml_model is not None:
            feature_columns = [
                'frequency', 'recency', 'T', 'avg_monetary', 'total_monetary'
            ]
            
            # Add additional features if available
            additional_features = [
                'total_orders', 'customer_lifespan_days', 'purchase_frequency',
                'category_diversity', 'std_order_value'
            ]
            
            for feature in additional_features:
                if feature in clv_data.columns:
                    feature_columns.append(feature)
            
            available_features = [col for col in feature_columns if col in clv_data.columns]
            X = clv_data[available_features].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            predictions['clv_ml'] = self.ml_model.predict(X_scaled).round(2)
        
        # Combined prediction (ensemble)
        if method == 'combined':
            if 'clv_bgf' in predictions.columns and 'clv_ml' in predictions.columns:
                predictions['clv_combined'] = (
                    predictions['clv_bgf'] * 0.6 + 
                    predictions['clv_ml'] * 0.4
                ).round(2)
            elif 'clv_bgf' in predictions.columns:
                predictions['clv_combined'] = predictions['clv_bgf']
            elif 'clv_ml' in predictions.columns:
                predictions['clv_combined'] = predictions['clv_ml']
        
        # Add CLV segments
        if method == 'combined' and 'clv_combined' in predictions.columns:
            target_clv = 'clv_combined'
        elif 'clv_ml' in predictions.columns:
            target_clv = 'clv_ml'
        else:
            target_clv = 'clv_bgf'
        
        if target_clv in predictions.columns:
            predictions['clv_segment'] = pd.qcut(
                predictions[target_clv],
                q=5,
                labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
                duplicates='drop'
            )
        
        # Add prediction metadata
        predictions['prediction_date'] = datetime.now().date()
        predictions['prediction_period_days'] = period_days
        predictions['model_version'] = f'{method}_v1.0'
        
        self.logger.info("CLV prediction completed")
        return predictions
    
    def calculate_cohort_clv(self, transaction_data, cohort_period='M'):

        self.logger.info("Calculating cohort CLV analysis...")
        
        # Prepare transaction data
        transaction_data['order_date'] = pd.to_datetime(transaction_data['order_date'])
        
        # Get customer's first purchase date
        customer_cohorts = transaction_data.groupby('customer_id')['order_date'].min().reset_index()
        customer_cohorts.columns = ['customer_id', 'cohort_date']
        
        # Create cohort periods
        if cohort_period == 'M':
            customer_cohorts['cohort_period'] = customer_cohorts['cohort_date'].dt.to_period('M')
        else:
            customer_cohorts['cohort_period'] = customer_cohorts['cohort_date'].dt.to_period('Q')
        
        # Merge with transaction data
        cohort_data = transaction_data.merge(customer_cohorts, on='customer_id')
        
        # Calculate CLV by cohort
        cohort_clv = cohort_data.groupby(['cohort_period', 'customer_id']).agg({
            'order_value': 'sum',
            'order_date': 'count'
        }).reset_index()
        
        cohort_clv.columns = ['cohort_period', 'customer_id', 'total_clv', 'total_orders']
        
        # Aggregate by cohort
        cohort_summary = cohort_clv.groupby('cohort_period').agg({
            'customer_id': 'count',
            'total_clv': ['sum', 'mean', 'median'],
            'total_orders': 'mean'
        }).round(2)
        
        # Flatten column names
        cohort_summary.columns = [
            'customers', 'total_revenue', 'avg_clv', 'median_clv', 'avg_orders'
        ]
        
        return cohort_summary.reset_index()
    
    def identify_high_value_customers(self, clv_predictions, percentile=80):
        # Determine CLV column to use
        if 'clv_combined' in clv_predictions.columns:
            clv_column = 'clv_combined'
        elif 'clv_ml' in clv_predictions.columns:
            clv_column = 'clv_ml'
        else:
            clv_column = 'clv_bgf'
        
        # Calculate threshold
        threshold = np.percentile(clv_predictions[clv_column], percentile)
        
        # Filter high-value customers
        high_value_customers = clv_predictions[
            clv_predictions[clv_column] >= threshold
        ].copy()
        
        # Sort by CLV
        high_value_customers = high_value_customers.sort_values(
            clv_column, ascending=False
        )
        
        self.logger.info(f"Identified {len(high_value_customers)} high-value customers")
        return high_value_customers
    
    def calculate_clv_roi(self, clv_predictions, acquisition_cost=None, retention_cost=None):
        # Determine CLV column
        if 'clv_combined' in clv_predictions.columns:
            clv_column = 'clv_combined'
        elif 'clv_ml' in clv_predictions.columns:
            clv_column = 'clv_ml'
        else:
            clv_column = 'clv_bgf'
        
        total_clv = clv_predictions[clv_column].sum()
        avg_clv = clv_predictions[clv_column].mean()
        
        roi_metrics = {
            'total_predicted_clv': total_clv,
            'average_clv': avg_clv,
            'median_clv': clv_predictions[clv_column].median(),
            'clv_std': clv_predictions[clv_column].std()
        }
        
        if acquisition_cost:
            roi_metrics['clv_to_cac_ratio'] = avg_clv / acquisition_cost
            roi_metrics['payback_period_months'] = acquisition_cost / (avg_clv / 12)
        
        if retention_cost:
            roi_metrics['retention_roi'] = (avg_clv - retention_cost) / retention_cost
        
        return roi_metrics
    
    def save_clv_models(self, model_name='clv_models'):
        model_path = self.models_dir / f"{model_name}.pkl"
        
        model_data = {
            'bgf_model': self.bgf_model,
            'ggf_model': self.ggf_model,
            'ml_model': self.ml_model,
            'scaler': self.scaler,
            'prediction_period_days': self.prediction_period_days
        }
        
        joblib.dump(model_data, model_path)
        self.logger.info(f"CLV models saved to {model_path}")
        
        return model_path
    
    def load_clv_models(self, model_path):
        model_data = joblib.load(model_path)
        
        self.bgf_model = model_data['bgf_model']
        self.ggf_model = model_data['ggf_model']
        self.ml_model = model_data['ml_model']
        self.scaler = model_data['scaler']
        self.prediction_period_days = model_data.get('prediction_period_days', CLV_PREDICTION_PERIOD)
        
        self.logger.info(f"CLV models loaded from {model_path}")
