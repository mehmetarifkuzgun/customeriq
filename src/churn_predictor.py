import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from sklearn.pipeline import Pipeline
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from config.settings import RANDOM_STATE, TEST_SIZE, CV_FOLDS, CHURN_THRESHOLD_DAYS, MODELS_DIR, RISK_CATEGORIES
from utils.metrics import ModelMetrics

class ChurnPredictor:
    def __init__(self, churn_threshold_days=CHURN_THRESHOLD_DAYS):
        self.churn_threshold_days = churn_threshold_days
        self.random_state = RANDOM_STATE
        self.models_dir = Path(MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Initialize models
        self.rf_model = None
        self.xgb_model = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_churn_labels(self, customer_data, reference_date=None):

        self.logger.info("Creating churn labels...")
        
        if reference_date is None:
            reference_date = datetime.now()
        
        data_with_labels = customer_data.copy()
        
        # Calculate days since last purchase if not available
        if 'days_since_last_purchase' not in data_with_labels.columns:
            if 'last_purchase_date' in data_with_labels.columns:
                data_with_labels['last_purchase_date'] = pd.to_datetime(data_with_labels['last_purchase_date'])
                data_with_labels['days_since_last_purchase'] = (
                    reference_date - data_with_labels['last_purchase_date']
                ).dt.days
            else:
                raise ValueError("No information available to calculate churn labels")
        
        # Create churn label (1 = churned, 0 = active)
        data_with_labels['is_churned'] = (
            data_with_labels['days_since_last_purchase'] > self.churn_threshold_days
        ).astype(int)
        
        churn_rate = data_with_labels['is_churned'].mean()
        self.logger.info(f"Churn rate: {churn_rate:.2%} (threshold: {self.churn_threshold_days} days)")
        
        return data_with_labels
    
    def engineer_features(self, customer_data, transaction_data=None):
        self.logger.info("Engineering features for churn prediction...")
        
        features_data = customer_data.copy()
        
        # Basic RFM features
        if 'days_since_last_purchase' in features_data.columns:
            features_data['recency'] = features_data['days_since_last_purchase']
        
        if 'total_orders' in features_data.columns:
            features_data['frequency'] = features_data['total_orders']
        
        if 'total_spent' in features_data.columns:
            features_data['monetary'] = features_data['total_spent']
        
        # Derived features
        if 'avg_order_value' in features_data.columns and 'total_orders' in features_data.columns:
            features_data['spending_consistency'] = (
                features_data['total_spent'] / 
                (features_data['avg_order_value'] * features_data['total_orders'])
            ).fillna(1)
        
        # Customer lifespan features
        if 'customer_lifespan_days' in features_data.columns:
            features_data['purchase_rate'] = (
                features_data['total_orders'] / 
                np.maximum(features_data['customer_lifespan_days'], 1)
            ) * 30  # Orders per month
        
        # Engagement features
        if 'category_diversity' in features_data.columns:
            features_data['category_diversity_score'] = features_data['category_diversity']
        
        # Seasonal features
        if 'preferred_month' in features_data.columns:
            features_data['preferred_month_encoded'] = features_data['preferred_month']
        
        if 'monthly_consistency' in features_data.columns:
            features_data['purchase_consistency'] = features_data['monthly_consistency']
        
        # Transaction-based features (if transaction data is provided)
        if transaction_data is not None:
            additional_features = self._create_transaction_features(features_data, transaction_data)
            features_data = features_data.merge(additional_features, on='customer_id', how='left')
        
        # Handle missing values
        numeric_columns = features_data.select_dtypes(include=[np.number]).columns
        features_data[numeric_columns] = features_data[numeric_columns].fillna(0)
        
        # Encode categorical variables
        categorical_columns = features_data.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col not in ['customer_id']]
        
        for col in categorical_columns:
            if features_data[col].dtype == 'object':
                le = LabelEncoder()
                features_data[f'{col}_encoded'] = le.fit_transform(features_data[col].astype(str))
        
        self.logger.info(f"Feature engineering completed. Shape: {features_data.shape}")
        return features_data
    
    def _create_transaction_features(self, customer_data, transaction_data):
        # Calculate time-based patterns
        transaction_data['order_date'] = pd.to_datetime(transaction_data['order_date'])
        transaction_data['days_between_orders'] = transaction_data.groupby('customer_id')['order_date'].diff().dt.days
        
        # Aggregate transaction patterns
        transaction_features = transaction_data.groupby('customer_id').agg({
            'days_between_orders': ['mean', 'std'],
            'order_value': ['std', 'min', 'max'],
            'quantity': ['mean', 'sum'] if 'quantity' in transaction_data.columns else ['mean'] * 2
        })
        
        # Flatten column names
        transaction_features.columns = [
            'avg_days_between_orders', 'std_days_between_orders',
            'std_order_value', 'min_order_value', 'max_order_value',
            'avg_quantity', 'total_quantity'
        ]
        
        # Fill missing values
        transaction_features = transaction_features.fillna(0)
        
        # Calculate recent activity (last 30, 60, 90 days)
        recent_activity = self._calculate_recent_activity(transaction_data)
        transaction_features = transaction_features.join(recent_activity, how='left')
        
        return transaction_features.reset_index()
    
    def _calculate_recent_activity(self, transaction_data):
        reference_date = transaction_data['order_date'].max()
        
        activity_features = pd.DataFrame(index=transaction_data['customer_id'].unique())
        
        for days in [30, 60, 90]:
            cutoff_date = reference_date - timedelta(days=days)
            recent_data = transaction_data[transaction_data['order_date'] >= cutoff_date]
            
            recent_activity = recent_data.groupby('customer_id').agg({
                'order_value': ['count', 'sum', 'mean']
            })
            
            recent_activity.columns = [
                f'orders_last_{days}d', f'spent_last_{days}d', f'avg_order_last_{days}d'
            ]
            
            activity_features = activity_features.join(recent_activity, how='left')
        
        return activity_features.fillna(0)
    
    def prepare_training_data(self, features_data):
        self.logger.info("Preparing training data...")
        
        # Select feature columns (exclude IDs and target)
        exclude_columns = [
            'customer_id', 'is_churned', 'last_purchase_date', 'first_purchase_date',
            'Segment', 'Segment_Description'
        ]
        
        feature_columns = [col for col in features_data.columns if col not in exclude_columns]
        feature_columns = [col for col in feature_columns if features_data[col].dtype in ['int64', 'float64']]
        
        X = features_data[feature_columns]
        y = features_data['is_churned']
        
        # Store feature names
        self.feature_names = feature_columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"Training data prepared. Features: {len(feature_columns)}")
        self.logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns
    
    def train_models(self, X_train, y_train, X_test, y_test):
        self.logger.info("Training churn prediction models...")
        
        models_performance = {}
        
        # Random Forest Model
        self.logger.info("Training Random Forest model...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state),
            rf_params,
            cv=CV_FOLDS,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        rf_grid.fit(X_train, y_train)
        self.rf_model = rf_grid.best_estimator_
        
        # Evaluate Random Forest
        rf_pred = self.rf_model.predict(X_test)
        rf_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
        
        models_performance['Random Forest'] = ModelMetrics.calculate_churn_metrics(
            y_test, rf_pred, rf_pred_proba
        )
        
        # XGBoost Model
        self.logger.info("Training XGBoost model...")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=self.random_state),
            xgb_params,
            cv=CV_FOLDS,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        xgb_grid.fit(X_train, y_train)
        self.xgb_model = xgb_grid.best_estimator_
        
        # Evaluate XGBoost
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        
        models_performance['XGBoost'] = ModelMetrics.calculate_churn_metrics(
            y_test, xgb_pred, xgb_pred_proba
        )
        
        # Create ensemble model (weighted average)
        ensemble_pred_proba = (rf_pred_proba * 0.5) + (xgb_pred_proba * 0.5)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        models_performance['Ensemble'] = ModelMetrics.calculate_churn_metrics(
            y_test, ensemble_pred, ensemble_pred_proba
        )
        
        # Select best model based on AUC-ROC
        best_model_name = max(models_performance.keys(), 
                            key=lambda x: models_performance[x]['auc_roc'])
        
        self.logger.info(f"Best model: {best_model_name}")
        self.logger.info("Model training completed")
        
        return models_performance
    
    def predict_churn(self, customer_data, model_type='ensemble'):
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Models not trained. Please train models first.")
        
        # Prepare features
        feature_data = customer_data[self.feature_names].fillna(0)
        feature_data_scaled = self.scaler.transform(feature_data)
        
        # Make predictions
        if model_type == 'rf':
            churn_proba = self.rf_model.predict_proba(feature_data_scaled)[:, 1]
        elif model_type == 'xgb':
            churn_proba = self.xgb_model.predict_proba(feature_data_scaled)[:, 1]
        else:  # ensemble
            rf_proba = self.rf_model.predict_proba(feature_data_scaled)[:, 1]
            xgb_proba = self.xgb_model.predict_proba(feature_data_scaled)[:, 1]
            churn_proba = (rf_proba * 0.5) + (xgb_proba * 0.5)
        
        # Create results DataFrame
        results = customer_data[['customer_id']].copy()
        results['churn_probability'] = churn_proba
        results['churn_prediction'] = (churn_proba > 0.5).astype(int)
        
        # Add risk categories
        results['risk_category'] = results['churn_probability'].apply(self._categorize_risk)
        
        # Add prediction date and model version
        results['prediction_date'] = datetime.now().date()
        results['model_version'] = f'{model_type}_v1.0'
        
        return results
    
    def _categorize_risk(self, probability):
        for category, config in RISK_CATEGORIES.items():
            if probability <= config['threshold']:
                return category
        return 'High'  # Default for probabilities > 1.0
    
    def get_feature_importance(self, model_type='rf', top_n=20):
        if model_type == 'rf' and self.rf_model is not None:
            importance = self.rf_model.feature_importances_
        elif model_type == 'xgb' and self.xgb_model is not None:
            importance = self.xgb_model.feature_importances_
        else:
            raise ValueError("Model not available or not trained")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)
    
    def save_models(self, model_name='churn_model'):
        """Save trained models to disk"""
        model_path = self.models_dir / f"{model_name}.pkl"
        
        model_data = {
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'churn_threshold_days': self.churn_threshold_days
        }
        
        joblib.dump(model_data, model_path)
        self.logger.info(f"Models saved to {model_path}")
        
        return model_path
    
    def load_models(self, model_path):
        """Load trained models from disk"""
        model_data = joblib.load(model_path)
        
        self.rf_model = model_data['rf_model']
        self.xgb_model = model_data['xgb_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.churn_threshold_days = model_data.get('churn_threshold_days', CHURN_THRESHOLD_DAYS)
        
        self.logger.info(f"Models loaded from {model_path}")
    
    def analyze_churn_factors(self, customer_data, high_risk_threshold=0.7):
        if 'churn_probability' not in customer_data.columns:
            raise ValueError("Churn predictions not found. Please run predictions first.")
        
        high_risk_customers = customer_data[
            customer_data['churn_probability'] >= high_risk_threshold
        ]
        
        low_risk_customers = customer_data[
            customer_data['churn_probability'] < 0.3
        ]
        
        analysis = {
            'high_risk_count': len(high_risk_customers),
            'high_risk_percentage': len(high_risk_customers) / len(customer_data) * 100,
            'avg_churn_probability': customer_data['churn_probability'].mean(),
            'feature_comparison': {}
        }
        
        numeric_features = [col for col in self.feature_names if col in customer_data.columns]
        
        for feature in numeric_features:
            if customer_data[feature].dtype in ['int64', 'float64']:
                analysis['feature_comparison'][feature] = {
                    'high_risk_avg': high_risk_customers[feature].mean(),
                    'low_risk_avg': low_risk_customers[feature].mean(),
                    'difference': high_risk_customers[feature].mean() - low_risk_customers[feature].mean()
                }
        
        return analysis
