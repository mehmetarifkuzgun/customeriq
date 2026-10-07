
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ModelMetrics:
    
    @staticmethod
    def calculate_churn_metrics(y_true, y_pred, y_pred_proba=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def customer_segment_quality(rfm_data, segments):
        segment_stats = {}
        
        for segment in segments.unique():
            segment_data = rfm_data[segments == segment]
            
            segment_stats[segment] = {
                'count': len(segment_data),
                'avg_recency': segment_data['Recency'].mean(),
                'avg_frequency': segment_data['Frequency'].mean(),
                'avg_monetary': segment_data['Monetary'].mean(),
                'total_value': segment_data['Monetary'].sum()
            }
        
        return segment_stats
    
    @staticmethod
    def clv_accuracy_metrics(actual_clv, predicted_clv):
        mape = np.mean(np.abs((actual_clv - predicted_clv) / actual_clv)) * 100
        rmse = np.sqrt(np.mean((actual_clv - predicted_clv) ** 2))
        mae = np.mean(np.abs(actual_clv - predicted_clv))
        
        correlation = np.corrcoef(actual_clv, predicted_clv)[0, 1]
        
        return {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation
        }

class BusinessMetrics:
    
    @staticmethod
    def _last_purchase(customer_data):
        """Last-purchase column: 'last_purchase_date' (DataProcessor) or legacy 'last_purchase'."""
        for col in ('last_purchase_date', 'last_purchase'):
            if col in customer_data.columns:
                return customer_data[col]
        raise KeyError("customer_data needs a 'last_purchase_date' column")

    @staticmethod
    def calculate_retention_rate(customer_data, period_days=90):
        last = pd.to_datetime(BusinessMetrics._last_purchase(customer_data))
        cutoff_date = last.max() - pd.Timedelta(days=period_days)
        
        active_customers = customer_data[last >= cutoff_date]
        retention_rate = len(active_customers) / len(customer_data)
        
        return retention_rate
    
    @staticmethod
    def calculate_churn_rate(customer_data, churn_threshold_days=90):
        last = pd.to_datetime(BusinessMetrics._last_purchase(customer_data))
        # measured against the newest order in the data (not wall-clock "now"),
        # consistent with the days_since_last_purchase column
        days_since_last = (last.max() - last).dt.days
        
        churned_customers = (days_since_last > churn_threshold_days).sum()
        churn_rate = churned_customers / len(customer_data)
        
        return churn_rate
    
    @staticmethod
    def customer_acquisition_cost(marketing_spend, new_customers):
        if new_customers == 0:
            return 0
        return marketing_spend / new_customers
    
    @staticmethod
    def calculate_segment_performance(customer_data):
        if 'Segment' not in customer_data.columns:
            return {}
        
        performance = customer_data.groupby('Segment').agg({
            'customer_id': 'count',
            'total_spent': ['sum', 'mean'],
            'order_count': 'mean',
            'days_since_last_purchase': 'mean'
        }).round(2)
        
        performance.columns = ['customer_count', 'total_revenue', 'avg_revenue_per_customer', 
                              'avg_orders', 'avg_days_since_last']
        
        return performance.to_dict('index')
    
    @staticmethod
    def roi_retention_campaign(campaign_cost, prevented_churn_customers, avg_clv):
        revenue_saved = prevented_churn_customers * avg_clv
        roi = (revenue_saved - campaign_cost) / campaign_cost * 100
        
        return {
            'campaign_cost': campaign_cost,
            'revenue_saved': revenue_saved,
            'roi_percentage': roi,
            'customers_retained': prevented_churn_customers
        }

class DataQualityMetrics:
    
    @staticmethod
    def data_completeness_report(df):
        completeness = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            completeness[column] = {
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(df)) * 100,
                'complete_percentage': ((len(df) - missing_count) / len(df)) * 100
            }
        
        return completeness
    
    @staticmethod
    def detect_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    @staticmethod
    def data_consistency_check(df):
        issues = []
        
        # Check for negative values in monetary columns
        monetary_columns = ['order_value', 'total_spent', 'monetary']
        for col in monetary_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"Found {negative_count} negative values in {col}")
        
        # Check for future dates
        date_columns = ['purchase_date', 'last_purchase']
        for col in date_columns:
            if col in df.columns:
                try:
                    future_dates = (pd.to_datetime(df[col]) > pd.Timestamp.now()).sum()
                    if future_dates > 0:
                        issues.append(f"Found {future_dates} future dates in {col}")
                except:
                    issues.append(f"Date format issues in {col}")
        
        # Check for duplicate customer IDs
        if 'customer_id' in df.columns:
            duplicates = df['customer_id'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate customer IDs")
        
        return issues

def generate_model_report(model, X_test, y_test, feature_names=None):
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None
    
    # Calculate metrics
    metrics = ModelMetrics.calculate_churn_metrics(y_test, y_pred, y_pred_proba)
    
    # Feature importance
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        if feature_names is not None:
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
        else:
            feature_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(feature_importance))],
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    report = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'feature_importance': feature_df if feature_importance is not None else None,
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return report
