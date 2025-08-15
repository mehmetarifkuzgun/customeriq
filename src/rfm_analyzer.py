
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from config.settings import RFM_QUANTILES, CUSTOMER_SEGMENTS

class RFMAnalyzer:
    def __init__(self, quantiles=RFM_QUANTILES):
        self.quantiles = quantiles
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_rfm(self, customer_data, reference_date=None):
        self.logger.info("Starting RFM calculation...")
        
        if reference_date is None:
            if 'last_purchase_date' in customer_data.columns:
                reference_date = pd.to_datetime(customer_data['last_purchase_date']).max()
            else:
                reference_date = datetime.now()
        
        rfm_data = customer_data.copy()
        
        # Ensure date columns are datetime
        if 'last_purchase_date' in rfm_data.columns:
            rfm_data['last_purchase_date'] = pd.to_datetime(rfm_data['last_purchase_date'])
            rfm_data['Recency'] = (reference_date - rfm_data['last_purchase_date']).dt.days
        elif 'days_since_last_purchase' in rfm_data.columns:
            rfm_data['Recency'] = rfm_data['days_since_last_purchase']
        else:
            raise ValueError("No recency information found in data")
        
        # Map frequency and monetary fields
        if 'total_orders' in rfm_data.columns:
            rfm_data['Frequency'] = rfm_data['total_orders']
        elif 'order_count' in rfm_data.columns:
            rfm_data['Frequency'] = rfm_data['order_count']
        else:
            raise ValueError("No frequency information found in data")
        
        if 'total_spent' in rfm_data.columns:
            rfm_data['Monetary'] = rfm_data['total_spent']
        elif 'total_revenue' in rfm_data.columns:
            rfm_data['Monetary'] = rfm_data['total_revenue']
        else:
            raise ValueError("No monetary information found in data")
        
        # Calculate RFM scores using quantiles
        rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], q=self.quantiles, labels=range(self.quantiles, 0, -1))
        rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), q=self.quantiles, labels=range(1, self.quantiles + 1))
        rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'].rank(method='first'), q=self.quantiles, labels=range(1, self.quantiles + 1))
        
        # Convert to numeric
        rfm_data['R_Score'] = rfm_data['R_Score'].astype(int)
        rfm_data['F_Score'] = rfm_data['F_Score'].astype(int)
        rfm_data['M_Score'] = rfm_data['M_Score'].astype(int)
        
        # Create combined RFM score
        rfm_data['RFM_Score'] = (
            rfm_data['R_Score'].astype(str) + 
            rfm_data['F_Score'].astype(str) + 
            rfm_data['M_Score'].astype(str)
        )
        
        # Calculate overall RFM score (weighted average)
        rfm_data['RFM_Score_Numeric'] = (
            rfm_data['R_Score'] * 0.15 +  # 15% weight for recency
            rfm_data['F_Score'] * 0.28 +  # 28% weight for frequency
            rfm_data['M_Score'] * 0.57    # 57% weight for monetary
        ).round(2)
        
        self.logger.info(f"RFM calculation completed for {len(rfm_data)} customers")
        return rfm_data
    
    def segment_customers(self, rfm_data):
        self.logger.info("Starting customer segmentation...")
        
        segmented_data = rfm_data.copy()
        
        # Define segmentation rules based on RFM scores
        def segment_customers_logic(row):
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            # Champions: High R, F, M
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            
            # Loyal Customers: Medium-High R, High F, Medium-High M
            elif r >= 3 and f >= 4 and m >= 3:
                return 'Loyal Customers'
            
            # Potential Loyalists: High R, Medium F, Medium M
            elif r >= 4 and f >= 2 and m >= 2:
                return 'Potential Loyalists'
            
            # Cannot Lose Them: Low R, High F, High M
            elif r <= 2 and f >= 4 and m >= 4:
                return 'Cannot Lose Them'
            
            # At Risk: Low-Medium R, Medium-High F, Medium-High M
            elif r <= 3 and f >= 3 and m >= 3:
                return 'At Risk'
            
            # New Customers: High R, Low F, Low-Medium M
            elif r >= 4 and f <= 2 and m <= 3:
                return 'New Customers'
            
            # Hibernating: Low R, Low F, High M
            elif r <= 2 and f <= 2 and m >= 3:
                return 'Hibernating'
            
            # Lost: Low R, Low F, Low M
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Lost'
            
            # Default category for edge cases
            else:
                return 'Others'
        
        segmented_data['Segment'] = segmented_data.apply(segment_customers_logic, axis=1)
        
        # Add segment descriptions
        segmented_data['Segment_Description'] = segmented_data['Segment'].map(
            lambda x: CUSTOMER_SEGMENTS.get(x, {}).get('description', 'Other customers')
        )
        
        # Calculate segment statistics
        segment_stats = self._calculate_segment_statistics(segmented_data)
        
        self.logger.info("Customer segmentation completed")
        self.logger.info(f"Segment distribution:\n{segmented_data['Segment'].value_counts()}")
        
        return segmented_data, segment_stats
    
    def _calculate_segment_statistics(self, segmented_data):
        segment_stats = segmented_data.groupby('Segment').agg({
            'customer_id': 'count',
            'Recency': ['mean', 'median'],
            'Frequency': ['mean', 'median'],
            'Monetary': ['mean', 'median', 'sum'],
            'RFM_Score_Numeric': ['mean', 'median']
        }).round(2)
        
        # Flatten column names
        segment_stats.columns = [
            'customer_count', 'avg_recency', 'median_recency',
            'avg_frequency', 'median_frequency',
            'avg_monetary', 'median_monetary', 'total_monetary',
            'avg_rfm_score', 'median_rfm_score'
        ]
        
        # Calculate percentages
        total_customers = segment_stats['customer_count'].sum()
        segment_stats['customer_percentage'] = (
            segment_stats['customer_count'] / total_customers * 100
        ).round(1)
        
        # Calculate revenue percentage
        total_revenue = segment_stats['total_monetary'].sum()
        segment_stats['revenue_percentage'] = (
            segment_stats['total_monetary'] / total_revenue * 100
        ).round(1)
        
        return segment_stats.reset_index()
    
    def identify_high_value_customers(self, rfm_data, top_percentage=0.2):
        # Calculate threshold for top customers
        threshold = rfm_data['RFM_Score_Numeric'].quantile(1 - top_percentage)
        
        high_value_customers = rfm_data[
            rfm_data['RFM_Score_Numeric'] >= threshold
        ].copy()
        
        # Add value tier
        high_value_customers['Value_Tier'] = 'High Value'
        
        # Sort by RFM score
        high_value_customers = high_value_customers.sort_values(
            'RFM_Score_Numeric', ascending=False
        )
        
        self.logger.info(f"Identified {len(high_value_customers)} high-value customers")
        return high_value_customers
    
    def recommend_actions(self, segmented_data):
        action_recommendations = {
            'Champions': {
                'strategy': 'Reward and Retain',
                'actions': [
                    'Offer exclusive products and early access',
                    'Create VIP loyalty program',
                    'Ask for referrals and reviews',
                    'Cross-sell premium products'
                ],
                'priority': 'High'
            },
            'Loyal Customers': {
                'strategy': 'Nurture and Upsell',
                'actions': [
                    'Offer loyalty rewards',
                    'Recommend complementary products',
                    'Provide excellent customer service',
                    'Send personalized offers'
                ],
                'priority': 'High'
            },
            'Potential Loyalists': {
                'strategy': 'Develop Relationship',
                'actions': [
                    'Offer membership programs',
                    'Increase purchase frequency with promotions',
                    'Provide product education',
                    'Send targeted campaigns'
                ],
                'priority': 'Medium'
            },
            'Cannot Lose Them': {
                'strategy': 'Win Back',
                'actions': [
                    'Send win-back campaigns',
                    'Offer significant discounts',
                    'Provide personal account manager',
                    'Survey for feedback and issues'
                ],
                'priority': 'Very High'
            },
            'At Risk': {
                'strategy': 'Retain',
                'actions': [
                    'Send limited-time offers',
                    'Recommend popular products',
                    'Provide customer support',
                    'Create re-engagement campaigns'
                ],
                'priority': 'High'
            },
            'New Customers': {
                'strategy': 'Onboard and Engage',
                'actions': [
                    'Provide onboarding support',
                    'Offer welcome series',
                    'Encourage second purchase',
                    'Share product tutorials'
                ],
                'priority': 'Medium'
            },
            'Hibernating': {
                'strategy': 'Reactivate',
                'actions': [
                    'Send reactivation campaigns',
                    'Offer comeback discounts',
                    'Share new product launches',
                    'Create nostalgia campaigns'
                ],
                'priority': 'Medium'
            },
            'Lost': {
                'strategy': 'Win Back or Let Go',
                'actions': [
                    'Send final win-back offer',
                    'Survey for feedback',
                    'Minimal marketing spend',
                    'Consider removing from active campaigns'
                ],
                'priority': 'Low'
            }
        }
        
        # Add recommendations to segmented data
        segmented_data['Recommended_Strategy'] = segmented_data['Segment'].map(
            lambda x: action_recommendations.get(x, {}).get('strategy', 'Monitor')
        )
        
        segmented_data['Priority'] = segmented_data['Segment'].map(
            lambda x: action_recommendations.get(x, {}).get('priority', 'Low')
        )
        
        return segmented_data, action_recommendations
    
    def calculate_segment_migration(self, current_rfm, previous_rfm):
        # Merge current and previous data
        migration_data = current_rfm[['customer_id', 'Segment']].merge(
            previous_rfm[['customer_id', 'Segment']], 
            on='customer_id', 
            suffixes=('_current', '_previous'),
            how='inner'
        )
        
        # Create migration matrix
        migration_matrix = pd.crosstab(
            migration_data['Segment_previous'],
            migration_data['Segment_current'],
            margins=True
        )
        
        # Calculate migration percentages
        migration_percentages = pd.crosstab(
            migration_data['Segment_previous'],
            migration_data['Segment_current'],
            normalize='index'
        ) * 100
        
        # Identify customers who changed segments
        segment_changes = migration_data[
            migration_data['Segment_current'] != migration_data['Segment_previous']
        ]
        
        return {
            'migration_matrix': migration_matrix,
            'migration_percentages': migration_percentages,
            'segment_changes': segment_changes
        }
    
    def export_rfm_analysis(self, rfm_data, filename='rfm_analysis.csv'):
        export_columns = [
            'customer_id', 'Recency', 'Frequency', 'Monetary',
            'R_Score', 'F_Score', 'M_Score', 'RFM_Score',
            'RFM_Score_Numeric', 'Segment', 'Segment_Description'
        ]
        
        # Filter available columns
        available_columns = [col for col in export_columns if col in rfm_data.columns]
        export_data = rfm_data[available_columns]
        
        # Save to file
        export_data.to_csv(filename, index=False)
        
        self.logger.info(f"RFM analysis exported to {filename}")
        return filename
