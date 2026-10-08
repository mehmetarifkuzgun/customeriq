import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config.settings import COLOR_PALETTE, PLOTLY_THEME

class CustomerVisualizer:
    def __init__(self):
        self.color_palette = COLOR_PALETTE
        self.theme = PLOTLY_THEME
        
    def plot_rfm_distribution(self, rfm_data):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Recency Distribution', 'Frequency Distribution', 
                          'Monetary Distribution', 'RFM Score Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Recency
        fig.add_trace(
            go.Histogram(x=rfm_data['Recency'], name='Recency', nbinsx=20),
            row=1, col=1
        )
        
        # Frequency
        fig.add_trace(
            go.Histogram(x=rfm_data['Frequency'], name='Frequency', nbinsx=20),
            row=1, col=2
        )
        
        # Monetary
        fig.add_trace(
            go.Histogram(x=rfm_data['Monetary'], name='Monetary', nbinsx=20),
            row=2, col=1
        )
        
        # RFM Score
        if 'RFM_Score' in rfm_data.columns:
            fig.add_trace(
                go.Histogram(x=rfm_data['RFM_Score'], name='RFM Score', nbinsx=15),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="RFM Analysis - Distribution Overview"
        )
        
        return fig
    
    def plot_customer_segments(self, segment_data):
        segment_counts = segment_data['Segment'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=segment_counts.index,
                y=segment_counts.values,
                marker_color=self.color_palette[:len(segment_counts)]
            )
        ])
        
        fig.update_layout(
            title="Customer Segment Distribution",
            xaxis_title="Customer Segments",
            yaxis_title="Number of Customers",
            template=self.theme
        )
        
        return fig
    
    def plot_churn_risk_heatmap(self, customer_data):
        # The database returns snake_case columns (segment, risk_category, churn_probability)
        customer_data = customer_data.rename(columns=str.lower)
        if not {'churn_probability', 'segment', 'risk_category'} <= set(customer_data.columns):
            return None
            
        heatmap_data = customer_data.groupby(['segment', 'risk_category']).size().unstack(fill_value=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn_r',
            showscale=True
        ))
        
        fig.update_layout(
            title="Churn Risk Heatmap by Customer Segment",
            xaxis_title="Risk Category",
            yaxis_title="Customer Segment",
            template=self.theme
        )
        
        return fig
    
    def plot_clv_analysis(self, clv_data):
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('CLV Distribution', 'CLV by Segment')
        )
        
        # Pick whichever CLV estimate the model produced
        clv_col = next((c for c in ('clv_combined', 'clv_ml', 'clv_bgf', 'predicted_clv', 'CLV')
                        if c in clv_data.columns), None)
        if clv_col is None:
            raise KeyError("clv_data has no CLV column (expected clv_combined / clv_ml / clv_bgf)")
        seg_col = next((c for c in ('clv_segment', 'Segment') if c in clv_data.columns), None)

        # CLV Distribution
        fig.add_trace(
            go.Histogram(x=clv_data[clv_col], name='CLV Distribution', nbinsx=30),
            row=1, col=1
        )
        
        # CLV by Segment
        if seg_col:
            segment_clv = clv_data.groupby(seg_col, observed=True)[clv_col].mean().sort_values(ascending=True)
            fig.add_trace(
                go.Bar(
                    x=segment_clv.values,
                    y=segment_clv.index,
                    orientation='h',
                    name='Avg CLV by Segment'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            title_text="Customer Lifetime Value Analysis",
            template=self.theme
        )
        
        return fig
    
    def plot_cohort_analysis(self, cohort_data):
        fig = go.Figure(data=go.Heatmap(
            z=cohort_data.values,
            x=cohort_data.columns,
            y=cohort_data.index,
            colorscale='Blues',
            showscale=True,
            text=np.around(cohort_data.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Customer Cohort Retention Analysis",
            xaxis_title="Period",
            yaxis_title="Cohort",
            template=self.theme
        )
        
        return fig
    
    def plot_feature_importance(self, feature_names, importance_scores, title="Feature Importance"):
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)[::-1][:20]  # Top 20 features
        
        fig = go.Figure(data=[
            go.Bar(
                x=importance_scores[sorted_idx],
                y=[feature_names[i] for i in sorted_idx],
                orientation='h',
                marker_color=self.color_palette[0]
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            template=self.theme,
            height=600
        )
        
        return fig
    
    def plot_customer_timeline(self, customer_data, customer_id):
        customer_history = customer_data[customer_data['customer_id'] == customer_id].copy()
        customer_history['purchase_date'] = pd.to_datetime(customer_history['purchase_date'])
        customer_history = customer_history.sort_values('purchase_date')
        
        fig = go.Figure()
        
        # Purchase timeline
        fig.add_trace(go.Scatter(
            x=customer_history['purchase_date'],
            y=customer_history['order_value'],
            mode='lines+markers',
            name='Purchase Value',
            line=dict(color=self.color_palette[0]),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Customer {customer_id} - Purchase Timeline",
            xaxis_title="Date",
            yaxis_title="Order Value",
            template=self.theme
        )
        
        return fig
    
    def create_dashboard_summary(self, summary_stats):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Customers', 'Average CLV', 'Churn Rate', 'Revenue'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Total Customers
        fig.add_trace(go.Indicator(
            mode="number",
            value=summary_stats.get('total_customers', 0),
            title={"text": "Total Customers"},
            number={'font': {'size': 40}}
        ), row=1, col=1)
        
        # Average CLV
        fig.add_trace(go.Indicator(
            mode="number",
            value=summary_stats.get('avg_clv', 0),
            title={"text": "Average CLV ($)"},
            number={'font': {'size': 40}, 'prefix': '$'}
        ), row=1, col=2)
        
        # Churn Rate
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=summary_stats.get('churn_rate', 0) * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Rate (%)"},
            gauge={'axis': {'range': [None, 100]},
                  'bar': {'color': "red"},
                  'steps': [
                      {'range': [0, 25], 'color': "lightgray"},
                      {'range': [25, 50], 'color': "gray"}],
                  'threshold': {'line': {'color': "red", 'width': 4},
                              'thickness': 0.75, 'value': 75}}
        ), row=2, col=1)
        
        # Revenue
        fig.add_trace(go.Indicator(
            mode="number",
            value=summary_stats.get('total_revenue', 0),
            title={"text": "Total Revenue ($)"},
            number={'font': {'size': 40}, 'prefix': '$'}
        ), row=2, col=2)
        
        fig.update_layout(height=400, template=self.theme)
        
        return fig
