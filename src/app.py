import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from data_processor import DataProcessor
from rfm_analyzer import RFMAnalyzer
from churn_predictor import ChurnPredictor
from clv_calculator import CLVCalculator
from database import DatabaseManager
from utils.visualization import CustomerVisualizer
from utils.metrics import BusinessMetrics
from config.settings import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, COLOR_PALETTE,
    CUSTOMER_SEGMENTS, RISK_CATEGORIES
)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

class CustomerIQApp:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.rfm_analyzer = RFMAnalyzer()
        self.churn_predictor = ChurnPredictor()
        self.clv_calculator = CLVCalculator()
        self.db_manager = DatabaseManager()
        self.visualizer = CustomerVisualizer()
        
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
    
    def main(self):
        """Main application entry point"""
        st.title("🎯 CustomerIQ - Intelligent Customer Analytics")
        st.markdown("### Transform your customer data into actionable insights")
        
        page = st.sidebar.selectbox(
            "Navigate to:",
            [
                "📊 Dashboard Overview",
                "📁 Data Upload & Processing",
                "🔍 RFM Analysis",
                "⚠️ Churn Prediction",
                "💰 Customer Lifetime Value",
                "👥 Customer Segmentation",
                "📈 Business Insights",
                "⚙️ Settings"
            ]
        )
        
        if page == "📊 Dashboard Overview":
            self.dashboard_overview()
        elif page == "📁 Data Upload & Processing":
            self.data_upload_page()
        elif page == "🔍 RFM Analysis":
            self.rfm_analysis_page()
        elif page == "⚠️ Churn Prediction":
            self.churn_prediction_page()
        elif page == "💰 Customer Lifetime Value":
            self.clv_analysis_page()
        elif page == "👥 Customer Segmentation":
            self.customer_segmentation_page()
        elif page == "📈 Business Insights":
            self.business_insights_page()
        elif page == "⚙️ Settings":
            self.settings_page()
    
    def dashboard_overview(self):
        """Main dashboard with key metrics and visualizations"""
        st.header("📊 Dashboard Overview")
        
        if not st.session_state.data_loaded:
            st.warning("Please upload and process your data first!")
            if st.button("Go to Data Upload"):
                st.experimental_rerun()
            return
        
        try:
            customer_data = self.db_manager.get_customer_data()
            
            if customer_data.empty:
                st.warning("No customer data found. Please process your data first.")
                return
            
            summary_stats = self._calculate_summary_stats(customer_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", f"{summary_stats['total_customers']:,}")
            
            with col2:
                st.metric("Total Revenue", f"${summary_stats['total_revenue']:,.0f}")
            
            with col3:
                st.metric("Average CLV", f"${summary_stats['avg_clv']:.0f}")
            
            with col4:
                churn_rate = summary_stats.get('churn_rate', 0)
                st.metric("Churn Rate", f"{churn_rate:.1%}")
            
            fig_summary = self.visualizer.create_dashboard_summary(summary_stats)
            st.plotly_chart(fig_summary, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Customer Segment Distribution")
                if 'segment' in customer_data.columns:
                    segment_counts = customer_data['segment'].value_counts()
                    fig_segments = px.pie(
                        values=segment_counts.values,
                        names=segment_counts.index,
                        title="Customers by Segment"
                    )
                    st.plotly_chart(fig_segments, use_container_width=True)
            
            with col2:
                st.subheader("Risk Distribution")
                if 'risk_category' in customer_data.columns:
                    risk_counts = customer_data['risk_category'].value_counts()
                    colors = [RISK_CATEGORIES.get(risk, {}).get('color', '#1f77b4') for risk in risk_counts.index]
                    risk_df = risk_counts.rename_axis('risk').reset_index(name='customers')
                    fig_risk = px.bar(
                        risk_df,
                        x='risk',
                        y='customers',
                        title="Customers by Risk Level",
                        color='risk',
                        color_discrete_sequence=colors
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
            
            st.subheader("📈 Recent Trends")
            self._display_recent_trends(customer_data, self.db_manager.get_transaction_data())
            
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")
    
    def data_upload_page(self):
        """Data upload and processing page"""
        st.header("📁 Data Upload & Processing")
        
        st.subheader("Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your customer transaction data"
        )
        
        if st.button("Use Sample Dataset"):
            with st.spinner("Creating sample dataset..."):
                sample_data = self.data_processor.create_sample_dataset(n_customers=1000)
                st.session_state.raw_data = sample_data
                st.success("Sample dataset created successfully!")
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading data..."):
                    if uploaded_file.name.endswith('.csv'):
                        raw_data = pd.read_csv(uploaded_file)
                    else:
                        raw_data = pd.read_excel(uploaded_file)
                
                st.session_state.raw_data = raw_data
                st.success(f"Data loaded successfully! Shape: {raw_data.shape}")
                
                st.subheader("Data Preview")
                st.dataframe(raw_data.head())
                
                with st.expander("Data Quality Report"):
                    quality_report = self.data_processor.generate_data_quality_report(raw_data)
                    self._display_data_quality_report(quality_report)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        
        if 'raw_data' in st.session_state:
            st.subheader("Process Data")
            
            if st.button("Process Transaction Data"):
                with st.spinner("Processing data..."):
                    try:
                        cleaned_data = self.data_processor.clean_transaction_data(
                            st.session_state.raw_data
                        )
                        
                        customer_features = self.data_processor.create_customer_features(
                            cleaned_data
                        )
                        
                        self.db_manager.create_tables()
                        
                        self.db_manager.insert_data('transactions', cleaned_data)
                        self.db_manager.insert_data('customers', customer_features)
                        
                        st.session_state.customer_data = customer_features
                        st.session_state.transaction_data = cleaned_data
                        st.session_state.data_loaded = True
                        
                        st.success("Data processed and stored successfully!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Customers", len(customer_features))
                        with col2:
                            st.metric("Transactions", len(cleaned_data))
                        
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
    
    def rfm_analysis_page(self):
        st.header("🔍 RFM Analysis")
        
        if not st.session_state.data_loaded:
            st.warning("Please upload and process your data first!")
            return
        
        try:
            customer_data = self.db_manager.get_customer_data()
            
            if customer_data.empty:
                st.warning("No customer data found.")
                return
            
            st.sidebar.subheader("RFM Settings")
            quantiles = st.sidebar.slider("Number of Quantiles", 3, 5, 5)
            
            if st.button("Run RFM Analysis"):
                with st.spinner("Calculating RFM scores..."):
                    rfm_data = self.rfm_analyzer.calculate_rfm(customer_data)
                    
                    segmented_data, segment_stats = self.rfm_analyzer.segment_customers(rfm_data)
                    
                    self.db_manager.update_rfm_scores(segmented_data)
                    
                    st.success("RFM analysis completed!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("RFM Distribution")
                        fig_rfm = self.visualizer.plot_rfm_distribution(rfm_data)
                        st.plotly_chart(fig_rfm, use_container_width=True)
                    
                    with col2:
                        st.subheader("Customer Segments")
                        fig_segments = self.visualizer.plot_customer_segments(segmented_data)
                        st.plotly_chart(fig_segments, use_container_width=True)
                    
                    st.subheader("Segment Performance")
                    st.dataframe(segment_stats)
                    
                    segmented_data, recommendations = self.rfm_analyzer.recommend_actions(segmented_data)
                    
                    st.subheader("Recommended Actions")
                    self._display_recommendations(recommendations)
                    
                    if st.button("Export RFM Results"):
                        export_file = self.rfm_analyzer.export_rfm_analysis(segmented_data)
                        st.success(f"Results exported to {export_file}")
            
        except Exception as e:
            st.error(f"Error in RFM analysis: {str(e)}")
    
    def churn_prediction_page(self):
        st.header("⚠️ Churn Prediction")
        
        if not st.session_state.data_loaded:
            st.warning("Please upload and process your data first!")
            return
        
        try:
            customer_data = self.db_manager.get_customer_data()
            transaction_data = self.db_manager.get_transaction_data()
            
            st.sidebar.subheader("Churn Settings")
            churn_threshold = st.sidebar.slider("Churn Threshold (days)", 30, 180, 90)
            
            if st.button("Train Churn Models"):
                with st.spinner("Training churn prediction models..."):
                    # Features as of a cutoff, label = no purchase in the following
                    # `churn_threshold` days (avoids label leakage, see ChurnPredictor)
                    labeled_data, past_transactions = self.churn_predictor.build_temporal_training_set(
                        transaction_data, self.data_processor.create_customer_features,
                        horizon_days=churn_threshold
                    )
                    
                    feature_data = self.churn_predictor.engineer_features(
                        labeled_data, past_transactions
                    )
                    
                    X_train, X_test, y_train, y_test, feature_names = self.churn_predictor.prepare_training_data(
                        feature_data
                    )
                    
                    model_performance = self.churn_predictor.train_models(
                        X_train, y_train, X_test, y_test
                    )
                    
                    model_path = self.churn_predictor.save_models()
                    
                    st.success("Churn prediction models trained successfully!")
                    
                    st.subheader("Model Performance")
                    performance_df = pd.DataFrame(model_performance).T
                    st.dataframe(performance_df.round(3))
                    
                    st.subheader("Feature Importance")
                    feature_importance = self.churn_predictor.get_feature_importance()
                    fig_importance = self.visualizer.plot_feature_importance(
                        feature_importance['feature'].values,
                        feature_importance['importance'].values,
                        "Churn Prediction - Feature Importance"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            if st.button("Predict Churn Risk"):
                with st.spinner("Predicting churn risk..."):
                    # Streamlit re-creates this object on every rerun, so fall back to
                    # the model saved by "Train Churn Models"
                    if self.churn_predictor.rf_model is None:
                        saved = self.churn_predictor.models_dir / "churn_model.pkl"
                        if saved.exists():
                            self.churn_predictor.load_models(saved)
                    feature_data = self.churn_predictor.engineer_features(customer_data, transaction_data)
                    
                    predictions = self.churn_predictor.predict_churn(feature_data)
                    
                    self.db_manager.update_churn_predictions(predictions)
                    
                    st.success("Churn predictions completed!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Risk Distribution")
                        risk_counts = predictions['risk_category'].value_counts()
                        fig_risk = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Churn Risk Distribution"
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)
                    
                    with col2:
                        st.subheader("High Risk Customers")
                        high_risk = predictions[predictions['risk_category'] == 'High']
                        st.metric("High Risk Count", len(high_risk))
                        st.dataframe(high_risk.head())
                    
                    st.subheader("Churn Analysis Insights")
                    # customer_data already holds (empty) prediction columns from the DB join
                    stale = [c for c in predictions.columns if c != 'customer_id' and c in customer_data.columns]
                    churn_analysis = self.churn_predictor.analyze_churn_factors(
                        customer_data.drop(columns=stale).merge(predictions, on='customer_id')
                    )
                    self._display_churn_insights(churn_analysis)
        
        except Exception as e:
            st.error(f"Error in churn prediction: {str(e)}")
    
    def clv_analysis_page(self):
        st.header("💰 Customer Lifetime Value")
        
        if not st.session_state.data_loaded:
            st.warning("Please upload and process your data first!")
            return
        
        try:
            customer_data = self.db_manager.get_customer_data()
            transaction_data = self.db_manager.get_transaction_data()
            
            st.sidebar.subheader("CLV Settings")
            prediction_period = st.sidebar.slider("Prediction Period (days)", 90, 730, 365)
            
            if st.button("Calculate Customer Lifetime Value"):
                with st.spinner("Calculating CLV..."):
                    clv_data = self.clv_calculator.prepare_clv_data(transaction_data, customer_data)
                    
                    historical_clv = self.clv_calculator.calculate_historical_clv(clv_data)
                    
                    bgf_model = self.clv_calculator.train_bgf_model(clv_data)
                    ggf_model = self.clv_calculator.train_ggf_model(clv_data)
                    ml_model, ml_metrics = self.clv_calculator.train_ml_model(historical_clv)
                    
                    clv_predictions = self.clv_calculator.predict_clv(
                        clv_data, method='combined', period_days=prediction_period
                    )
                    
                    self.db_manager.update_clv_predictions(clv_predictions)
                    
                    st.success("CLV analysis completed!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("CLV Distribution")
                        fig_clv = self.visualizer.plot_clv_analysis(clv_predictions)
                        st.plotly_chart(fig_clv, use_container_width=True)
                    
                    with col2:
                        st.subheader("High-Value Customers")
                        high_value = self.clv_calculator.identify_high_value_customers(clv_predictions)
                        st.metric("High-Value Count", len(high_value))
                        st.dataframe(high_value.head())
                    
                    st.subheader("CLV Metrics")
                    clv_roi = self.clv_calculator.calculate_clv_roi(clv_predictions)
                    self._display_clv_metrics(clv_roi)
                    
                    st.subheader("Cohort CLV Analysis")
                    cohort_clv = self.clv_calculator.calculate_cohort_clv(transaction_data)
                    st.dataframe(cohort_clv)
        
        except Exception as e:
            st.error(f"Error in CLV analysis: {str(e)}")
    
    def customer_segmentation_page(self):
        st.header("👥 Customer Segmentation")
        
        if not st.session_state.data_loaded:
            st.warning("Please upload and process your data first!")
            return
        
        try:
            customer_data = self.db_manager.get_customer_data()
            segment_summary = self.db_manager.get_segment_summary()
            
            if customer_data.empty:
                st.warning("No customer data with segments found. Please run RFM analysis first.")
                return
            
            st.subheader("Segment Overview")
            
            segments = customer_data['segment'].dropna().unique()
            selected_segment = st.selectbox("Select Segment to Analyze:", ['All'] + list(segments))
            
            if selected_segment != 'All':
                filtered_data = customer_data[customer_data['segment'] == selected_segment]
            else:
                filtered_data = customer_data
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Customers", len(filtered_data))
            
            with col2:
                avg_revenue = filtered_data['total_spent'].mean() if 'total_spent' in filtered_data.columns else 0
                st.metric("Avg Revenue", f"${avg_revenue:.0f}")
            
            with col3:
                if 'churn_probability' in filtered_data.columns:
                    avg_risk = filtered_data['churn_probability'].mean()
                    st.metric("Avg Churn Risk", f"{avg_risk:.1%}")
            
            st.subheader("Segment Comparison")
            if not segment_summary.empty:
                st.dataframe(segment_summary)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'segment' in customer_data.columns and 'risk_category' in customer_data.columns:
                    st.subheader("Risk by Segment")
                    fig_heatmap = self.visualizer.plot_churn_risk_heatmap(customer_data)
                    if fig_heatmap:
                        st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col2:
                if 'segment' in customer_data.columns:
                    st.subheader("Segment Revenue")
                    segment_revenue = customer_data.groupby('segment')['total_spent'].sum().sort_values(ascending=True)
                    fig_revenue = px.bar(
                        x=segment_revenue.values,
                        y=segment_revenue.index,
                        orientation='h',
                        title="Revenue by Segment"
                    )
                    st.plotly_chart(fig_revenue, use_container_width=True)
            
            st.subheader("Customer Details")
            
            search_customer = st.number_input("Search Customer ID:", min_value=0, step=1)
            
            if search_customer > 0:
                customer_details = customer_data[customer_data['customer_id'] == search_customer]
                if not customer_details.empty:
                    st.write("### Customer Profile")
                    st.dataframe(customer_details)
                    
                    if st.button("Show Customer Timeline"):
                        transaction_data = self.db_manager.get_transaction_data(customer_id=search_customer)
                        if not transaction_data.empty:
                            fig_timeline = self.visualizer.plot_customer_timeline(transaction_data, search_customer)
                            st.plotly_chart(fig_timeline, use_container_width=True)
                else:
                    st.warning("Customer not found.")
        
        except Exception as e:
            st.error(f"Error in customer segmentation: {str(e)}")
    
    def business_insights_page(self):
        st.header("📈 Business Insights")
        
        if not st.session_state.data_loaded:
            st.warning("Please upload and process your data first!")
            return
        
        try:
            customer_data = self.db_manager.get_customer_data()
            
            st.subheader("Key Business Metrics")
            
            retention_rate = BusinessMetrics.calculate_retention_rate(customer_data)
            churn_rate = BusinessMetrics.calculate_churn_rate(customer_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Retention Rate", f"{retention_rate:.1%}")
            
            with col2:
                st.metric("Churn Rate", f"{churn_rate:.1%}")
            
            with col3:
                avg_clv = customer_data['predicted_clv'].mean() if 'predicted_clv' in customer_data.columns else 0
                st.metric("Average CLV", f"${avg_clv:.0f}")
            
            with col4:
                high_risk_count = len(customer_data[customer_data['risk_category'] == 'High']) if 'risk_category' in customer_data.columns else 0
                st.metric("High Risk Customers", high_risk_count)
            
            st.subheader("Segment Performance Analysis")
            if 'segment' in customer_data.columns:
                segment_performance = BusinessMetrics.calculate_segment_performance(customer_data)
                if segment_performance:
                    performance_df = pd.DataFrame(segment_performance).T
                    st.dataframe(performance_df)
            
            st.subheader("Recommended Actions")
            
            if 'risk_category' in customer_data.columns:
                high_risk_customers = self.db_manager.get_high_risk_customers()
                if not high_risk_customers.empty:
                    st.write("#### 🚨 Immediate Attention Required")
                    st.write(f"**{len(high_risk_customers)} high-risk customers** need immediate retention efforts")
                    
                    with st.expander("View High-Risk Customers"):
                        st.dataframe(high_risk_customers)
                    
                    campaign_cost = st.number_input("Retention Campaign Cost per Customer ($)", value=50.0)
                    if campaign_cost > 0:
                        prevented_churn = len(high_risk_customers) * 0.3  # Assume 30% success rate
                        roi_metrics = BusinessMetrics.roi_retention_campaign(
                            campaign_cost * len(high_risk_customers),
                            prevented_churn,
                            avg_clv
                        )
                        
                        st.write("##### Campaign ROI Projection")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Campaign Cost", f"${roi_metrics['campaign_cost']:,.0f}")
                        with col2:
                            st.metric("Revenue at Risk", f"${roi_metrics['revenue_saved']:,.0f}")
                        with col3:
                            st.metric("ROI", f"{roi_metrics['roi_percentage']:.1f}%")
            
            st.subheader("Growth Opportunities")
            
            if 'predicted_clv' in customer_data.columns:
                high_clv_threshold = customer_data['predicted_clv'].quantile(0.8)
                st.write(f"#### 💎 Focus on High-Value Segments")
                st.write(f"Target customers with predicted CLV > ${high_clv_threshold:.0f}")
            
            if 'segment' in customer_data.columns:
                loyal_customers = customer_data[customer_data['segment'] == 'Loyal Customers']
                if not loyal_customers.empty:
                    st.write(f"#### 🎯 Cross-Sell to {len(loyal_customers)} Loyal Customers")
                    st.write("These customers have high engagement and are likely to purchase additional products")
        
        except Exception as e:
            st.error(f"Error loading business insights: {str(e)}")
    
    def settings_page(self):
        st.header("⚙️ Settings")
        
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**RFM Analysis**")
            quantiles = st.slider("RFM Quantiles", 3, 5, 5)
            
            st.write("**Churn Prediction**")
            churn_threshold = st.slider("Churn Threshold (days)", 30, 180, 90)
        
        with col2:
            st.write("**CLV Prediction**")
            clv_period = st.slider("Prediction Period (days)", 90, 730, 365)
            
            st.write("**Risk Categories**")
            for category, config in RISK_CATEGORIES.items():
                st.write(f"- {category}: ≤ {config['threshold']}")
        
        st.subheader("Data Export")
        
        export_format = st.selectbox("Export Format", ["CSV", "Excel"])
        
        if st.button("Export Customer Report"):
            try:
                export_path = self.db_manager.export_customer_report(format=export_format.lower())
                st.success(f"Report exported to {export_path}")
                
                with open(export_path, 'rb') as f:
                    st.download_button(
                        label=f"Download {export_format} Report",
                        data=f.read(),
                        file_name=f"customer_report.{export_format.lower()}",
                        mime="application/octet-stream"
                    )
            except Exception as e:
                st.error(f"Export error: {str(e)}")
        
        st.subheader("Database Management")
        
        if st.button("Reset Database"):
            if st.checkbox("I understand this will delete all data"):
                self.db_manager.create_tables()
                st.success("Database reset successfully!")
                st.session_state.data_loaded = False
                st.session_state.analysis_complete = False
    
    def _calculate_summary_stats(self, customer_data):
        stats = {
            'total_customers': len(customer_data),
            'total_revenue': customer_data['total_spent'].sum() if 'total_spent' in customer_data.columns else 0,
            'avg_clv': customer_data['predicted_clv'].mean() if 'predicted_clv' in customer_data.columns else 0,
            'churn_rate': 0
        }
        
        if 'churn_probability' in customer_data.columns:
            stats['churn_rate'] = (customer_data['churn_probability'] > 0.5).mean()
        
        return stats
    
    def _display_data_quality_report(self, quality_report):
        st.write("**Dataset Info:**")
        st.write(f"- Shape: {quality_report['dataset_info']['shape']}")
        st.write(f"- Memory Usage: {quality_report['dataset_info']['memory_usage'] / 1024 / 1024:.1f} MB")
        
        if quality_report['consistency_issues']:
            st.write("**Data Issues:**")
            for issue in quality_report['consistency_issues']:
                st.warning(issue)
    
    def _display_recent_trends(self, customer_data, transaction_data=None):
        """Last 30 days vs the 30 days before, anchored on the newest order in the data."""
        if transaction_data is None or transaction_data.empty:
            st.caption("Process transaction data to see recent trends.")
            return

        tx = transaction_data.copy()
        tx['order_date'] = pd.to_datetime(tx['order_date'])
        end = tx['order_date'].max()
        recent = tx[tx['order_date'] > end - pd.Timedelta(days=30)]
        prior = tx[(tx['order_date'] <= end - pd.Timedelta(days=30))
                   & (tx['order_date'] > end - pd.Timedelta(days=60))]

        first_order = tx.groupby('customer_id')['order_date'].min()
        new_customers = int((first_order > end - pd.Timedelta(days=30)).sum())

        def change(now, before):
            return f"{(now - before) / before:+.1%}" if before else "n/a"

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Last 30 days**")
            st.write(f"- New customers: {new_customers:,}")
            st.write(f"- Active customers: {recent['customer_id'].nunique():,}")
            st.write(f"- Revenue: ${recent['order_value'].sum():,.0f}")
        with col2:
            st.write("**vs. previous 30 days**")
            st.write(f"- Revenue: {change(recent['order_value'].sum(), prior['order_value'].sum())}")
            st.write(f"- Orders: {change(len(recent), len(prior))}")
            st.write(f"- Active customers: {change(recent['customer_id'].nunique(), prior['customer_id'].nunique())}")

    def _display_recommendations(self, recommendations):
        for segment, rec in recommendations.items():
            with st.expander(f"{segment} - {rec['strategy']}"):
                st.write(f"**Priority:** {rec['priority']}")
                st.write("**Recommended Actions:**")
                for action in rec['actions']:
                    st.write(f"• {action}")
    
    def _display_churn_insights(self, churn_analysis):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("High Risk Customers", churn_analysis['high_risk_count'])
            st.metric("High Risk %", f"{churn_analysis['high_risk_percentage']:.1f}%")
        
        with col2:
            st.metric("Average Churn Probability", f"{churn_analysis['avg_churn_probability']:.1%}")
    
    def _display_clv_metrics(self, clv_metrics):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predicted CLV", f"${clv_metrics['total_predicted_clv']:,.0f}")
        
        with col2:
            st.metric("Average CLV", f"${clv_metrics['average_clv']:.0f}")
        
        with col3:
            st.metric("Median CLV", f"${clv_metrics['median_clv']:.0f}")


# Run the application
if __name__ == "__main__":
    app = CustomerIQApp()
    app.main()
