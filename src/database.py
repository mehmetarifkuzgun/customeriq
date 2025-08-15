import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from pathlib import Path
from config.settings import DATABASE_URL, PROJECT_ROOT

class DatabaseManager:
    def __init__(self, db_url=None):
        self.db_url = db_url or DATABASE_URL
        self.engine = create_engine(self.db_url)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_tables(self):
        try:
            with self.engine.connect() as conn:
                # Customer data table
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id INTEGER PRIMARY KEY,
                    first_purchase_date DATE,
                    last_purchase_date DATE,
                    total_orders INTEGER,
                    total_spent REAL,
                    avg_order_value REAL,
                    days_since_last_purchase INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """))
                
                # Transaction data table
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id INTEGER PRIMARY KEY,
                    customer_id INTEGER,
                    order_date DATE,
                    order_value REAL,
                    product_category TEXT,
                    quantity INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                )
                """))
                
                # RFM scores table
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rfm_scores (
                    customer_id INTEGER PRIMARY KEY,
                    recency INTEGER,
                    frequency INTEGER,
                    monetary REAL,
                    r_score INTEGER,
                    f_score INTEGER,
                    m_score INTEGER,
                    rfm_score TEXT,
                    segment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                )
                """))
                
                # Churn predictions table
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS churn_predictions (
                    customer_id INTEGER PRIMARY KEY,
                    churn_probability REAL,
                    risk_category TEXT,
                    prediction_date DATE,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                )
                """))
                
                # CLV predictions table
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS clv_predictions (
                    customer_id INTEGER PRIMARY KEY,
                    predicted_clv REAL,
                    prediction_period_days INTEGER,
                    prediction_date DATE,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                )
                """))
                
                conn.commit()
                self.logger.info("Database tables created successfully")
                
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    def insert_data(self, table_name, data, if_exists='replace'):
        try:
            data.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            self.logger.info(f"Data inserted into {table_name} successfully")
        except SQLAlchemyError as e:
            self.logger.error(f"Error inserting data into {table_name}: {e}")
            raise
    
    def fetch_data(self, query, params=None):
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn, params=params)
            return result
        except SQLAlchemyError as e:
            self.logger.error(f"Error fetching data: {e}")
            raise
    
    def get_customer_data(self, customer_id=None):
        query = """
        SELECT c.*, r.segment, r.rfm_score, 
               ch.churn_probability, ch.risk_category,
               clv.predicted_clv
        FROM customers c
        LEFT JOIN rfm_scores r ON c.customer_id = r.customer_id
        LEFT JOIN churn_predictions ch ON c.customer_id = ch.customer_id
        LEFT JOIN clv_predictions clv ON c.customer_id = clv.customer_id
        """
        
        if customer_id:
            query += " WHERE c.customer_id = :customer_id"
            return self.fetch_data(query, params={'customer_id': customer_id})
        
        return self.fetch_data(query)
    
    def get_transaction_data(self, customer_id=None, start_date=None, end_date=None):
        query = "SELECT * FROM transactions WHERE 1=1"
        params = {}
        
        if customer_id:
            query += " AND customer_id = :customer_id"
            params['customer_id'] = customer_id
            
        if start_date:
            query += " AND order_date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND order_date <= :end_date"
            params['end_date'] = end_date
        
        query += " ORDER BY order_date"
        
        return self.fetch_data(query, params)
    
    def update_rfm_scores(self, rfm_data):
        try:
            # Drop existing RFM scores
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM rfm_scores"))
                conn.commit()
            
            # Insert new RFM scores
            self.insert_data('rfm_scores', rfm_data, if_exists='append')
            self.logger.info("RFM scores updated successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error updating RFM scores: {e}")
            raise
    
    def update_churn_predictions(self, predictions_data):
        try:
            # Drop existing predictions
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM churn_predictions"))
                conn.commit()
            
            # Insert new predictions
            self.insert_data('churn_predictions', predictions_data, if_exists='append')
            self.logger.info("Churn predictions updated successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error updating churn predictions: {e}")
            raise
    
    def update_clv_predictions(self, clv_data):
        try:
            # Drop existing CLV predictions
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM clv_predictions"))
                conn.commit()
            
            # Insert new CLV predictions
            self.insert_data('clv_predictions', clv_data, if_exists='append')
            self.logger.info("CLV predictions updated successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error updating CLV predictions: {e}")
            raise
    
    def get_segment_summary(self):
        query = """
        SELECT 
            r.segment,
            COUNT(*) as customer_count,
            AVG(c.total_spent) as avg_total_spent,
            AVG(c.total_orders) as avg_total_orders,
            AVG(c.days_since_last_purchase) as avg_days_since_last,
            AVG(ch.churn_probability) as avg_churn_probability,
            AVG(clv.predicted_clv) as avg_predicted_clv
        FROM customers c
        LEFT JOIN rfm_scores r ON c.customer_id = r.customer_id
        LEFT JOIN churn_predictions ch ON c.customer_id = ch.customer_id
        LEFT JOIN clv_predictions clv ON c.customer_id = clv.customer_id
        WHERE r.segment IS NOT NULL
        GROUP BY r.segment
        ORDER BY avg_total_spent DESC
        """
        
        return self.fetch_data(query)
    
    def get_high_risk_customers(self, risk_threshold=0.7):
        query = """
        SELECT c.customer_id, c.total_spent, c.days_since_last_purchase,
               r.segment, ch.churn_probability, ch.risk_category
        FROM customers c
        JOIN churn_predictions ch ON c.customer_id = ch.customer_id
        LEFT JOIN rfm_scores r ON c.customer_id = r.customer_id
        WHERE ch.churn_probability >= :risk_threshold
        ORDER BY ch.churn_probability DESC
        """
        
        return self.fetch_data(query, params={'risk_threshold': risk_threshold})
    
    def export_customer_report(self, format='csv'):
        query = """
        SELECT 
            c.customer_id,
            c.first_purchase_date,
            c.last_purchase_date,
            c.total_orders,
            c.total_spent,
            c.avg_order_value,
            c.days_since_last_purchase,
            r.segment,
            r.rfm_score,
            r.r_score,
            r.f_score,
            r.m_score,
            ch.churn_probability,
            ch.risk_category,
            clv.predicted_clv
        FROM customers c
        LEFT JOIN rfm_scores r ON c.customer_id = r.customer_id
        LEFT JOIN churn_predictions ch ON c.customer_id = ch.customer_id
        LEFT JOIN clv_predictions clv ON c.customer_id = clv.customer_id
        ORDER BY c.total_spent DESC
        """
        
        data = self.fetch_data(query)
        
        if format.lower() == 'csv':
            filepath = PROJECT_ROOT / "data" / "customer_report.csv"
            data.to_csv(filepath, index=False)
        elif format.lower() == 'excel':
            filepath = PROJECT_ROOT / "data" / "customer_report.xlsx"
            data.to_excel(filepath, index=False)
        
        return filepath
    
    def close_connection(self):
        self.engine.dispose()
        self.logger.info("Database connection closed")
