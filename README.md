# CustomerIQ: Intelligent Customer Segmentation & Churn Prevention Platform

A comprehensive data science platform that helps e-commerce businesses understand their customers, predict churn risk, and optimize retention strategies through advanced analytics and machine learning.

## 🚀 Features

- **Customer 360° View**: Individual customer profiles with risk scores
- **RFM Analysis**: Automated Recency, Frequency, Monetary value calculations
- **Churn Prediction**: ML-powered risk assessment with Random Forest + XGBoost
- **Customer Lifetime Value**: Predictive CLV using survival analysis
- **Interactive Dashboard**: Real-time predictions and visualizations
- **Auto-Reports**: Weekly/monthly executive summaries

## 🛠️ Tech Stack

- **Backend**: Python 3.9+
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, Seaborn
- **Web App**: Streamlit
- **Database**: SQLite/PostgreSQL

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customeriq.git
cd customeriq
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Run the Streamlit app:
```bash
streamlit run src/app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload your customer data or use the sample dataset

## 📊 Project Structure

```
customeriq/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── data_processor.py      # Data cleaning and preprocessing
│   ├── rfm_analyzer.py        # RFM analysis engine
│   ├── churn_predictor.py     # Churn prediction model
│   ├── clv_calculator.py      # Customer Lifetime Value
│   └── database.py            # Database operations
├── data/
│   ├── raw/                   # Raw data files
│   ├── processed/             # Cleaned data
│   └── sample_data.csv        # Sample dataset
├── models/
│   ├── churn_model.pkl        # Trained churn prediction model
│   └── model_artifacts/       # Model artifacts and metadata
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_rfm_analysis.ipynb
│   ├── 03_churn_modeling.ipynb
│   └── 04_clv_analysis.ipynb
├── utils/
│   ├── __init__.py
│   ├── visualization.py       # Plotting utilities
│   └── metrics.py            # Custom metrics
├── config/
│   └── settings.py           # Configuration settings
└── tests/
    └── test_*.py             # Unit tests
```

## 📈 Usage Examples

### RFM Analysis
```python
from src.rfm_analyzer import RFMAnalyzer

analyzer = RFMAnalyzer()
rfm_scores = analyzer.calculate_rfm(customer_data)
segments = analyzer.segment_customers(rfm_scores)
```

### Churn Prediction
```python
from src.churn_predictor import ChurnPredictor

predictor = ChurnPredictor()
predictions = predictor.predict_churn(customer_features)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
