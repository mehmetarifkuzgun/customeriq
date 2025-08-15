# CustomerIQ - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Installation
```bash
# Clone and setup
git clone https://github.com/yourusername/customeriq.git
cd customeriq

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run src/app.py
```

Open your browser to `http://localhost:8501`

### 3. Quick Demo
1. Navigate to **"Data Upload"** page
2. Click **"Generate Sample Data"**
3. Go to **"RFM Analysis"** to see customer segmentation
4. Check **"Churn Prediction"** for risk analysis
5. View **"CLV Analysis"** for customer value insights

## 📊 Using Your Own Data

### Data Format Requirements

Your CSV file should contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `customer_id` | Unique customer identifier | "CUST_001" |
| `transaction_date` | Purchase date | "2023-01-15" |
| `amount` | Transaction amount | 156.99 |
| `product_category` | Product category (optional) | "Electronics" |

### Sample Data Structure
```csv
customer_id,transaction_date,amount,product_category
CUST_001,2023-01-15,156.99,Electronics
CUST_001,2023-02-20,89.50,Books
CUST_002,2023-01-10,299.99,Clothing
```

## 🎯 Key Features

### 📈 RFM Analysis
- **Recency**: How recently did the customer purchase?
- **Frequency**: How often do they purchase?
- **Monetary**: How much do they spend?

**Business Value:**
- Identify VIP customers
- Target dormant customers
- Optimize marketing campaigns

### 🚨 Churn Prediction
- ML models predict customer churn risk
- Risk categories: Low, Medium, High
- Feature importance analysis

**Business Value:**
- Proactive retention campaigns
- Reduce customer acquisition costs
- Improve customer lifetime value

### 💰 Customer Lifetime Value (CLV)
- Predict future customer value
- Cohort analysis
- Revenue forecasting

**Business Value:**
- Optimize marketing spend
- Identify high-value customers
- Strategic resource allocation

## 📋 Main Dashboard Features

### 1. Overview Dashboard
- Key metrics summary
- Customer distribution
- Revenue trends
- Quick insights

### 2. Data Upload & Management
- Upload CSV files
- Data validation
- Generate sample data
- Data quality reports

### 3. RFM Analysis
- Interactive customer segmentation
- Segment distribution charts
- Business recommendations
- Export capabilities

### 4. Churn Prediction
- Risk assessment dashboard
- Model performance metrics
- Feature importance analysis
- Customer action plans

### 5. CLV Analysis
- Customer value predictions
- Cohort analysis
- Revenue impact forecasting
- Strategic insights

## 🛠️ Advanced Usage

### Custom Configuration

Edit `config/settings.py` to customize:

```python
# Churn prediction threshold
CHURN_THRESHOLD_DAYS = 90  # Days since last purchase

# RFM scoring parameters
RFM_QUANTILES = 5  # Number of quantiles for scoring

# Risk categories
CHURN_RISK_THRESHOLDS = {
    'low': 0.3,
    'medium': 0.7,
    'high': 1.0
}
```

### Database Integration

For production use with PostgreSQL:

```python
# In config/settings.py
DATABASE_URL = "postgresql://user:password@localhost:5432/customeriq"
```

### Jupyter Notebooks

Explore the methodology:
```bash
jupyter notebook notebooks/
```

Available notebooks:
- `01_data_exploration.ipynb` - Data analysis and insights
- `02_rfm_analysis.ipynb` - RFM methodology explained
- `03_churn_modeling.ipynb` - Churn prediction deep dive
- `04_clv_analysis.ipynb` - CLV calculation methods

## 🔧 Troubleshooting

### Common Issues

**Issue**: Import errors
```bash
# Solution: Activate virtual environment
venv\Scripts\activate
pip install -r requirements.txt
```

**Issue**: Data upload fails
- Check CSV format matches requirements
- Ensure date format is YYYY-MM-DD
- Verify numeric columns contain valid numbers

**Issue**: Models not training
- Ensure sufficient data (minimum 100 customers)
- Check for data quality issues
- Verify date ranges are reasonable

**Issue**: Streamlit app won't start
```bash
# Check if port is in use
netstat -an | findstr 8501

# Use different port
streamlit run src/app.py --server.port 8502
```

### Performance Tips

**For Large Datasets:**
1. Use data sampling for initial exploration
2. Process data in chunks
3. Consider database backend for production

**Memory Optimization:**
```python
# Use appropriate data types
df['customer_id'] = df['customer_id'].astype('category')
df['amount'] = df['amount'].astype('float32')
```

## 📊 Understanding the Results

### RFM Segments

| Segment | Description | Action |
|---------|-------------|--------|
| **Champions** | High R, F, M scores | Reward, upsell premium products |
| **Loyal Customers** | High F, M; moderate R | Offer loyalty programs |
| **Potential Loyalists** | Recent customers, moderate F | Convert to loyal customers |
| **At Risk** | Low R; high F, M | Re-engagement campaigns |
| **Can't Lose Them** | Very low R; high F, M | Immediate intervention |
| **Hibernating** | Low R, F, M | Reactivation campaigns |

### Churn Risk Interpretation

- **Low Risk (0-30%)**: Engaged customers, maintain satisfaction
- **Medium Risk (30-70%)**: Monitor closely, implement retention
- **High Risk (70-100%)**: Immediate action required

### CLV Insights

- **High CLV**: Focus retention, upselling opportunities
- **Medium CLV**: Growth potential through engagement
- **Low CLV**: Cost-effective retention strategies

## 🚀 Deployment Options

### Local Development
```bash
streamlit run src/app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

### Docker
```bash
docker build -t customeriq .
docker run -p 8501:8501 customeriq
```

### Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
```

## 📈 Best Practices

### Data Quality
1. **Clean data regularly** - Remove duplicates, handle missing values
2. **Validate inputs** - Check date formats, numeric ranges
3. **Monitor data drift** - Track changes in customer behavior

### Model Management
1. **Retrain models monthly** - Capture seasonal patterns
2. **Monitor performance** - Track accuracy over time
3. **A/B test predictions** - Validate model effectiveness

### Business Integration
1. **Start with pilot** - Test with subset of customers
2. **Measure impact** - Track ROI of actions taken
3. **Iterate based on results** - Refine strategies continuously

## 🎓 Learning Resources

### Understanding Customer Analytics
- [RFM Analysis Guide](https://blog.hubspot.com/service/what-does-rfm-stand-for)
- [Churn Prediction Methods](https://towardsdatascience.com/churn-prediction-3a4a36c2129a)
- [Customer Lifetime Value](https://blog.hubspot.com/service/what-does-cltv-mean)

### Technical Documentation
- [Streamlit Docs](https://docs.streamlit.io/)
- [Pandas Tutorials](https://pandas.pydata.org/docs/getting_started/tutorials.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## 💡 Pro Tips

1. **Start Small**: Begin with sample data to understand the platform
2. **Regular Updates**: Update models monthly for best performance
3. **Action-Oriented**: Focus on actionable insights, not just analytics
4. **Measure Impact**: Track the ROI of your retention strategies
5. **Continuous Learning**: Use Jupyter notebooks to understand methodology

## 🆘 Support

- **Documentation**: Check `DEVELOPMENT.md` for detailed technical guide
- **Issues**: Report bugs on GitHub Issues
- **Questions**: Use GitHub Discussions for community support
- **Email**: contact@customeriq.com (if available)

---

**Ready to start?** Run `streamlit run src/app.py` and begin your customer analytics journey! 🚀
