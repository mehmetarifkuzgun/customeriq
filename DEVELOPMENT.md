# CustomerIQ Development Guide

## Project Overview

CustomerIQ is an intelligent customer segmentation and churn prevention platform built with Python. It provides comprehensive analytics for e-commerce businesses to understand their customers, predict churn risk, and optimize retention strategies.

## Architecture

### Core Components

1. **Data Processing Pipeline** (`src/data_processor.py`)
   - Automated data cleaning and validation
   - Feature engineering for customer analytics
   - Outlier detection and handling

2. **RFM Analysis Engine** (`src/rfm_analyzer.py`)
   - Recency, Frequency, Monetary value calculations
   - Dynamic customer segmentation
   - Business recommendations generation

3. **Churn Prediction Model** (`src/churn_predictor.py`)
   - Ensemble ML models (Random Forest + XGBoost)
   - Feature importance analysis
   - Risk categorization (Low/Medium/High)

4. **Customer Lifetime Value Calculator** (`src/clv_calculator.py`)
   - Predictive CLV using survival analysis
   - Cohort analysis for retention curves
   - Revenue impact forecasting

5. **Database Management** (`src/database.py`)
   - SQLite/PostgreSQL support
   - Data persistence and retrieval
   - Query optimization

6. **Visualization Components** (`utils/visualization.py`)
   - Interactive Plotly charts
   - Dashboard components
   - Custom visualizations

7. **Streamlit Application** (`src/app.py`)
   - Web-based dashboard
   - Real-time analytics
   - Export functionality

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/customeriq.git
   cd customeriq
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests:**
   ```bash
   python -m pytest tests/ -v
   ```

5. **Start development server:**
   ```bash
   streamlit run src/app.py
   ```

## Code Structure

```
customeriq/
├── src/                    # Source code
│   ├── app.py             # Main Streamlit application
│   ├── data_processor.py   # Data cleaning and preprocessing
│   ├── rfm_analyzer.py     # RFM analysis engine
│   ├── churn_predictor.py  # Churn prediction models
│   ├── clv_calculator.py   # Customer lifetime value
│   └── database.py         # Database operations
├── utils/                  # Utility modules
│   ├── visualization.py    # Plotting and charts
│   └── metrics.py         # Custom metrics and evaluation
├── config/                 # Configuration files
│   └── settings.py        # Application settings
├── data/                   # Data storage
│   ├── raw/               # Raw data files
│   └── processed/         # Cleaned and processed data
├── models/                 # Trained ML models
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_rfm_analysis.ipynb
│   ├── 03_churn_modeling.ipynb
│   └── 04_clv_analysis.ipynb
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Development Workflow

### 1. Adding New Features

1. Create a feature branch:
   ```bash
   git checkout -b feature/new-feature-name
   ```

2. Implement the feature following the existing code structure
3. Add unit tests in the `tests/` directory
4. Update documentation if necessary
5. Run tests to ensure everything works:
   ```bash
   python -m pytest tests/ -v
   ```

6. Create a pull request

### 2. Data Pipeline Development

When adding new data processing features:

1. **Extend DataProcessor class** (`src/data_processor.py`)
2. **Add validation methods** for data quality checks
3. **Update database schema** if new fields are required
4. **Add corresponding tests** in `tests/test_customeriq.py`

Example:
```python
def new_feature_extraction(self, data):
    """Extract new customer features"""
    # Implementation here
    return enhanced_data
```

### 3. ML Model Development

For new machine learning models:

1. **Create model class** following existing patterns
2. **Implement training pipeline** with cross-validation
3. **Add evaluation metrics** and model performance tracking
4. **Save/load model artifacts** using joblib
5. **Integrate with main application**

Example structure:
```python
class NewPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def train_model(self, X, y):
        # Training implementation
        pass
    
    def predict(self, X):
        # Prediction implementation
        pass
    
    def save_model(self, path):
        # Model persistence
        pass
```

### 4. Visualization Development

For new visualizations:

1. **Add methods to CustomerVisualizer** class
2. **Use Plotly for interactive charts**
3. **Follow consistent styling** (colors, themes)
4. **Add to Streamlit app** if needed

Example:
```python
def plot_new_analysis(self, data):
    """Create new analysis visualization"""
    fig = go.Figure()
    # Plot implementation
    return fig
```

## Testing Guidelines

### Unit Testing

- Write tests for all new functions and classes
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies when necessary

Example test:
```python
def test_new_feature(self):
    """Test new feature functionality"""
    # Arrange
    test_data = self.create_test_data()
    
    # Act
    result = self.processor.new_feature(test_data)
    
    # Assert
    self.assertIsNotNone(result)
    self.assertEqual(len(result), expected_length)
```

### Integration Testing

- Test complete workflows end-to-end
- Verify data pipeline integrity
- Test model training and prediction pipelines

### Performance Testing

- Monitor memory usage for large datasets
- Test processing time for different data sizes
- Optimize bottlenecks in data processing

## Configuration Management

### Settings (`config/settings.py`)

All configuration should be centralized:

```python
# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Business rules
CHURN_THRESHOLD_DAYS = 90
CLV_PREDICTION_PERIOD = 365

# UI settings
PAGE_TITLE = "CustomerIQ"
COLOR_PALETTE = ["#1f77b4", "#ff7f0e", ...]
```

### Environment Variables

Use environment variables for sensitive data:
- Database URLs
- API keys
- External service credentials

## Database Design

### Schema Design

Current tables:
- `customers`: Customer master data
- `transactions`: Transaction history
- `rfm_scores`: RFM analysis results
- `churn_predictions`: Churn risk scores
- `clv_predictions`: Customer lifetime value

### Migration Strategy

For schema changes:
1. Create migration scripts
2. Backup existing data
3. Apply changes incrementally
4. Validate data integrity

## Performance Optimization

### Data Processing

1. **Vectorized operations** with pandas/numpy
2. **Chunked processing** for large datasets
3. **Caching** of expensive computations
4. **Parallel processing** where applicable

### Memory Management

1. **Data type optimization** (use appropriate dtypes)
2. **Memory profiling** with tools like memory_profiler
3. **Garbage collection** for long-running processes

### Database Optimization

1. **Proper indexing** on frequently queried columns
2. **Query optimization** using EXPLAIN ANALYZE
3. **Connection pooling** for high concurrency

## Deployment Strategies

### Local Development

```bash
streamlit run src/app.py --server.port 8501
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py"]
```

### Cloud Deployment

Options:
1. **Streamlit Cloud**: Direct deployment from GitHub
2. **Heroku**: Platform-as-a-Service deployment
3. **AWS/Azure/GCP**: Container-based deployment
4. **Docker**: Containerized deployment

## Monitoring and Logging

### Application Monitoring

1. **Performance metrics**: Response times, memory usage
2. **Error tracking**: Exception monitoring and alerting
3. **User analytics**: Feature usage and user behavior

### Model Monitoring

1. **Model performance drift**: Accuracy degradation over time
2. **Data drift**: Changes in input data distribution
3. **Prediction monitoring**: Anomaly detection in predictions

### Logging Best Practices

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use throughout application
logger.info("Processing customer data...")
logger.warning("Low data quality detected")
logger.error("Model training failed", exc_info=True)
```

## Contributing Guidelines

### Code Quality

1. **Follow PEP 8** style guidelines
2. **Use type hints** where appropriate
3. **Write docstrings** for all functions and classes
4. **Keep functions small** and focused
5. **Use meaningful variable names**

### Documentation

1. **Update README** for new features
2. **Document API changes**
3. **Provide usage examples**
4. **Maintain changelog**

### Code Review Process

1. **Create descriptive pull requests**
2. **Include test results**
3. **Explain design decisions**
4. **Address review feedback promptly**

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce data chunk sizes
2. **Import errors**: Check virtual environment activation
3. **Database connection issues**: Verify connection string
4. **Model loading errors**: Check model file paths

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Performance Profiling

Use cProfile for performance analysis:
```bash
python -m cProfile -s cumulative src/app.py
```

## Security Considerations

### Data Protection

1. **Anonymize sensitive data**
2. **Encrypt data at rest**
3. **Secure data transmission**
4. **Implement access controls**

### Application Security

1. **Input validation** for all user inputs
2. **SQL injection prevention**
3. **Cross-site scripting (XSS) protection**
4. **Secure configuration management**

## Future Enhancements

### Planned Features

1. **Real-time data ingestion**
2. **Advanced ML models** (deep learning)
3. **A/B testing framework**
4. **Multi-tenant support**
5. **Advanced reporting** (PDF/Excel exports)

### Technical Improvements

1. **Microservices architecture**
2. **API development** (REST/GraphQL)
3. **Message queue integration**
4. **Automated ML pipeline**
5. **Enhanced visualization**

## Support and Resources

### Documentation

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Community

- GitHub Issues for bug reports
- Discussions for feature requests
- Stack Overflow for general questions

### Contact

- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
