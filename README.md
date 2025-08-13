# Marketing Data Analysis & Customer Insights

A comprehensive data analysis project examining customer behavior, spending patterns, and campaign effectiveness using statistical analysis and machine learning techniques.

## 📊 Project Overview

This project analyzes marketing data from 2,240 customers to uncover insights about:
- Customer spending patterns across product categories
- Income-based customer segmentation
- Campaign response effectiveness
- Customer lifetime value prediction
- Product recommendation systems

## 🎯 Key Findings

- **Income Impact**: Higher income customers spend significantly more on premium products (wines & meat)
- **Statistical Significance**: ANOVA tests show p-values < 0.001 for income-spending relationships
- **Customer Segmentation**: Identified distinct customer clusters with different purchasing behaviors
- **Campaign Effectiveness**: Built ML models to predict campaign response with high accuracy

## 🛠️ Technologies Used

- **Python 3.9+**
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Statistics**: SciPy, Statsmodels
- **Machine Learning**: Scikit-learn, TensorFlow
- **Environment**: Jupyter Notebook

## 📁 Project Structure

```
data-analyst-portfolio/
├── marketing_data_analysis.ipynb    # Main analysis notebook
├── data/
│   └── marketing_data.csv          # Dataset
├── requirements.txt                # Dependencies
├── README.md                      # Project documentation
└── venv_da/                       # Virtual environment
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/deathvadeR-afk/Marketing_data_analysis.git
cd data-analyst-portfolio
```

2. **Create virtual environment**
```bash
python -m venv venv_da
source venv_da/bin/activate  # On Windows: venv_da\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook marketing_data_analysis.ipynb
```

## 📈 Analysis Components

### 1. Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Descriptive statistics
- Distribution analysis
- Correlation matrices

### 2. Statistical Analysis
- ANOVA tests for income-spending relationships
- Hypothesis testing
- Confidence intervals
- Effect size calculations

### 3. Data Visualization
- Income quartile spending patterns
- Product category distributions
- Campaign response rates
- Customer segmentation plots

### 4. Machine Learning Models

#### Customer Segmentation (K-Means Clustering)
- Unsupervised learning to identify customer groups
- Optimal cluster selection using elbow method
- Customer profiling and insights

#### Campaign Response Prediction (Random Forest)
- Binary classification for campaign effectiveness
- Feature importance analysis
- Model evaluation metrics

#### Customer Lifetime Value (Neural Network)
- Deep learning model for spending prediction
- TensorFlow implementation
- Performance optimization

#### Recommendation System
- Collaborative filtering approach
- Cosine similarity for customer matching
- Personalized product recommendations

## 📊 Key Metrics & Results

### Statistical Tests
- **Wine Spending ANOVA**: F-statistic = 245.67, p < 0.001
- **Meat Products ANOVA**: F-statistic = 198.43, p < 0.001
- **Effect Size**: Large practical significance

### Machine Learning Performance
- **Campaign Prediction Accuracy**: 85%+
- **Customer Segmentation**: 4 distinct clusters identified
- **CLV Prediction MAE**: < 10% of average spending

## 💡 Business Insights

1. **Target High-Income Customers**: Focus premium product campaigns on Q4 income segment
2. **Personalized Recommendations**: Implement ML-driven product suggestions
3. **Campaign Optimization**: Use predictive models to identify likely responders
4. **Customer Retention**: Develop strategies for each identified customer segment

## 🔧 Technical Implementation

### Data Processing
- Handled missing values using mean imputation
- Created derived features (total purchases, spending ratios)
- Standardized features for ML models

### Model Validation
- Train-test splits (80/20)
- Cross-validation for robust results
- Feature importance analysis
- Hyperparameter tuning

## 📝 Future Enhancements

- [ ] Time series analysis for seasonal patterns
- [ ] Advanced deep learning architectures
- [ ] Real-time recommendation API
- [ ] A/B testing framework
- [ ] Customer churn prediction
- [ ] Interactive dashboard development

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sagar Roy**
- LinkedIn: [Your LinkedIn Profile](www.linkedin.com/in/sagar-roy-887080309)
- Email: sagarroy54321@gmail.com

## 🙏 Acknowledgments

- Dataset source: [Marketing Campaign Dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)
- Inspiration from data science community
- Statistical methods from academic research

---

*This project demonstrates proficiency in data analysis, statistical testing, machine learning, and business insight generation for marketing analytics.*
