# SecureUPI — ML-Driven UPI Fraud Detection System

Based on the research paper:
> "Secure UPI: Machine Learning-Driven Fraud Detection System for UPI Transactions"
> 2024 2nd International Conference on Disruptive Technologies (ICDT)

## Features
- **10 ML Models**: Gradient Boosting, Random Forest, Extra Trees, SVM, KNN, AdaBoost, Bagging, Decision Tree, Logistic Regression, Naive Bayes
- **SMOTE**: Manual synthetic minority oversampling to balance the 8% fraud rate
- **PCA**: Dimensionality reduction retaining 95% variance (12 → 11 components)
- **EDA**: 6 exploratory plots (class distribution, amount, correlation, time, category, boxplot)
- **Overfitting Check**: Train vs test accuracy gap + learning curves for all models
- **Flask UI**: 5-page web app with bright, modern design

## Setup

```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn joblib

# Train models (run once)
python train.py

# Start web server
python app.py
```

Then open: http://localhost:5000

## Pages
| Page | Description |
|------|-------------|
| `/` | Fraud prediction form + leaderboard |
| `/eda` | Exploratory Data Analysis plots |
| `/preprocessing` | SMOTE + PCA pipeline visualization |
| `/models` | All 10 models with confusion matrices & learning curves |
| `/comparison` | Side-by-side comparison + feature importance |

## Results

| Rank | Model | Accuracy | F1 | Status |
|------|-------|----------|----|--------|
| 1 | Bagging | 99.7% | 98.14% | Good Fit |
| 2 | Extra Trees | 99.65% | 97.79% | Good Fit |
| 3 | SVM | 99.65% | 97.82% | Good Fit |
| 4 | Gradient Boosting | 99.6% | 97.48% | Good Fit |
| 5 | KNN | 99.5% | 96.82% | Good Fit |
| 6 | Random Forest | 99.35% | 96.02% | Good Fit |
| 7 | AdaBoost | 99.35% | 96.0% | Good Fit |
| 8 | Decision Tree | 99.35% | 96.0% | Good Fit |
| 9 | Logistic Regression | 99.2% | 95.09% | Good Fit |
| 10 | Naive Bayes | 98.0% | 88.7% | Good Fit |

## Dataset
Synthetic UPI transaction dataset (10,000 records, 8% fraud) with features:
- Transaction amount, frequency, hour
- PIN attempts, device/location changes
- UPI ID age, recipient novelty
- State, merchant category, DOB year
