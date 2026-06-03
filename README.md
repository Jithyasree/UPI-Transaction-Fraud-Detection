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

## 🛠 Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- HTML
- CSS

## 📊 Data Analysis & Visualization

- Transaction Amount Distribution
- Transaction Hour Analysis
- Class Distribution Analysis
- Correlation Heatmap
- Feature Importance Analysis
- SMOTE Class Balancing Visualization
- PCA Variance Analysis
- Model Comparison Charts
- Confusion Matrices
- Learning Curves

## 🏆 Top Performing Models

| Rank | Model | Accuracy |
|--------|--------|----------|
| 1 | Bagging | 99.7% |
| 2 | Extra Trees | 99.65% |
| 3 | Support Vector Machine | 99.65% |
| 4 | Gradient Boosting | 99.6% |
| 5 | K-Nearest Neighbors | 99.5% |

## 📂 Project Structure

```text
UPI-Transaction-Fraud-Detection/
│
├── README.md
├── upi_transactions.csv
├── results.json
│
├── Models/
│   ├── random_forest.pkl
│   ├── support_vector_machine.pkl
│   ├── gradient_boosting_xgbequiv.pkl
│   └── other trained models
│
├── Screenshots/
│   ├── home_page.png
│   ├── prediction_result.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── eda_correlation.png
│   └── smote_balance.png
│
└── Application Files
```
## 🎯 Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Class Balancing using SMOTE
6. PCA Implementation
7. Model Training
8. Model Evaluation
9. Web Application Deployment
10. Fraud Prediction

## ## ⚙️ Project Execution

The project was executed using the Anaconda environment.

- Uploaded all project files into the Anaconda working environment.
- Installed and imported the required libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Flask.
- Opened the Anaconda Prompt and navigated to the environment and click on project file to open terminal.
- Executed the application using Python commands.
```bash
(upi) C:\Users\jithy>e:

(upi) E:\>cd E:\Final YR PROJECT\upi_fraud\upi_fraud

(upi) E:\Final YR PROJECT\upi_fraud\upi_fraud>python app.py
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 220-851-747
127.0.0.1 - - [03/Jun/2026 15:16:57] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [03/Jun/2026 15:16:57] "GET /favicon.ico HTTP/1.1" 404 -  
```
- Successfully launched the UPI Fraud Detection web interface on the local server.
- Tested multiple transaction inputs and verified fraud prediction results through the web application.

## 📈 Results

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
Dataset not included due to file size limitations.

Synthetic UPI transaction dataset (10,000 records, 8% fraud) with features:
- Transaction amount, frequency, hour
- PIN attempts, device/location changes
- UPI ID age, recipient novelty
- State, merchant category, DOB year
