"""
SecureUPI Flask Web App
"""
import os, json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE   = os.path.dirname(__file__)
MODELS = os.path.join(BASE, 'models')
STATIC = os.path.join(BASE, 'static')

# Load artifacts
scaler    = joblib.load(f"{MODELS}/scaler.pkl")
pca       = joblib.load(f"{MODELS}/pca.pkl")
le_state  = joblib.load(f"{MODELS}/le_state.pkl")
le_cat    = joblib.load(f"{MODELS}/le_cat.pkl")
feat_cols = joblib.load(f"{MODELS}/feature_cols.pkl")
with open(f"{MODELS}/results.json") as f:
    model_results = json.load(f)

# Model name → file mapping
MODEL_FILES = {
    'Gradient Boosting (XGB-equiv)': 'gradient_boosting_xgbequiv',
    'Random Forest':                 'random_forest',
    'Extra Trees':                   'extra_trees',
    'AdaBoost':                      'adaboost',
    'Bagging':                       'bagging',
    'Decision Tree':                 'decision_tree',
    'K-Nearest Neighbors':           'knearest_neighbors',
    'Support Vector Machine':        'support_vector_machine',
    'Logistic Regression':           'logistic_regression',
    'Naive Bayes':                   'naive_bayes',
}

loaded_models = {}
for name, fname in MODEL_FILES.items():
    path = f"{MODELS}/{fname}.pkl"
    if os.path.exists(path):
        loaded_models[name] = joblib.load(path)

STATES = ['Maharashtra','Delhi','Karnataka','Tamil Nadu','Gujarat',
          'Uttar Pradesh','West Bengal','Rajasthan','Telangana','Kerala']
CATEGORIES = ['Grocery','Electronics','Food','Travel','Clothing',
              'Utilities','Healthcare','Entertainment','Education','Fuel']

def encode_input(form):
    try:
        state   = le_state.transform([form['state']])[0]
    except:
        state = 0
    try:
        cat = le_cat.transform([form['merchant_category']])[0]
    except:
        cat = 0

    raw = {
        'transaction_amount':     float(form['transaction_amount']),
        'transaction_frequency':  int(form['transaction_frequency']),
        'transaction_hour':       int(form['transaction_hour']),
        'pin_attempts':           int(form['pin_attempts']),
        'device_change_count':    int(form['device_change_count']),
        'new_recipient':          int(form['new_recipient']),
        'location_change':        int(form['location_change']),
        'upi_id_age_days':        int(form['upi_id_age_days']),
        'avg_transaction_amount': float(form['avg_transaction_amount']),
        'dob_year':               int(form['dob_year']),
        'state_enc':              state,
        'cat_enc':                cat,
    }
    X = np.array([[raw[c] for c in feat_cols]])
    X_sc  = scaler.transform(X)
    X_pca = pca.transform(X_sc)
    return X_pca

@app.route('/')
def index():
    best = max(model_results, key=lambda x: model_results[x]['accuracy'])
    return render_template('index.html',
                           states=STATES,
                           categories=CATEGORIES,
                           model_results=model_results,
                           best_model=best)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        X = encode_input(request.form)
        selected_model = request.form.get('selected_model', 'Gradient Boosting (XGB-equiv)')
        model = loaded_models.get(selected_model)
        if model is None:
            return jsonify({'error': 'Model not found'}), 400
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        fraud_prob = float(prob[1]) if prob is not None else (1.0 if pred else 0.0)
        return jsonify({
            'prediction': int(pred),
            'fraud_probability': round(fraud_prob * 100, 1),
            'legitimate_probability': round((1-fraud_prob)*100, 1),
            'model_used': selected_model,
            'model_accuracy': model_results[selected_model]['accuracy'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/eda')
def eda():
    plots = [
        ('eda_class_dist.png',  'Class Distribution'),
        ('eda_amount_dist.png', 'Amount Distribution'),
        ('eda_correlation.png', 'Correlation Heatmap'),
        ('eda_by_hour.png',     'Transactions by Hour'),
        ('eda_by_category.png', 'Fraud by Category'),
        ('eda_boxplot.png',     'Amount Box Plot'),
    ]
    return render_template('eda.html', plots=plots)

@app.route('/preprocessing')
def preprocessing():
    plots = [
        ('smote_balance.png', 'SMOTE Balancing'),
        ('pca_variance.png',  'PCA Explained Variance'),
    ]
    return render_template('preprocessing.html', plots=plots)

@app.route('/models')
def models_page():
    # Sort by accuracy
    sorted_results = sorted(model_results.items(), key=lambda x: -x[1]['accuracy'])
    cm_plots = {}
    lc_plots = {}
    for name in MODEL_FILES:
        fn = name.lower().replace(' ','_').replace('(','').replace(')','').replace('-','')
        cm_plots[name] = f"plots/cm_{fn}.png"
        lc_plots[name] = f"plots/lc_{fn}.png"
    return render_template('models.html',
                           sorted_results=sorted_results,
                           cm_plots=cm_plots,
                           lc_plots=lc_plots)

@app.route('/comparison')
def comparison():
    plots = [
        ('plots/model_comparison.png',  'Model Performance Comparison'),
        ('plots/overfit_check.png',     'Overfitting / Underfitting Check'),
        ('plots/feature_importance.png','Feature Importance'),
    ]
    sorted_results = sorted(model_results.items(), key=lambda x: -x[1]['accuracy'])
    return render_template('comparison.html', plots=plots, sorted_results=sorted_results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
