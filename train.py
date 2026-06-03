"""
SecureUPI - UPI Fraud Detection Training
10 ML models | Manual SMOTE | PCA | EDA | Overfit Check
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = "upi_fraud"
STATIC = f"{BASE}/static/plots"
MODELS = f"{BASE}/models"
for d in [STATIC, MODELS]: os.makedirs(d, exist_ok=True)

# ── 1. GENERATE DATASET ──────────────────────────────────────────────────────
def generate_dataset(n=10000):
    rng = np.random.default_rng(42)
    states = ['Maharashtra','Delhi','Karnataka','Tamil Nadu','Gujarat',
              'Uttar Pradesh','West Bengal','Rajasthan','Telangana','Kerala']
    cats   = ['Grocery','Electronics','Food','Travel','Clothing',
              'Utilities','Healthcare','Entertainment','Education','Fuel']
    nf = int(n * 0.08)
    nl = n - nf

    def make(size, fraud):
        amounts = np.clip(rng.exponential(5000 if fraud else 1200, size), 10, 100000)
        return pd.DataFrame({
            'transaction_amount':   amounts.round(2),
            'transaction_frequency':rng.integers(1, 30 if fraud else 10, size),
            'transaction_hour':     (rng.choice(np.r_[np.arange(0,6), np.arange(22,24)], size)
                                     if fraud else rng.integers(6,22,size)),
            'pin_attempts':         rng.integers(1, 5 if fraud else 2, size),
            'device_change_count':  rng.integers(0, 3 if fraud else 2, size),
            'new_recipient':        rng.integers(0, 2 if fraud else 1, size),
            'location_change':      rng.integers(0, 3 if fraud else 2, size),
            'upi_id_age_days':      rng.integers(1, 60 if fraud else 730, size),
            'avg_transaction_amount':np.clip(rng.exponential(4000 if fraud else 1000, size),10,100000).round(2),
            'state':                rng.choice(states, size),
            'merchant_category':    rng.choice(cats, size),
            'dob_year':             rng.integers(1970, 2000, size),
            'is_fraud':             np.full(size, int(fraud)),
        })
    df = pd.concat([make(nf,True), make(nl,False)], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── 2. MANUAL SMOTE ──────────────────────────────────────────────────────────
def manual_smote(X, y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    minority = X[y == 1]
    majority = X[y == 0]
    n_gen = len(majority) - len(minority)
    synthetic = []
    for _ in range(n_gen):
        idx = rng.integers(0, len(minority))
        sample = minority[idx]
        diffs = minority - sample
        dists = np.sqrt((diffs**2).sum(axis=1))
        dists[idx] = np.inf
        nn_idx = np.argsort(dists)[:k]
        chosen = minority[rng.choice(nn_idx)]
        gap = rng.random()
        synthetic.append(sample + gap * (chosen - sample))
    Xs = np.vstack([X, np.array(synthetic)])
    ys = np.concatenate([y, np.ones(n_gen, dtype=int)])
    shuffle = rng.permutation(len(ys))
    return Xs[shuffle], ys[shuffle]

# ── 3. EDA ───────────────────────────────────────────────────────────────────
def run_eda(df):
    print(f"\nEDA: shape={df.shape}, missing={df.isnull().sum().sum()}, fraud={df.is_fraud.sum()}")

    # Class distribution
    fig, ax = plt.subplots(figsize=(5,4))
    counts = df['is_fraud'].value_counts().sort_index()
    bars = ax.bar(['Legitimate','Fraud'], counts.values, color=['#00C851','#FF4444'], edgecolor='white', lw=1.5)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Count')
    for b,v in zip(bars,counts.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+40, f'{v:,}', ha='center', fontweight='bold', fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{STATIC}/eda_class_dist.png", dpi=120); plt.close()

    # Amount by class
    fig, axes = plt.subplots(1,2, figsize=(12,4))
    for i, (lbl, c, t) in enumerate([(0,'#2196F3','Legitimate'),(1,'#F44336','Fraud')]):
        d = df[df.is_fraud==lbl]['transaction_amount']
        axes[i].hist(d, bins=40, color=c, alpha=0.85, edgecolor='white')
        axes[i].set_title(f'{t} — Amount Distribution', fontweight='bold')
        axes[i].set_xlabel('Amount (₹)'); axes[i].set_ylabel('Count')
        axes[i].spines[['top','right']].set_visible(False)
    plt.suptitle('Transaction Amount by Class', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(f"{STATIC}/eda_amount_dist.png", dpi=120); plt.close()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10,8))
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                ax=ax, linewidths=0.5, annot_kws={'size':8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout(); plt.savefig(f"{STATIC}/eda_correlation.png", dpi=120); plt.close()

    # Fraud by hour
    fig, ax = plt.subplots(figsize=(12,4))
    fh = df.groupby(['transaction_hour','is_fraud']).size().unstack(fill_value=0)
    fh.plot(kind='bar', ax=ax, color=['#00C851','#FF4444'], alpha=0.85, edgecolor='white')
    ax.set_title('Transactions by Hour of Day', fontsize=13, fontweight='bold')
    ax.set_xlabel('Hour'); ax.set_ylabel('Count')
    ax.legend(['Legitimate','Fraud']); ax.spines[['top','right']].set_visible(False)
    plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{STATIC}/eda_by_hour.png", dpi=120); plt.close()

    # Fraud by category
    fig, ax = plt.subplots(figsize=(10,5))
    df[df.is_fraud==1]['merchant_category'].value_counts().plot(
        kind='bar', ax=ax, color='#FF4444', alpha=0.85, edgecolor='white')
    ax.set_title('Fraud Count by Merchant Category', fontsize=13, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel('Fraud Count')
    ax.spines[['top','right']].set_visible(False)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(); plt.savefig(f"{STATIC}/eda_by_category.png", dpi=120); plt.close()

    # Boxplot amount
    fig, ax = plt.subplots(figsize=(7,5))
    data = [df[df.is_fraud==0]['transaction_amount'].values,
            df[df.is_fraud==1]['transaction_amount'].values]
    bp = ax.boxplot(data, labels=['Legitimate','Fraud'], patch_artist=True,
                    medianprops=dict(color='white', lw=2))
    colors = ['#2196F3','#F44336']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title('Transaction Amount — Box Plot', fontweight='bold', fontsize=13)
    ax.set_ylabel('Amount (₹)'); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{STATIC}/eda_boxplot.png", dpi=120); plt.close()

    print("EDA plots saved.")

# ── 4. PREPROCESSING ─────────────────────────────────────────────────────────
def preprocess(df):
    print("\nPreprocessing...")
    le_s = LabelEncoder(); le_c = LabelEncoder()
    df['state_enc'] = le_s.fit_transform(df['state'])
    df['cat_enc']   = le_c.fit_transform(df['merchant_category'])
    feat_cols = ['transaction_amount','transaction_frequency','transaction_hour',
                 'pin_attempts','device_change_count','new_recipient','location_change',
                 'upi_id_age_days','avg_transaction_amount','dob_year','state_enc','cat_enc']
    X = df[feat_cols].values
    y = df['is_fraud'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # SMOTE
    print(f"  Before SMOTE: {np.bincount(y_tr)}")
    X_tr_sm, y_tr_sm = manual_smote(X_tr_sc, y_tr)
    print(f"  After SMOTE:  {np.bincount(y_tr_sm)}")

    # SMOTE plot
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    for ax, yd, title in zip(axes, [y_tr, y_tr_sm], ['Before SMOTE','After SMOTE']):
        counts = np.bincount(yd)
        bars = ax.bar(['Legit','Fraud'], counts, color=['#00C851','#FF4444'], edgecolor='white')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel('Count'); ax.spines[['top','right']].set_visible(False)
        for b,v in zip(bars,counts):
            ax.text(b.get_x()+b.get_width()/2, v+30, f'{v:,}', ha='center', fontweight='bold')
    plt.suptitle('SMOTE Balancing Effect', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(f"{STATIC}/smote_balance.png", dpi=120); plt.close()

    # PCA
    pca = PCA(n_components=0.95, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_sm)
    X_te_pca = pca.transform(X_te_sc)
    print(f"  PCA components: {pca.n_components_}")

    # PCA variance plot
    fig, ax = plt.subplots(figsize=(8,4))
    cumvar = np.cumsum(pca.explained_variance_ratio_)*100
    ax.plot(range(1, len(cumvar)+1), cumvar, 'o-', color='#2196F3', lw=2.5, markersize=8)
    ax.axhline(95, color='#F44336', linestyle='--', lw=1.5, label='95% threshold')
    ax.fill_between(range(1, len(cumvar)+1), 0, cumvar, alpha=0.1, color='#2196F3')
    ax.set_title('PCA — Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Components'); ax.set_ylabel('Cumulative Variance (%)')
    ax.legend(); ax.grid(alpha=0.25); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{STATIC}/pca_variance.png", dpi=120); plt.close()

    joblib.dump(scaler,  f"{MODELS}/scaler.pkl")
    joblib.dump(pca,     f"{MODELS}/pca.pkl")
    joblib.dump(le_s,    f"{MODELS}/le_state.pkl")
    joblib.dump(le_c,    f"{MODELS}/le_cat.pkl")
    joblib.dump(feat_cols, f"{MODELS}/feature_cols.pkl")
    return X_tr_pca, X_te_pca, y_tr_sm, y_te

# ── 5. MODELS ─────────────────────────────────────────────────────────────────
def get_models():
    return {
        'Gradient Boosting (XGB-equiv)': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42),
        'Random Forest':   RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        'Extra Trees':     ExtraTreesClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        'AdaBoost':        AdaBoostClassifier(n_estimators=150, random_state=42),
        'Bagging':         BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Decision Tree':   DecisionTreeClassifier(max_depth=12, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42, C=1.0),
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Naive Bayes':     GaussianNB(),
    }

def eval_model(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    yp_tr = model.predict(X_tr)
    yprob = model.predict_proba(X_te)[:,1] if hasattr(model,'predict_proba') else None
    tr_acc = accuracy_score(y_tr, yp_tr)
    te_acc = accuracy_score(y_te, yp)
    gap = (tr_acc - te_acc)*100
    status = 'Overfitting' if gap>10 else ('Underfitting' if te_acc<0.70 else 'Good Fit')
    return {
        'train_accuracy': round(tr_acc*100,2),
        'accuracy':       round(te_acc*100,2),
        'precision':      round(precision_score(y_te,yp,zero_division=0)*100,2),
        'recall':         round(recall_score(y_te,yp,zero_division=0)*100,2),
        'f1':             round(f1_score(y_te,yp,zero_division=0)*100,2),
        'roc_auc':        round(roc_auc_score(y_te,yprob)*100,2) if yprob is not None else 0,
        'overfit_gap':    round(gap,2),
        'fit_status':     status,
    }, model

def plot_cm(name, y_te, y_pred):
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'],
                linewidths=1, linecolor='white', annot_kws={'size':13})
    ax.set_title(f'Confusion Matrix\n{name}', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    fn = name.lower().replace(' ','_').replace('(','').replace(')','').replace('-','')
    plt.savefig(f"{STATIC}/cm_{fn}.png", dpi=120); plt.close()

def plot_lc(model, name, X_tr, y_tr):
    try:
        sizes, tr_s, val_s = learning_curve(
            model, X_tr, y_tr, cv=3, train_sizes=np.linspace(0.2,1.0,5),
            scoring='accuracy', n_jobs=-1)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(sizes, tr_s.mean(1), 'o-', color='#2196F3', lw=2, label='Train')
        ax.plot(sizes, val_s.mean(1), 'o-', color='#F44336', lw=2, label='Validation')
        ax.fill_between(sizes, tr_s.mean(1)-tr_s.std(1), tr_s.mean(1)+tr_s.std(1), alpha=0.1, color='#2196F3')
        ax.fill_between(sizes, val_s.mean(1)-val_s.std(1), val_s.mean(1)+val_s.std(1), alpha=0.1, color='#F44336')
        ax.set_title(f'Learning Curve — {name}', fontweight='bold')
        ax.set_xlabel('Training Size'); ax.set_ylabel('Accuracy')
        ax.legend(); ax.grid(alpha=0.25); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        fn = name.lower().replace(' ','_').replace('(','').replace(')','').replace('-','')
        plt.savefig(f"{STATIC}/lc_{fn}.png", dpi=120); plt.close()
    except Exception as e:
        print(f"  LC failed {name}: {e}")

def train_all(X_tr, X_te, y_tr, y_te):
    models = get_models()
    results = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        res, trained = eval_model(model, X_tr, X_te, y_tr, y_te)
        results[name] = res
        plot_cm(name, y_te, trained.predict(X_te))
        plot_lc(trained, name, X_tr, y_tr)
        fn = name.lower().replace(' ','_').replace('(','').replace(')','').replace('-','')
        joblib.dump(trained, f"{MODELS}/{fn}.pkl")
        print(f"    Acc={res['accuracy']}% F1={res['f1']}% {res['fit_status']}")
    return results

def plot_comparisons(results):
    names = list(results.keys())
    short_names = [n.replace(' (XGB-equiv)','').replace(' ','_') for n in names]
    accs  = [results[n]['accuracy'] for n in names]
    f1s   = [results[n]['f1'] for n in names]
    rocs  = [results[n]['roc_auc'] for n in names]

    # 3-panel comparison
    fig, axes = plt.subplots(1,3, figsize=(17,6))
    for ax, vals, title, color in zip(axes,
                                       [accs,f1s,rocs],
                                       ['Accuracy (%)','F1 Score (%)','ROC-AUC (%)'],
                                       ['#2196F3','#00C851','#FF9800']):
        bars = ax.barh(names, vals, color=color, alpha=0.85, edgecolor='white', height=0.6)
        ax.set_xlabel(title, fontsize=11); ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlim(0, 110); ax.axvline(90, color='red', linestyle='--', alpha=0.4, lw=1)
        for b,v in zip(bars,vals):
            ax.text(v+0.3, b.get_y()+b.get_height()/2, f'{v}%', va='center', fontsize=8, fontweight='bold')
        ax.grid(axis='x', alpha=0.25); ax.spines[['top','right']].set_visible(False)
    plt.suptitle('All Models — Performance Comparison', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(); plt.savefig(f"{STATIC}/model_comparison.png", dpi=130, bbox_inches='tight'); plt.close()

    # Overfit check
    fig, ax = plt.subplots(figsize=(13,5))
    x = np.arange(len(names))
    tr_accs = [results[n]['train_accuracy'] for n in names]
    ax.bar(x-0.2, tr_accs, 0.38, label='Train Accuracy', color='#2196F3', alpha=0.85, edgecolor='white')
    ax.bar(x+0.2, accs,    0.38, label='Test Accuracy',  color='#00C851', alpha=0.85, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Train vs Test Accuracy — Overfitting / Underfitting Check', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10); ax.set_ylim(0, 115); ax.grid(axis='y', alpha=0.25)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{STATIC}/overfit_check.png", dpi=120); plt.close()

    # Feature importance (GBT)
    try:
        gbt = joblib.load(f"{MODELS}/gradient_boosting_xgbequiv.pkl")
        fi = gbt.feature_importances_
        feat_labels = ['Amount','Frequency','Hour','PIN Attempts','Device Change',
                       'New Recipient','Loc Change','UPI Age','Avg Amount','DOB Year','State','Merch Cat']
        fi = fi[:len(feat_labels)]
        feat_labels = feat_labels[:len(fi)]
        idx = np.argsort(fi)
        fig, ax = plt.subplots(figsize=(9,5))
        colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi)))
        ax.barh([feat_labels[i] for i in idx], fi[idx], color=[colors_fi[i] for i in idx], edgecolor='white')
        ax.set_title('Gradient Boosting — Feature Importance', fontweight='bold', fontsize=13)
        ax.set_xlabel('Importance Score'); ax.grid(axis='x', alpha=0.25)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); plt.savefig(f"{STATIC}/feature_importance.png", dpi=120); plt.close()
    except Exception as e:
        print(f"FI plot error: {e}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== SecureUPI Training Pipeline ===")
    df = generate_dataset(10000)
    df.to_csv(f"{MODELS}/upi_transactions.csv", index=False)
    print(f"Dataset: {df.shape} | Fraud: {df.is_fraud.sum()}")

    run_eda(df)
    X_tr, X_te, y_tr, y_te = preprocess(df)

    print("\nTraining 10 models...")
    results = train_all(X_tr, X_te, y_tr, y_te)
    plot_comparisons(results)

    with open(f"{MODELS}/results.json",'w') as f:
        json.dump(results, f, indent=2)

    print("\n===== FINAL RESULTS =====")
    for n,r in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"{n:38s} Acc={r['accuracy']}%  F1={r['f1']}%  {r['fit_status']}")
    print("\nAll done!")
