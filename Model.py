
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc


df = pd.read_csv('cleaned_FE_data.csv')

# Create Engagement Label
# make level by like_view_ratio 
df['like_view_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['like_view_ratio'].fillna(0, inplace=True)

median_like_view = df['like_view_ratio'].median()
df['engagement_label'] = (df['like_view_ratio'] > median_like_view).astype(int)

print("Engagement label distribution:")
print(df['engagement_label'].value_counts())

# Features selecting

drop_cols = [
    'video_id', 'title', 'description', 'tags', 'description_clean', 'tags_clean', 'text',
    'engagement_label',  
    'view_count', 'like_count', 'comment_count',  
    'subscriber_count', 'video_count', 'view_count_total'  
]

X = df.drop(columns=drop_cols, errors='ignore')
X = X.select_dtypes(include=[np.number])
X = X.fillna(0)

y = df['engagement_label']


# Train & Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 3 Models，Tuning by GridSearchCV

# Logistic Regression + GridSearch
logreg_params = {
    'C': [0.01, 0.1, 1, 10, 100]
}
logreg = LogisticRegression(max_iter=1000, random_state=42)
grid_logreg = GridSearchCV(logreg, logreg_params, cv=5, scoring='f1')
grid_logreg.fit(X_train, y_train)

print("\nBest Logistic Regression params:", grid_logreg.best_params_)

# Random Forest + GridSearch
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, rf_params, cv=5, scoring='f1')
grid_rf.fit(X_train, y_train)

print("\nBest Random Forest params:", grid_rf.best_params_)

# XGBoost + GridSearch
xgb_params = {
    'n_estimators': [50, 100],            
    'max_depth': [2, 3],                  
    'learning_rate': [0.01, 0.05],        
    'subsample': [0.6, 0.8],             
    'colsample_bytree': [0.6, 0.8]        
}

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_xgb = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='f1')
grid_xgb.fit(X_train, y_train)

print("\nBest XGBoost params:", grid_xgb.best_params_)

# Best model on Test

models = {
    'Logistic Regression': grid_logreg.best_estimator_,
    'Random Forest': grid_rf.best_estimator_,
    'XGBoost': grid_xgb.best_estimator_
}

for model_name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n=== {model_name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ROC
plt.figure(figsize=(8,6))

for model_name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

