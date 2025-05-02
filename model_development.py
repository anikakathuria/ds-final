import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("cleaned_FE_data.csv")

# Fill missing text cols
for col in ['description_clean', 'tags_clean', 'text']:
    df[col] = df[col].fillna("")

# Generate text embeddings
text_model = SentenceTransformer('all-MiniLM-L6-v2')
X_text = text_model.encode(df['text'].tolist(), show_progress_bar=True)
X_desc = text_model.encode(df['description_clean'].tolist(), show_progress_bar=True)
X_tags = text_model.encode(df['tags_clean'].tolist(), show_progress_bar=True)

# Structured features
structured_features = [
    'title_sentiment', 'is_clickbait', 'has_number', 'has_punctuation',
    'publish_hour', 'publish_weekday', 'is_weekday', 'publish_month', 'is_holiday',
    'cluster', 'tag_count', 'duration_seconds'
]
X_struct = df[structured_features].fillna(0).values

# Create label
threshold = df['like_view_ratio'].mean()
df['label'] = (df['like_view_ratio'] > threshold).astype(int)
y = df['label'].values

# Combine all features
scaler = StandardScaler()
X_struct_scaled = scaler.fit_transform(X_struct)
X_all = np.hstack([X_text, X_struct_scaled])  # optionally add X_desc, X_tags too

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#  Define models and grids 
param_grids = {
    "RandomForest": {
        "n_estimators": [100, 300],
        "max_depth": [5, 10],
        "min_samples_leaf": [1, 3]
    },
    "XGBoost": {
        "n_estimators": [100, 300],
        "max_depth": [5, 10],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8]
    },
    "LogisticRegression_L2": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ['liblinear'],
        "penalty": ['l2'],
        "max_iter": [1000]
    }
}

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "LogisticRegression_L2": LogisticRegression()
}

# GridSearchCV + Evaluation
results = []

for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], scoring='f1', cv=5,
                               verbose=0, n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_model = grid_search.best_estimator_

    # Predict
    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train_resampled)

    # Get CV std
    std_f1 = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

    results.append({
        "Model": name,
        "Best Params": grid_search.best_params_,
        "Best F1 Score (CV)": grid_search.best_score_,
        "CV Std": std_f1,
        "Train F1": f1_score(y_train_resampled, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test Precision": precision_score(y_test, y_pred),
        "Test Recall": recall_score(y_test, y_pred),
        "Test F1": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison Summary:")
print(results_df[["Model", "Best F1 Score (CV)", "CV Std", "Train F1", "Test Accuracy", "Test Precision", "Test Recall", "Test F1"]].to_string(index=False))
print("\nBest Parameters:")
print(results_df[["Model", "Best Params"]].to_string(index=False))



#ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], scoring='f1', cv=5, n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_model = grid_search.best_estimator_
    
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of Models")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



#SHAP
import shap

final_model = XGBClassifier(**results_df[results_df["Model"]=="XGBoost"]["Best Params"].values[0], eval_metric='logloss')
final_model.fit(X_train_resampled, y_train_resampled)

explainer = shap.Explainer(final_model)
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values, max_display=10)



# train best XGBoost again
xgb_best_params = results_df[results_df["Model"] == "XGBoost"]["Best Params"].values[0]
xgb_model = XGBClassifier(**xgb_best_params, eval_metric='logloss')
xgb_model.fit(X_train_resampled, y_train_resampled)

# only explain structured features 
explainer = shap.Explainer(xgb_model, X_train_resampled)
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values, max_display=10)


#t-SNE
from sklearn.manifold import TSNE
import seaborn as sns

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_text)  

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=df['label'], palette='Set2', alpha=0.7)
plt.title("t-SNE Visualization of Sentence Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()


# StackingClassifier model
from sklearn.ensemble import StackingClassifier

rf_best_params = results_df[results_df["Model"] == "RandomForest"]["Best Params"].values[0]
lr_best_params = results_df[results_df["Model"] == "LogisticRegression_L2"]["Best Params"].values[0]
xgb_best_params = results_df[results_df["Model"] == "XGBoost"]["Best Params"].values[0]

stacking_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(**rf_best_params, random_state=42)),
        ('lr', LogisticRegression(**lr_best_params))
    ],
    final_estimator=XGBClassifier(**xgb_best_params, eval_metric='logloss', random_state=42)
)

stacking_model.fit(X_train_resampled, y_train_resampled)

y_pred_stack = stacking_model.predict(X_test)
print("Stacking Model F1:", f1_score(y_test, y_pred_stack))
