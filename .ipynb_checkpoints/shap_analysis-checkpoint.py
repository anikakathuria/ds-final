# shap_analysis.py
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

df = pd.read_csv('cleaned_FE_data.csv')

median = df['like_view_ratio'].median()
df['engagement_label'] = (df['like_view_ratio'] > median).astype(int)

drop_cols = [
    'video_id', 'title', 'description', 'tags', 'description_clean', 'tags_clean', 'text',
    'engagement_label',
    'view_count', 'like_count', 'comment_count',
    'subscriber_count', 'video_count', 'view_count_total'
]

X = df.drop(columns=drop_cols, errors='ignore')
X = X.select_dtypes(include='number').fillna(0)
y = df['engagement_label']

# Ex. XGBoost model
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    colsample_bytree=0.6,
    learning_rate=0.01,
    max_depth=2,
    n_estimators=100,
    subsample=0.8,
    random_state=42
)

model.fit(X, y)

# SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

# SHAP summary plot
shap.summary_plot(shap_values, X, show=True)

# Select a value: ex.title_sentiment
shap.dependence_plot("title_sentiment", shap_values.values, X, show=True)


#### Model Interpretability with SHAP:
#To interpret our XGBoost model, we used SHAP to visualize the contribution of each feature to the prediction of video engagement.
#The SHAP summary plot revealed that `like_view_ratio`, `comment_view_ratio`, and `duration_seconds` were among the most influential features. Additionally, content-based variables such as `title_sentiment`, `tag_count`, and binary indicators like `is_clickbait` and `has_number` also contributed significantly.
#The dependence plot of `title_sentiment` further showed that more positive titles tend to slightly increase the probability of being classified as high engagement. This supports the hypothesis that emotional tone and textual structure of the title play a role in audience interaction.
#Overall, the SHAP analysis validates the informativeness of our engineered features and confirms that the model does not rely solely on viewership statistics.

#Although Logistic Regression was selected as our final model for its simplicity and generalizability, we used XGBoost in conjunction with SHAP to better understand the impact of different features.
#XGBoost is a tree-based model that supports detailed feature attribution via SHAP. This allows us to visualize and confirm which features contributed most to the model's predictions, offering greater interpretability even if it's not the final chosen model.
#By doing so, we validate the strength of our feature engineering and demonstrate that content-based features like title sentiment and clickbait patterns are indeed informative.
