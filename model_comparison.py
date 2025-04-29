
import pandas as pd

def generate_model_comparison_table():
    # Define model performance data
    model_comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.88, 1.00, 1.00],
        'F1 Score': [0.88, 1.00, 1.00],
        'ROC AUC': [0.95, 1.00, 1.00],
        'Precision (Class 1)': [0.86, 1.00, 1.00],
        'Recall (Class 1)': [0.91, 1.00, 1.00],
        'Best Params': [
            "{'C': 1}",
            "{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}",
            "{'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 100, 'subsample': 0.8}"
        ],
        'Notes': [
            'Balanced, interpretable, selected model',
            'Overfitting (Perfect on test set)',
            'Overfitting despite regularization'
        ]
    })

    # Save as CSV
    output_path = 'model_comparison_summary.csv'
    model_comparison.to_csv(output_path, index=False)
    print(f"Model comparison summary saved to: {output_path}")

    print(model_comparison)

if __name__ == '__main__':
    generate_model_comparison_table()
