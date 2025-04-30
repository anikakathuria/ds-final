# YouTube Engagement Prediction Dashboard

This interactive dashboard showcases the complete data science pipeline for predicting YouTube engagement on climate-related videos, including:

- Data preprocessing and cleaning
- Exploratory data analysis with visualizations
- Feature engineering techniques
- Model selection, comparison, and improvement strategies

## Setup and Installation

1. Make sure you have Python installed (3.8+ recommended)
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

From the project root directory:

```bash
streamlit run app/app.py
```

## Dashboard Features

The dashboard is organized into sections:

1. **Home**: Project overview and navigation
2. **Data Preprocessing**: Data cleaning steps and transformations
3. **Exploratory Data Analysis**: Key insights and visualizations
4. **Feature Engineering**: Feature creation techniques and importance
5. **Model Selection & Improvement**: Model comparison, tuning, and interpretability

## Data Files

Make sure these CSV files are in the project root directory:

- `climate_youtube_with_channels.csv`: Raw dataset
- `cleaned_data.csv`: Preprocessed dataset
- `cleaned_FE_data.csv`: Dataset with engineered features
- `model_comparison_summary.csv`: Model performance metrics
