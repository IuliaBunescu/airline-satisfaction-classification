# Airline Passenger Satisfaction Classification

This project analyzes the Airline Passenger Satisfaction dataset and builds machine learning models to predict whether a passenger is **satisfied** or **neutral/dissatisfied** based on demographic information, travel details, service ratings, and delay-related variables.

The project was completed as the final project for **CSE811 / STT811 Applied Statistical Modeling for Data Scientists**. The main goal was to compare baseline prediction, individual classification models, and ensemble learning approaches using a reproducible modeling workflow.

## Project Overview

Passenger satisfaction is treated as a binary classification problem. The target variable is:

- `satisfied`
- `neutral or dissatisfied`

The project compares multiple modeling approaches:

- Majority-class baseline
- Logistic Regression
- Support Vector Machine with RBF kernel
- XGBoost
- Soft Voting ensemble
- Stacking ensemble
- Bagging with XGBoost

The final comparison is based mainly on **macro average F1-score**, since this metric balances performance across both satisfaction classes.

## Repository Structure

**data/**
- `train.csv`: training dataset
- `test.csv`: test dataset

**img/**
- Contains figures and visualizations used in the final report, such as model performance plots, confusion matrices, validation curves, ROC curves, feature importance plots, and SHAP-related outputs.

**notebooks/**
- `811_XGboost.ipynb`: XGBoost modeling workflow, SHAP-based feature importance analysis, and the XGBoost-based bagging experiment.
- `EDA_lr.ipynb`: exploratory data analysis and initial Logistic Regression-related analysis.
- `Karina_lr.ipynb`: Logistic Regression modeling workflow, feature engineering, model evaluation, validation curves, grid search, and pipeline construction.
- `svm-julia.ipynb`: Support Vector Machine modeling workflow, feature selection, and SVM model tuning.
- `soft_voting.ipynb`: Soft Voting ensemble model combining predicted probabilities from Logistic Regression, SVM, and XGBoost.
- `stacking-julia.ipynb`: Stacking ensemble model using multiple base learners and Logistic Regression as the meta-model.
- `pipelinetest.ipynb`: testing code for loading and evaluating saved pipeline files.
- `utils.py`: helper functions used across notebooks to support preprocessing, model evaluation, and repeated analysis tasks.

**pipelines/**
- `logistic_regression_pipeline.pkl`
- `svm_pipeline.pkl`
- `xgb_pipeline.pkl`

The pipelines were saved to make the workflow more reusable and easier to share across team members. Instead of rerunning all preprocessing and training code from scratch, these files allow the trained models and their preprocessing steps to be loaded and evaluated consistently.

## Methods

### Data Preparation

The dataset was cleaned and transformed before modeling. Categorical variables were encoded, unnecessary columns were removed, and selected continuous variables were transformed into more useful model inputs.

Examples of preprocessing and feature engineering include:

- Encoding categorical variables such as gender, customer type, type of travel, and class
- Grouping age into age categories
- Grouping flight distance into distance categories
- Combining departure and arrival delay into a total delay feature
- Creating an average service-quality score from multiple service rating variables
- Dropping columns that were not used for modeling

### Baseline Model

A majority-class baseline model was used as the grounding benchmark. This model always predicts the most frequent class, which provides a lower-bound comparison point for the machine learning models.

The majority-class baseline achieved a macro average F1-score of **0.36170** on the test set.

### Logistic Regression

Logistic Regression was used as a simple and interpretable linear classification model. It served as a baseline machine learning model and allowed us to examine the relationship between features and passenger satisfaction.

The Logistic Regression workflow included:

- Feature engineering
- Standardization
- Regularization tuning
- Validation curve analysis
- Grid search with cross-validation
- Pipeline construction and saving

The final Logistic Regression model achieved a macro average F1-score of **0.8590**.

### Support Vector Machine

Support Vector Machine with an RBF kernel was used as an additional classification model to capture nonlinear decision boundaries. Feature selection was applied to reduce the feature space and focus on the most informative predictors.

The final SVM model used six selected features and achieved a macro average F1-score of **0.9284**.

### XGBoost

XGBoost was the strongest individual model in this project. It was used as a boosted tree-based classifier capable of capturing nonlinear relationships and feature interactions.

The XGBoost workflow included:

- Hyperparameter tuning
- Feature importance analysis
- SHAP-based feature selection
- Final model evaluation
- Bagging experiment using XGBoost as the base learner

The final tuned XGBoost model used the top 13 SHAP-ranked features and achieved a macro average F1-score of **0.9637**.

### Ensemble Models

The project also tested multiple ensemble learning approaches.

#### Soft Voting

Soft Voting combined the predicted probabilities from Logistic Regression, SVM, and XGBoost. This approach used the confidence scores from each model rather than only hard class labels.

The Soft Voting ensemble achieved a macro average F1-score of **0.9484**.

#### Stacking

Stacking combined Logistic Regression, SVM, and XGBoost as base learners, with Logistic Regression used as the meta-model. The stacking model learned how to combine predictions from the individual models.

The Stacking model achieved a macro average F1-score of **0.96261**.

#### Bagging with XGBoost

Bagging was tested using XGBoost as the base learner. Since XGBoost was the strongest individual model, this experiment tested whether training multiple XGBoost models on sampled versions of the data could improve or stabilize performance.

The Bagging with XGBoost model achieved a macro average F1-score of **0.962156**.

## Final Scores

| Dataset | Model | Best Params | Macro Avg F1-Score | Notes | Time (test-train) |
|---|---|---|---:|---|---|
| test | Majority Class (Baseline) | N/A | 0.36170 | Grounding baseline | 0.3s |
| test | Logistic Regression | `max_iter=1000`, `solver='lbfgs'` | 0.8590 | Interpretable linear classifier | N/A |
| test | kernel SVM (RBF) | `C=100`, `gamma='scale'` | 0.9284 | 6 selected features | 1m 39s - 10m |
| test | Soft Voting | N/A | 0.9484 | Probability-based ensemble | N/A |
| test | Bagging (XGBoost) | `max_features=1.0`, `max_samples=1.0`, `n_estimators=20` | 0.962156 | Bagging ensemble using XGBoost | N/A |
| test | Stacking | Meta-model: Logistic Regression | 0.96261 | Ensemble using LR, SVM, and XGBoost base learners | 11m |
| test | XGBoost (Tuned) | `n_estimators=200`, `max_depth=9`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8` | 0.9637 | Top 13 SHAP features | N/A |

Overall, the tuned XGBoost model achieved the highest macro average F1-score among all tested models. Stacking and Bagging with XGBoost were also highly competitive, with macro average F1-scores close to the tuned XGBoost model. Soft Voting improved over Logistic Regression and SVM but did not outperform the strongest tree-based and ensemble approaches. The majority-class baseline performed poorly, confirming that feature-based machine learning models were necessary for this classification task.

## Reproducibility

The code was organized to support reproducibility and easier collaboration. Each major model has its own notebook, and the trained pipelines were saved as `.pkl` files.

To reproduce the project workflow:

1. Clone this repository.
2. Place the dataset files in the `data/` folder if they are not already included.
3. Run the notebooks in the `notebooks/` folder.
4. Use the saved `.pkl` files in the `pipelines/` folder to load trained pipelines directly.
5. Review the generated figures in the `img` folder.

Example pipeline loading:

`import joblib`

`import pandas as pd`

`pipeline = joblib.load("pipelines/logistic_regression_pipeline.pkl")`

`test_data = pd.read_csv("data/test.csv")`

`X_test = test_data.drop(columns=["satisfaction"], errors="ignore")`

`predictions = pipeline.predict(X_test)`

## Requirements

The project uses common Python data science and machine learning libraries, including:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `shap`
- `joblib`

If needed, these packages can be installed with:

`pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib`

## Team Members

- Jungbum Cho
- Karina Shih
- Julia Bunescu

## References

- Kaggle Airline Passenger Satisfaction Dataset:  
  https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

- Project Repository:  
  https://github.com/IuliaBunescu/airline-satisfaction-classification
