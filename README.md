# Airline Passenger Satisfaction Classification

This repository contains our final project for **CSE811 / STT811 Applied Statistical Modeling for Data Scientists**. We worked with the Airline Passenger Satisfaction dataset and built several classification models to predict whether a passenger was **satisfied** or **neutral/dissatisfied**.

The main purpose of the project was not only to get a high prediction score, but also to compare different modeling approaches in a consistent way. We started with a majority-class baseline, then tested individual models such as Logistic Regression, SVM, and XGBoost. After that, we compared ensemble methods including Soft Voting, Stacking, and Bagging with XGBoost.

## Repository Layout

| Folder / File | Description |
|---|---|
| `data/` | Contains the original `train.csv` and `test.csv` files. |
| `img/` | Stores figures used in the final report, such as confusion matrices, validation curves, ROC curves, and feature importance plots. |
| `notebooks/` | Contains the main modeling notebooks for Logistic Regression, SVM, XGBoost, Soft Voting, Stacking, and pipeline testing. |
| `pipelines/` | Contains saved `.pkl` pipeline files for Logistic Regression, SVM, and XGBoost. |
| `README.md` | Overview of the project, repository structure, methods, and final results. |

Most of the modeling work is separated by method. For example, `Karina_lr.ipynb` contains the Logistic Regression workflow, `svm-julia.ipynb` contains the SVM workflow, and `811_XGboost.ipynb` contains the XGBoost workflow, including the XGBoost-based bagging experiment. The ensemble notebooks contain the Soft Voting and Stacking experiments.

## Data and Preprocessing

The dataset includes passenger information, travel characteristics, service ratings, and delay-related variables. The target variable is passenger satisfaction, which was treated as a binary outcome.

Before modeling, we cleaned the data, encoded categorical variables, removed unnecessary columns, and created several additional predictors. For example, age and flight distance were grouped into categories, departure and arrival delays were combined into a total delay variable, and multiple service ratings were averaged into a service-quality score.

## Modeling Approach

We compared a mix of simpler, interpretable models and more flexible machine learning models. Logistic Regression was used as an interpretable baseline model, while SVM was included to capture nonlinear decision boundaries. XGBoost was used as the main tree-based boosting model because it can handle nonlinear relationships and feature interactions well.

We also tested ensemble methods. Soft Voting combined predicted probabilities from Logistic Regression, SVM, and XGBoost. Stacking used the same base models and trained a Logistic Regression meta-model. Bagging was tested with XGBoost as the base learner.

For model comparison, we mainly used **macro average F1-score** because it gives balanced attention to both satisfaction classes.

## Final Scores

| Dataset | Model | Best Params | Macro Avg F1-Score | Notes | Time |
|---|---|---|---:|---|---|
| test | Majority Class Baseline | N/A | 0.36170 | Grounding baseline | 0.3s |
| test | Logistic Regression | `max_iter=1000`, `solver='lbfgs'` | 0.8590 | Linear baseline model | N/A |
| test | kernel SVM (RBF) | `C=100`, `gamma='scale'` | 0.9284 | Used 6 selected features | 1m 39s - 10m |
| test | Soft Voting | N/A | 0.9484 | Probability-based ensemble | N/A |
| test | Bagging (XGBoost) | `max_features=1.0`, `max_samples=1.0`, `n_estimators=20` | 0.962156 | XGBoost bagging ensemble | N/A |
| test | Stacking | Meta-model: Logistic Regression | 0.96261 | LR, SVM, and XGBoost as base models | 11m |
| test | XGBoost (Tuned) | `n_estimators=200`, `max_depth=9`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8` | 0.9637 | Top 13 SHAP features | N/A |

Overall, the tuned XGBoost model produced the best macro average F1-score. Stacking and Bagging with XGBoost were very close, but they did not clearly outperform the tuned XGBoost model. Soft Voting improved over Logistic Regression and SVM, but the strongest results came from the tree-based and ensemble approaches.

## Reproducibility

To make the project easier to rerun and share, we saved the main trained models as pipeline files in the `pipelines/` folder. These files include the preprocessing and modeling steps, so they can be loaded directly instead of rebuilding each model from scratch.

A typical use case is:

`pipeline = joblib.load("pipelines/logistic_regression_pipeline.pkl")`

Then the loaded pipeline can be used to make predictions on the test data after reading the dataset.

The main Python packages used in this project were `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`, and `joblib`.

## Team Members

Jungbum Cho, Karina Shih, and Julia Bunescu

## References

Kaggle Airline Passenger Satisfaction Dataset:  
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

