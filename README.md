# airline-satisfaction-classification
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction


https://docs.google.com/presentation/d/1oRDDgLgQoRqdz-ynhFcY6gvp0A5aB5dX9N_sAqJYoHI/edit?usp=sharing

---

## SVM Model

### Required Imports

```python
import pandas as pd
import joblib
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
```

### Required Features

Your test data must contain these columns:
- `Class`
- `Inflight wifi service`
- `Online boarding`
- `Inflight entertainment`
- `On-board service`
- `Type of Travel`

### Setup

```python
X_test = test_df.drop(columns=["satisfaction", "id"])
y_test = test_df["satisfaction"]
```

### Making Predictions

**For class labels:**
```python
predictions = predict_svm(X_test)
```

**For decision scores (ensemble/stacking):**
```python
scores = predict_svm_scores(X_test)
```

### Evaluation

```python
print("Accuracy:", accuracy_score(y_test, predictions))
print("F1 Macro:", f1_score(y_test, predictions, average="macro"))
print(classification_report(y_test, predictions))
```

### Helper Functions

- `load_svm_pipeline(path)` — Load the saved pipeline
- `predict_svm(X_test, path)` — Get class predictions
- `predict_svm_scores(X_test, path)` — Get decision function scores

---
# Final Scores

| Dataset | Model | Best params | Macro Avg F1-Score | Notes | Time (test-train) |
| :--- | :--- | :--- | :--- | :--- |:--- |
| test | Majority Class (Baseline) |  | 0.36170 | grounding | 0.3s |
| test| kernel SVM (RBF) | C=100, gamma: scale | 0.9284  | best SVM | 1m 39s - 10m |
| test| stacking |  | 0.9321  | placeholder, other models not ready | 1m - 12m|
