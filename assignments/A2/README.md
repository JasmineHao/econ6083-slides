# Homework 2: Income Prediction with Tree-Based Models

**Due Date:** Week 6 at 23:59
**Weight:** 15% of Total Grade

---

## 📋 Assignment Overview

In this assignment, you will build **income classification models** using the **UCI Adult Income Dataset**. You will predict whether an individual earns more than $50,000 per year based on census data. You will compare the performance of **Decision Trees**, **Random Forests**, and **Gradient Boosting** to understand ensemble methods and handle class imbalance.

### Learning Objectives

By completing this assignment, you will:
- Fit and tune decision tree classifiers
- Understand ensemble methods (bagging vs boosting)
- Handle imbalanced datasets with appropriate metrics
- Perform feature importance analysis from an economic perspective
- Evaluate algorithmic fairness and bias

---

## 📁 Files Included

- `a2_code.py` - Python template with skeleton code (complete this)
- `a2_report.md` - Report template (fill in your answers)
- `README.md` - This file with instructions
- `requirements.txt` - Required Python packages

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Test Data Loading

```python
import pandas as pd

# Load UCI Adult Income data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['income'].value_counts()}")
print(f"Missing values:\n{df.isnull().sum()}")
```

**Expected output:**
- Shape: (32561, 15)
- Class distribution: ~75% ≤50K, ~25% >50K
- Missing values in: workclass, occupation, native-country

---

## 📝 Assignment Tasks

### Part I: Theoretical Concepts (15 points)

Complete the questions in `a2_report.md`. These questions test your understanding of:
- Decision tree splitting criteria (Gini vs Entropy)
- Why ensembles reduce variance
- Difference between bagging and boosting
- Bias-variance tradeoff in tree depth
- Handling imbalanced data

---

### Part II: Implementation (55 points)

Complete the `IncomeClassifier` class in `a2_code.py`. You must implement:

#### Task A: Data Preprocessing (15 points)

```python
def _preprocess_data(self, df):
    # 1. Handle missing values ('?')
    # 2. Handle categorical variables:
    #    - Binary encoding for binary features (sex, income)
    #    - OneHotEncoding or LabelEncoding for multi-class features
    # 3. Create new features (feature engineering):
    #    - capital_gain_indicator (binary: has capital gain or not)
    #    - hours_category (part-time, full-time, overtime)
    #    - age_group (young, middle-aged, senior)
    # 4. Drop redundant features (e.g., education-num vs education)
    # Return: X (features), y (target)
```

**Economic Motivation:**
- Capital gains are highly skewed (many zeros) → indicator variable captures participation
- Hours worked: Non-linear relationship with income (part-time vs overtime)
- Age: Life-cycle income patterns suggest grouping

---

#### Task B: Model Pipeline (25 points)

Build and compare **three models**:

**1. Single Decision Tree (8 points)**
```python
def train_decision_tree(self, X_train, y_train):
    # - Use GridSearchCV to tune max_depth, min_samples_split
    # - Parameter grid: max_depth in [3, 5, 7, 10, 15]
    #                   min_samples_split in [10, 20, 50]
    # - Use 5-fold cross-validation
    # - Return best model
```

**2. Random Forest (8 points)**
```python
def train_random_forest(self, X_train, y_train):
    # - Tune n_estimators, max_depth, min_samples_split
    # - Parameter grid: n_estimators in [50, 100, 200]
    #                   max_depth in [10, 15, 20, None]
    #                   min_samples_split in [10, 20]
    # - Use 5-fold cross-validation
    # - Extract feature importance after training
    # - Return best model
```

**3. Gradient Boosting (9 points)**
```python
def train_gradient_boosting(self, X_train, y_train):
    # - Use XGBoost or LightGBM
    # - Tune n_estimators, learning_rate, max_depth
    # - Parameter grid: n_estimators in [50, 100, 200]
    #                   learning_rate in [0.01, 0.05, 0.1]
    #                   max_depth in [3, 5, 7]
    # - Use 5-fold cross-validation
    # - Return best model
```

**Important Requirements:**
- Use **train/validation/test split** (60% / 20% / 20%) OR train/test (80/20) with CV
- Implement with **sklearn.pipeline.Pipeline** to prevent data leakage
- Use **class_weight='balanced'** or custom weights to handle imbalance
- Report **Precision, Recall, F1-Score, AUC-ROC** (not just accuracy!)

---

#### Task C: Model Evaluation (15 points)

```python
def evaluate_models(self, X_test, y_test):
    # For each of the 3 models:
    # 1. Generate predictions on test set
    # 2. Calculate and display:
    #    - Confusion matrix
    #    - Classification report (precision, recall, F1)
    #    - ROC curve and AUC score
    #    - Precision-Recall curve
    # 3. Create comparison table
    # Return: dictionary with all metrics
```

**Deliverable:** Create a summary table like this:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Decision Tree | 0.83 | 0.68 | 0.55 | 0.61 | 0.78 |
| Random Forest | 0.86 | 0.75 | 0.62 | 0.68 | 0.89 |
| Gradient Boosting | 0.87 | 0.76 | 0.65 | 0.70 | 0.91 |

---

### Part III: Economic Analysis (30 points)

In `a2_report.md`, provide thoughtful written analysis:

#### 1. Interpretability vs Performance (10 points)

**Question:** Compare the single decision tree (interpretable) vs ensemble methods (accurate).

**Address:**
- Which model would you deploy for policy decisions? Why?
- Trade-offs in regulatory contexts (e.g., Fair Lending laws requiring explainability)
- Can you achieve both interpretability AND performance?
- What is the economic cost of the accuracy gap?

**Expected length:** 250-350 words

---

#### 2. Feature Importance Analysis (10 points)

**Question:** Analyze the top 5 most important features from your Random Forest or Gradient Boosting model.

**Address:**
- List the top 5 features and their importance scores
- Provide **economic interpretation** for each:
  - Why is this feature predictive of income?
  - Does it align with labor economics theory?
  - Examples: "Education is the strongest predictor, consistent with human capital theory..."
- Compare feature importance across models (are they consistent?)

**Expected length:** 200-300 words + table

---

#### 3. Fairness and Bias (10 points)

**Question:** Examine potential discrimination in your model's predictions.

**Tasks:**
- Calculate error rates (False Positive Rate, False Negative Rate) by:
  - Sex (male vs female)
  - Race (if comfortable doing so)
- Discuss: Are error rates balanced across groups?
- If not, what are the implications?
- How would you mitigate bias? (e.g., fairness constraints, re-weighting)

**Expected length:** 250-350 words + metrics table

**Example analysis:**
```
Group       FPR    FNR   F1-Score
Male        0.15   0.28   0.72
Female      0.18   0.35   0.65

→ Female applicants have higher FNR (more missed high earners)
→ Potential causes: historical discrimination in labor market
→ Mitigation: Train separate models, add fairness constraints
```

---

## 📊 Grading Breakdown

| Component | Points | Description |
|-----------|--------|-------------|
| **Theoretical Questions** | 15 | Part I in report |
| **Data Preprocessing** | 15 | Handling categoricals, missing values, feature engineering |
| **Decision Tree** | 8 | Correct implementation with hyperparameter tuning |
| **Random Forest** | 8 | Correct implementation with feature importance |
| **Gradient Boosting** | 9 | XGBoost/LightGBM with tuning |
| **Model Evaluation** | 15 | Metrics, ROC curves, comparison table |
| **Interpretability Analysis** | 10 | Part III.1 in report |
| **Feature Importance** | 10 | Part III.2 in report |
| **Fairness Analysis** | 10 | Part III.3 in report |
| **Total** | **100** | |

---

## 🎯 Tips for Success

### Do's ✅

1. **Use pipelines** to prevent data leakage
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler

   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('model', DecisionTreeClassifier())
   ])
   ```

2. **Handle missing values** before encoding
   ```python
   df.replace(' ?', np.nan, inplace=True)
   df.fillna(df.mode().iloc[0], inplace=True)  # Or use SimpleImputer
   ```

3. **Check class distribution**
   ```python
   print(y_train.value_counts(normalize=True))
   # If imbalanced, use class_weight='balanced'
   ```

4. **Use multiple metrics** (not just accuracy)
   ```python
   from sklearn.metrics import classification_report, roc_auc_score
   print(classification_report(y_test, y_pred))
   print(f"AUC: {roc_auc_score(y_test, y_pred_proba)}")
   ```

### Don'ts ❌

1. **Don't ignore class imbalance** - Use F1-score, not just accuracy
2. **Don't fit on test data** - Always fit preprocessing on train only
3. **Don't skip cross-validation** - Needed for hyperparameter tuning
4. **Don't use default hyperparameters** - GridSearchCV is required
5. **Don't forget to encode categoricals** - Trees need numerical input

---

## 🔍 Common Mistakes to Avoid

### Mistake 1: Data Leakage
```python
# WRONG
encoder.fit(X)  # Uses future information!
X_train, X_test = train_test_split(X, y)

# CORRECT
X_train, X_test = train_test_split(X, y)
encoder.fit(X_train)  # Only training data
X_train_enc = encoder.transform(X_train)
X_test_enc = encoder.transform(X_test)
```

### Mistake 2: Ignoring Missing Values
```python
# WRONG: Dataset has '?' not np.nan
df.isnull().sum()  # Shows 0 missing (but they're there as '?')

# CORRECT
df.replace(' ?', np.nan, inplace=True)
df.isnull().sum()  # Now shows missing values
```

### Mistake 3: Wrong Metric for Imbalanced Data
```python
# WRONG: High accuracy but useless model
print(f"Accuracy: {0.75}")  # Just predicting majority class!

# CORRECT: Check precision, recall, F1
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## 📚 Helpful Resources

### Scikit-learn Documentation
- [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

### XGBoost/LightGBM
- [XGBoost Python API](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)

### Handling Imbalanced Data
- [imbalanced-learn Documentation](https://imbalanced-learn.org/)
- Use `class_weight='balanced'` in sklearn models

---

## 🆘 Getting Help

### Before Asking for Help

1. Check error messages carefully
2. Re-read the instructions
3. Review Lecture 3 and 4 slides
4. Check In-Class Exercise 3 solution
5. Search for the error on StackOverflow

### Where to Ask

- **Course Forum:** Post conceptual questions (no code sharing!)
- **Office Hours:** [TBD]
- **TA Sessions:** [TBD]

### What NOT to Do

- ❌ Share your code with classmates
- ❌ Copy code from online sources without attribution
- ❌ Ask TAs to debug your entire code
- ❌ Post full solutions online

---

## 📦 Submission Instructions

### File Naming

```
a2_code_<StudentID>_<LastName>.py
a2_report_<StudentID>_<LastName>.md
```

Example: `a2_code_12345678_Smith.py`

### Submission Checklist

- [ ] Code runs without errors
- [ ] All three models implemented
- [ ] Metrics table completed
- [ ] All theoretical questions answered
- [ ] Economic analysis sections completed (250+ words each)
- [ ] Fairness analysis with metrics
- [ ] Files named correctly
- [ ] Both files submitted to Moodle

### Testing Your Code

```bash
# Run your code to make sure it works
python a2_code_12345678_Smith.py

# Check for errors
# Should print: "All models trained successfully"
```

---

## ⚖️ Academic Integrity

This is an **individual assignment**. You may:
- Discuss high-level concepts with classmates
- Ask clarifying questions on the forum
- Use online documentation (sklearn, pandas)

You may NOT:
- Share code with classmates
- Copy code from online sources without citation
- Use ChatGPT/AI to write code for you
- Submit work that is not your own

**Violations will be reported and may result in course failure.**

---

## 🎓 Connection to Lectures

This assignment builds on:
- **Lecture 3:** Trees, Random Forests, Boosting
- **Lecture 4:** Cross-Validation and Model Selection
- **In-Class Exercise 3:** Classification with Trees
- **In-Class Exercise 4:** Cross-Validation

Make sure you understand these lectures before starting!

---

## 💡 Final Tips

1. **Start early** - Don't wait until the last minute
2. **Test incrementally** - Don't write all code at once
3. **Use print statements** - Debug by checking intermediate results
4. **Read the error messages** - They often tell you exactly what's wrong
5. **Think like an economist** - This isn't just coding; it's applied economics

Good luck! 🚀

---

**Questions?** Post on the course forum or attend office hours.
**Due:** Week 6 at 23:59 - No late submissions accepted without prior approval.
