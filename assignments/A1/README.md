# Homework 1: Hedonic Pricing with OLS and Penalized Regression

**Due Date:** Friday, Week 4 at 23:59
**Weight:** 10% of Total Grade

---

## 📋 Assignment Overview

In this assignment, you will build an Automated Valuation Model (AVM) for housing prices using the **Ames Housing Dataset**. You will compare the performance of **Ordinary Least Squares (OLS)** against **Penalized Regression methods (Ridge and Lasso)** to empirically demonstrate the **Bias-Variance Trade-off**.

### Learning Objectives

By completing this assignment, you will:
- Understand the limitations of OLS in high-dimensional settings
- Apply regularization techniques (Ridge and Lasso) for prediction tasks
- Perform feature engineering with economic intuition
- Interpret model coefficients from a hedonic pricing perspective
- Compare model performance using proper validation techniques

---

## 📁 Files Included

- `hw1_code.py` - Python template with skeleton code (complete this)
- `hw1_report.md` - Report template (fill in your answers)
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
from sklearn.datasets import fetch_openml

# Load Ames Housing data
data = fetch_openml(name="house_prices", version=1, as_frame=True, parser='auto')
X = data.data
y = data.target

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Target variable: {y.name}")
```

---

## 📝 Assignment Tasks

### Part I: Theoretical Concepts (10 points)

Complete the fill-in-the-blank questions in `hw1_report.md`. These questions test your understanding of:
- Bias-variance decomposition
- Overfitting in high dimensions
- How regularization helps
- Properties of Lasso vs Ridge

### Part II: Implementation (50 points)

Complete the `EconomicHousingModel` class in `hw1_code.py`. You must implement:

#### Task A: Feature Engineering (15 points)
- Log-transform the target variable (SalePrice)
- Handle missing values with economic intuition
- Create at least 3 new features:
  - TotalSF (Total Square Footage)
  - QualityArea (Quality × Area interaction)
  - HouseAge
  - [One more of your choice]

#### Task B: Model Pipeline (20 points)
- Build a scikit-learn Pipeline with proper preprocessing
- Prevent data leakage (fit scaler on train only!)
- Implement OLS, Ridge, and Lasso models
- Use GridSearchCV for hyperparameter tuning (Ridge/Lasso)

#### Task C: Model Evaluation (15 points)
- Split data into 80% train / 20% test
- Calculate RMSE for all three models on test set
- Extract and analyze feature importance (Lasso)

### Part III: Economic Analysis (40 points)

In `hw1_report.md`, provide thoughtful analysis:

1. **OLS Failure Analysis (15 points)**
   - Explain why OLS underperforms in high dimensions
   - Discuss bias-variance tradeoff

2. **Sparsity Analysis (10 points)**
   - How many features did Lasso eliminate?
   - What does this tell us about feature redundancy?

3. **Hedonic Interpretation (15 points)**
   - List top 5 most important features
   - Interpret their economic significance
   - Do the results align with theory?

---

## 📤 Submission Instructions

### What to Submit

Submit **exactly two files** to Moodle:

1. **`hw1_code.py`** - Your completed Python code
2. **`hw1_report.md`** - Your completed analysis report

### File Naming Convention

**IMPORTANT:** Use this exact format:

```
hw1_code_<StudentID>_<LastName>.py
hw1_report_<StudentID>_<LastName>.md
```

Example:
```
hw1_code_12345678_Smith.py
hw1_report_12345678_Smith.md
```

### Checklist Before Submission

- [ ] My code runs without errors
- [ ] I included my name and student ID in both files
- [ ] All TODO sections are completed
- [ ] My `EconomicHousingModel` class has `fit` and `predict` methods
- [ ] I filled in all blanks in the report
- [ ] I included RMSE results for all three models
- [ ] I provided economic interpretation of the top features
- [ ] File names follow the required format

---

## 🔍 Testing Your Code

Before submitting, test your implementation:

```python
# Test basic functionality
from hw1_code import EconomicHousingModel, load_and_split_data, compare_models

# Load data
X_train, X_test, y_train, y_test = load_and_split_data()

# Test OLS model
model = EconomicHousingModel(model_type='ols')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"OLS predictions shape: {predictions.shape}")

# Test Ridge model
model_ridge = EconomicHousingModel(model_type='ridge')
model_ridge.fit(X_train, y_train, tune_hyperparameters=True)

# Compare all models
results = compare_models(X_train, X_test, y_train, y_test)
print(results)
```

---

## 📊 Expected Results

Your models should achieve approximately:

- **OLS RMSE:** 0.12 - 0.18 (log scale)
- **Ridge RMSE:** 0.11 - 0.14 (log scale)
- **Lasso RMSE:** 0.11 - 0.14 (log scale)

*Note: Exact values depend on your feature engineering and random seed.*

If your RMSE is much higher (>0.25), check:
- Did you log-transform the target variable?
- Are you scaling numerical features?
- Did you handle missing values correctly?

---

## 💡 Hints and Tips

### Feature Engineering Hints

**Missing Values with Economic Meaning:**
- `PoolQC`, `Fence`, `Alley`, `FireplaceQu`, `GarageType` → NA means "None"
- These should be filled with "None" or a separate category

**Good Interaction Features:**
- Total living space (basement + floors)
- Quality-adjusted area (OverallQual × GrLivArea)
- Bathroom count (FullBath + 0.5×HalfBath)
- Age effects (current year - year built)

### Pipeline Best Practices

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate preprocessing for numerical and categorical
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Search in log-space for alpha
param_grid = {'alpha': np.logspace(-4, 4, 20)}

grid_search = GridSearchCV(
    Lasso(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
```

---

## ❓ Common Questions

**Q: How many features should I create?**
A: At minimum 3 as specified, but more is better! Think about economically meaningful interactions.

**Q: Should I include all original features?**
A: Yes! Let Lasso do the feature selection. Don't manually drop features unless they cause technical issues.

**Q: What if my OLS performs better than Lasso?**
A: This is unlikely if you have many features. Check your implementation - you might have data leakage or incorrect cross-validation.

**Q: How do I interpret a negative coefficient?**
A: In log-scale, a coefficient β means a 1-unit increase in X leads to a exp(β) multiplicative change in price.

**Q: Can I use Python notebooks (.ipynb)?**
A: No. Submit only `.py` files. The autograder cannot run notebooks.

---

## 📚 Resources

### Required Reading
- Varian, H. R. (2014). "Big Data: New Tricks for Econometrics"
- Mullainathan & Spiess (2017). "Machine Learning: An Applied Econometric Approach"

### Documentation
- [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Pandas Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Ames Housing Data Dictionary](https://www.openml.org/d/42165)

### Tutorials
- [Ridge vs Lasso Explained](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)

---

## 🆘 Getting Help

If you're stuck:
1. Check the hints section above
2. Review the lecture slides on Regularization
3. Post on the course discussion forum (no code sharing!)
4. Attend office hours: [Time TBD]

---

## ⚠️ Academic Integrity

- You may discuss concepts with classmates
- You may NOT share code or copy solutions
- All submitted work must be your own
- Violation will result in a zero and disciplinary action

---

## 🎯 Grading Rubric

| Component | Points | Description |
|-----------|--------|-------------|
| **Code Functionality** | 30 | Code runs without errors, follows class structure |
| **Methodology** | 20 | Correct CV, feature engineering, no data leakage |
| **Theoretical Understanding** | 10 | Fill-in-the-blank questions correct |
| **Empirical Results** | 20 | Ridge/Lasso beat OLS, reasonable RMSE values |
| **Economic Interpretation** | 20 | Clear analysis of bias-variance, features, sparsity |
| **Total** | 100 | |

---

**Good luck! Remember: prediction ≠ inference, but both are valuable!** 🚀
