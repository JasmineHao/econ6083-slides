# Homework 3: Demand Estimation with IV-DML

**Due Date:** April 24 at 23:59
**Weight:** 10% of Total Grade

---

## 📋 Assignment Overview

In this assignment, you will estimate **price elasticity of demand** using **Instrumental Variables with Double/Debiased Machine Learning (IV-DML)**. You will address the classic simultaneity problem in demand estimation: prices are determined by supply and demand equilibrium, making naive OLS estimates biased.

### Learning Objectives

By completing this assignment, you will:
- Understand endogeneity in demand estimation (simultaneity bias)
- Apply instrumental variables to isolate exogenous price variation
- Implement traditional 2SLS and modern DML-IV
- Compare parametric vs semiparametric approaches
- Interpret price elasticity from an economic perspective

---

## 📁 Files Included

- `hw3_code.py` - Python template with skeleton code (complete this)
- `hw3_report.md` - Report template (fill in your answers)
- `README.md` - This file with instructions
- `requirements.txt` - Required Python packages
- `data/grocery_demand.csv` - Dataset (provided separately)

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

# Load data
df = pd.read_csv('data/grocery_demand.csv')

print(f"Dataset shape: {df.shape}")
print(df.head())
```

**Expected output:**
- Shape: (5000, 8)
- Columns: quantity, price, cost_shock, store_id, product_category, income, population, trend

---

## 📝 Assignment Tasks

### Part I: Data Exploration (10 points)

Understand the dataset and visualize key relationships.

#### Task 1.1: Load and Describe Data

```python
# Load data
df = pd.read_csv('data/grocery_demand.csv')

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())
```

**Deliverable:** Include summary statistics in your report.

---

#### Task 1.2: Visualize Price vs Quantity

```python
import matplotlib.pyplot as plt

# Scatter plot: log(price) vs log(quantity)
plt.figure(figsize=(10, 6))
plt.scatter(np.log(df['price']), np.log(df['quantity']), alpha=0.3)
plt.xlabel('Log(Price)')
plt.ylabel('Log(Quantity)')
plt.title('Price vs Quantity (Log Scale)')
plt.show()
```

**Question:** What pattern do you observe? Does it suggest a downward-sloping demand curve?

---

#### Task 1.3: Check First Stage Relationship

```python
# Plot: cost_shock vs price
plt.figure(figsize=(10, 6))
plt.scatter(df['cost_shock'], df['price'], alpha=0.3)
plt.xlabel('Cost Shock (IV)')
plt.ylabel('Price')
plt.title('First Stage: IV vs Endogenous Variable')
plt.show()
```

**Question:** Is there a positive relationship? (If yes, the IV is relevant)

---

#### Task 1.4: Discuss Endogeneity

In your report, explain:
- **Why is price endogenous in demand estimation?**
- **What is the expected direction of bias in OLS?**

**Hint:** Price and quantity are determined simultaneously by supply and demand equilibrium. Price is correlated with unobserved demand shocks (preferences, quality, etc.).

---

### Part II: Implementation (50 points)

Implement three methods to estimate price elasticity.

---

#### Task 2.1: Naive OLS (10 points)

Estimate the demand equation using OLS (ignoring endogeneity):

$$\log(Q_i) = \beta_0 + \beta_1 \log(P_i) + \beta_2' X_i + \varepsilon_i$$

where:
- $Q_i$ = quantity
- $P_i$ = price
- $X_i$ = controls (income, population, store FE, product category, trend)

**Code:**

```python
import numpy as np
import statsmodels.api as sm

# Create log variables
df['log_quantity'] = np.log(df['quantity'])
df['log_price'] = np.log(df['price'])

# Prepare controls (example)
X_controls = df[['income', 'population', 'trend']]
# Add categorical dummies for store_id and product_category
X_controls = pd.get_dummies(df[['income', 'population', 'trend', 'store_id', 'product_category']],
                            columns=['store_id', 'product_category'], drop_first=True)

# OLS regression
X_ols = sm.add_constant(pd.concat([df[['log_price']], X_controls], axis=1))
model_ols = sm.OLS(df['log_quantity'], X_ols).fit(cov_type='HC3')

print("OLS Results:")
print(f"Price Elasticity: {model_ols.params['log_price']:.4f}")
print(f"Standard Error: {model_ols.bse['log_price']:.4f}")
print(f"95% CI: [{model_ols.conf_int().loc['log_price', 0]:.4f}, "
      f"{model_ols.conf_int().loc['log_price', 1]:.4f}]")
```

**Expected result:** Price elasticity ≈ -0.7 (biased toward zero)

**Deliverable:** Report elasticity estimate, SE, and 95% CI in your report table.

---

#### Task 2.2: Traditional 2SLS (20 points)

Estimate using Two-Stage Least Squares with cost_shock as the instrument.

**First Stage:**
$$\log(P_i) = \gamma_0 + \gamma_1 Z_i + \gamma_2' X_i + v_i$$

**Second Stage:**
$$\log(Q_i) = \beta_0 + \beta_1 \widehat{\log(P_i)} + \beta_2' X_i + \varepsilon_i$$

where $Z_i$ = cost_shock (instrument)

**Code:**

```python
from linearmodels.iv import IV2SLS

# Prepare variables
y = df['log_quantity']
X_exog = X_controls  # Controls
X_endog = df[['log_price']]  # Endogenous variable
Z = df[['cost_shock']]  # Instrument

# 2SLS estimation
model_2sls = IV2SLS(dependent=y, exog=X_exog, endog=X_endog, instruments=Z).fit(cov_type='robust')

print("\n2SLS Results:")
print(f"Price Elasticity: {model_2sls.params['log_price']:.4f}")
print(f"Standard Error: {model_2sls.std_errors['log_price']:.4f}")
print(f"First Stage F-stat: {model_2sls.first_stage.diagnostics['f.stat'].iloc[0]:.2f}")
```

**Key outputs:**
- Price elasticity estimate
- Standard error (robust)
- **First stage F-statistic** (should be > 10)

**Deliverable:** Report all three values in your report.

---

#### Task 2.3: DML-IV (20 points)

Estimate using Double/Debiased Machine Learning with flexible nuisance models.

**Using DoubleML package:**

```python
from doubleml import DoubleMLData, DoubleMLPLIV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# Prepare data for DoubleML
# Outcome: log_quantity
# Treatment (endogenous): log_price
# Instrument: cost_shock
# Controls: income, population, trend, store FE, product category FE

dml_data = DoubleMLData(
    df,
    y_col='log_quantity',
    d_cols='log_price',
    z_cols='cost_shock',
    x_cols=['income', 'population', 'trend']  # Add categorical dummies if needed
)

# Specify ML methods for nuisance parameters
# ml_g: E[Y | X, Z] (reduced form)
# ml_m: E[D | X, Z] (first stage)
# ml_r: E[Y | D, X, Z] (outcome model - not needed for PLIV)

ml_g = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
ml_m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Create DML-IV model
dml_pliv = DoubleMLPLIV(dml_data, ml_g, ml_m, n_folds=5, n_rep=1)

# Fit
dml_pliv.fit()

# Results
print("\nDML-IV Results:")
print(f"Price Elasticity: {dml_pliv.coef[0]:.4f}")
print(f"Standard Error: {dml_pliv.se[0]:.4f}")
print(f"95% CI: [{dml_pliv.confint()['2.5 %'].iloc[0]:.4f}, "
      f"{dml_pliv.confint()['97.5 %'].iloc[0]:.4f}]")
```

**Alternative: Manual Implementation (Advanced)**

If you prefer to implement DML-IV manually (similar to Lecture 8 exercise), you can use sample splitting and cross-fitting. See template code for structure.

**Deliverable:** Report elasticity, SE, and 95% CI.

---

### Part III: Economic Analysis (40 points)

Write thoughtful analysis in your report.

---

#### 1. Comparison of Methods (15 points)

**Create a table comparing all three methods:**

| Method | Elasticity | Std Error | 95% CI | First Stage F |
|--------|-----------|-----------|--------|---------------|
| OLS | [fill in] | [fill in] | [fill in] | - |
| 2SLS | [fill in] | [fill in] | [fill in] | [fill in] |
| DML-IV | [fill in] | [fill in] | [fill in] | - |

**Discussion questions (200-250 words):**

1. **Why is OLS biased?**
   - Explain the simultaneity problem
   - Direction of bias (toward zero or away from zero?)

2. **Do 2SLS and DML-IV give similar results?**
   - If yes, why? (Hint: relationship is approximately linear)
   - If no, what might explain the difference?

3. **When would DML-IV outperform 2SLS?**
   - High-dimensional controls
   - Nonlinear relationships
   - Complex interactions

---

#### 2. Economic Interpretation (15 points)

**Interpret the price elasticity estimate (200-250 words):**

Assume your 2SLS estimate is -1.2. Discuss:

1. **What does this number mean?**
   - "A 10% increase in price leads to a ___% decrease in quantity demanded"

2. **Is demand elastic or inelastic?**
   - Elastic: |elasticity| > 1
   - Inelastic: |elasticity| < 1

3. **Revenue implications:**
   - If elastic: Lowering price increases total revenue
   - If inelastic: Raising price increases total revenue

4. **Pricing strategy:**
   - Should the firm raise or lower prices?
   - What is the revenue-maximizing price?

**Formula:** Revenue = Price × Quantity. Optimal pricing depends on elasticity.

---

#### 3. IV Validity Discussion (10 points)

**Discuss the three IV assumptions (200-250 words):**

**Assumption 1: Relevance**
- Does cost_shock predict price?
- Evidence: First stage F-statistic = ___
- Interpretation: F > 10 → strong instrument ✓

**Assumption 2: Exclusion Restriction**
- Does cost_shock affect quantity ONLY through price?
- Economic argument: Supply shocks (e.g., shipping costs, oil prices) affect production costs but not consumer preferences
- **Potential threats:**
  - If cost shocks are correlated with product quality
  - If cost shocks proxy for demand shocks (e.g., seasonal patterns)
- **Your assessment:** Is this IV likely valid?

**Assumption 3: Monotonicity**
- Does higher cost always lead to higher price?
- Plausible? (Yes, in competitive markets)

**Conclusion:** Based on your analysis, is the IV credible? What are the main threats to validity?

---

## 📊 Grading Breakdown

| Component | Points | Description |
|-----------|--------|-------------|
| **Data Exploration** | 10 | Visualizations, summary stats, discussion |
| **OLS Implementation** | 10 | Correct regression, interpretation |
| **2SLS Implementation** | 20 | First + second stage, F-stat, results |
| **DML-IV Implementation** | 20 | DoubleML or manual, correct results |
| **Comparison Table** | 15 | Accurate table, thoughtful discussion |
| **Economic Interpretation** | 15 | Elasticity meaning, revenue, pricing |
| **IV Validity** | 10 | Three assumptions, thoughtful analysis |
| **Total** | **100** | |

---

## 🎯 Tips for Success

### Do's ✅

1. **Use log transformations**
   ```python
   df['log_quantity'] = np.log(df['quantity'])
   df['log_price'] = np.log(df['price'])
   ```

2. **Include all controls**
   - Income, population, time trend
   - Store fixed effects (dummies for store_id)
   - Product category fixed effects

3. **Check first stage strength**
   - F-stat > 10 (good)
   - F-stat < 10 (weak instrument problem)

4. **Use robust standard errors**
   - For OLS: `cov_type='HC3'`
   - For 2SLS: `cov_type='robust'`

### Don'ts ❌

1. **Don't forget the instrument**
   ```python
   # WRONG: Using price as its own instrument
   IV2SLS(..., instruments=X_endog)

   # CORRECT: Using cost_shock
   IV2SLS(..., instruments=Z)
   ```

2. **Don't run regression in levels**
   ```python
   # WRONG
   sm.OLS(df['quantity'], df['price'])

   # CORRECT (elasticity interpretation)
   sm.OLS(df['log_quantity'], df['log_price'])
   ```

3. **Don't ignore weak instruments**
   - If F < 10, your results are unreliable
   - Discuss this limitation in your report

---

## 🔍 Common Mistakes to Avoid

### Mistake 1: Positive Elasticity
```python
# If you get a positive elasticity, check:
# 1. Did you put log_quantity on the left (dependent variable)?
# 2. Did you include the correct controls?
# 3. Is your data loaded correctly?
```

### Mistake 2: Weak Instruments
```python
# First stage F-stat = 2
# Problem: Instrument is too weak
# Solution: Check that cost_shock is included and has variation
```

### Mistake 3: DML-IV Errors
```python
# Error: "columns not found"
# Problem: Make sure column names match exactly
# Solution: Check df.columns and dml_data specification
```

---

## 📚 Helpful Resources

### DoubleML Documentation
- Installation: `pip install doubleml`
- Docs: https://docs.doubleml.org/
- Example: https://docs.doubleml.org/stable/examples/py_double_ml_pliv.html

### linearmodels Documentation
- Installation: `pip install linearmodels`
- Docs: https://bashtage.github.io/linearmodels/
- IV2SLS: https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.model.IV2SLS.html

### Econometric Theory
- Angrist & Pischke (2009), *Mostly Harmless Econometrics*, Chapter 4
- Wooldridge (2010), *Econometric Analysis of Cross Section and Panel Data*, Chapter 5

---

## 🆘 Getting Help

### Before Asking for Help

1. Check error messages carefully
2. Re-read the instructions
3. Review Lecture 8 slides on IV-DML
4. Check In-Class Exercise 8 solution

### Where to Ask

- **Course Forum:** Post questions (no code sharing!)
- **Office Hours:** [TBD]
- **TA Sessions:** [TBD]

### What NOT to Do

- ❌ Share your code with classmates
- ❌ Copy code from online sources
- ❌ Ask ChatGPT to write your code
- ❌ Submit work that is not your own

---

## 📦 Submission Instructions

### File Naming

```
hw3_code_<StudentID>_<LastName>.py
hw3_report_<StudentID>_<LastName>.md
```

Example: `hw3_code_12345678_Smith.py`

### Submission Checklist

- [ ] Code runs without errors
- [ ] All three methods implemented (OLS, 2SLS, DML-IV)
- [ ] Comparison table completed
- [ ] Economic interpretation (250+ words)
- [ ] IV validity discussion (250+ words)
- [ ] First stage F-stat reported
- [ ] Files named correctly
- [ ] Both files submitted to Moodle

### Testing Your Code

```bash
# Run your code to make sure it works
python hw3_code_12345678_Smith.py

# Should print: "All estimators completed successfully"
```

---

## ⚖️ Academic Integrity

This is an **individual assignment**. You may:
- Discuss high-level concepts with classmates
- Ask clarifying questions on the forum
- Use DoubleML and linearmodels documentation

You may NOT:
- Share code with classmates
- Copy code from online sources without citation
- Use AI tools to write your code
- Submit work that is not your own

**Violations will be reported and may result in course failure.**

---

## 💡 Final Tips

1. **Start early** - Don't wait until the last minute
2. **Test incrementally** - Run each method separately
3. **Check your results** - Do they make economic sense?
4. **Read the error messages** - They tell you what's wrong
5. **Think like an economist** - Interpret, don't just compute

Good luck! 🚀

---

**Questions?** Post on the course forum or attend office hours.
**Due:** April 24 at 23:59 - No late submissions without prior approval.
