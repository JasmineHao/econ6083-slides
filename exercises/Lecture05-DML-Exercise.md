# In-Class Exercise: Double/Debiased Machine Learning (DML)

**Course:** ECON6083 - Machine Learning in Economics
**Topic:** DML for Average Treatment Effect (ATE)
**Time:** 25 minutes

---

## Part I: Conceptual Understanding (10 minutes)

### Question 1: Why Naive ML Plug-In Fails

Suppose we want to estimate the effect of job training (D) on earnings (Y), controlling for confounders X (age, education, prior earnings, etc.).

**Naive Approach:**
1. Use Random Forest to predict $\hat{m}_0(X) = \hat{E}[Y|X, D=0]$
2. Estimate treatment effect as: $\hat{\tau}_{naive} = \frac{1}{n_1}\sum_{D_i=1} (Y_i - \hat{m}_0(X_i))$

**Questions:**

**(a)** What is the key problem with this approach?

- [ ] A. Random Forest is too slow
- [ ] B. $\hat{m}_0(X)$ is a biased estimator due to regularization
- [ ] C. We should use Lasso instead
- [ ] D. Random Forest can't handle continuous Y

**Answer:** ___________

**Explanation:**

Even though Random Forest is consistent for $m_0(X) = E[Y|X, D=0]$, the prediction error $\hat{m}_0(X) - m_0(X)$ causes __regularization bias__ in the treatment effect estimate. This bias does NOT vanish even with large samples because it depends on the convergence rate of the ML estimator.

---

**(b)** How does DML solve this problem?

Fill in the blanks:

DML uses ____________ (sample splitting / bootstrapping) to separate the sample used for ___________ (training / testing) the nuisance models from the sample used for ___________ (training / estimating) the treatment effect. This ensures that the prediction errors are ____________ (orthogonal / parallel) to the treatment effect estimand.

**Answers:**

- Sample splitting
- Training
- Estimating
- Orthogonal

---

### Question 2: Cross-Fitting vs Single Split

You have N = 1000 observations. Compare two approaches:

**Approach A (Single Split):**
- Split data 50-50
- Train $\hat{m}(X)$ and $\hat{g}(X)$ on sample 1
- Estimate $\theta$ using residuals on sample 2
- Final estimate uses only 500 observations

**Approach B (Cross-Fitting with K=2 folds):**
- Split data 50-50
- Train on fold 1, estimate on fold 2 → get $\hat{\theta}_1$
- Train on fold 2, estimate on fold 1 → get $\hat{\theta}_2$
- Average: $\hat{\theta} = \frac{1}{2}(\hat{\theta}_1 + \hat{\theta}_2)$
- Final estimate uses all 1000 observations

**Question:** What is the main advantage of Approach B?

**Answer:**

Cross-fitting (Approach B) uses all data for final estimation while maintaining the orthogonality property. This reduces variance compared to single split. The averaging also makes the estimator more stable (less sensitive to particular split).

**Practical tip:** Use K=5 or K=10 folds in practice (not just K=2).

---

## Part II: Mini Implementation (15 minutes)

### Scenario: Effect of 401(k) Participation on Wealth

You are given a simulated dataset with:
- **Treatment (D):** 401(k) participation (0/1)
- **Outcome (Y):** Net financial assets (thousands $)
- **Confounders (X):** Age, income, education, marital status, family size (5 features)

**True data generating process:**
```
D ~ Bernoulli(expit(0.5*age + 0.3*income - 0.2*education))
Y = 5000 + 2500*D + 500*age + 1000*income + (εY with some heterogeneity)
```

**True ATE = $2,500**

---

### Task: Implement Simple DML by Hand

Complete the following code:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data (provided)
# df has columns: D (treatment), Y (outcome), X1-X5 (confounders)
# N = 500

# Step 0: Load and prepare data
# (Assume data is already loaded as df)
X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
D = df['D']
Y = df['Y']

# TODO: Step 1 - Split sample in half (random_state=42)
# Hint: Use train_test_split with test_size=0.5

sample1_idx, sample2_idx = train_test_split(
    np.arange(len(df)), test_size=0.5, random_state=42
)

X1, X2 = X.iloc[sample1_idx], X.iloc[sample2_idx]
D1, D2 = D.iloc[sample1_idx], D.iloc[sample2_idx]
Y1, Y2 = Y.iloc[sample1_idx], Y.iloc[sample2_idx]

# TODO: Step 2 - Fit outcome model m(X) = E[Y|X] on sample 1
# Use RandomForestRegressor with default settings
# Then predict on sample 2

m_model = RandomForestRegressor(random_state=42)
# YOUR CODE HERE: Fit m_model on X1, Y1


# YOUR CODE HERE: Predict Y on sample 2
Y2_pred = # YOUR CODE


# TODO: Step 3 - Fit treatment model g(X) = E[D|X] on sample 1
# Use RandomForestClassifier (or Regressor if you prefer)
# Then predict on sample 2

g_model = # YOUR CODE (RandomForestClassifier or Regressor)


# YOUR CODE HERE: Fit g_model on X1, D1


# YOUR CODE HERE: Predict D on sample 2 (get probabilities if classifier)
D2_pred = # YOUR CODE


# TODO: Step 4 - Compute residuals
# V = Y - m(X)  (outcome residual)
# W = D - g(X)  (treatment residual)

V2 = Y2 - Y2_pred
W2 = D2 - D2_pred

# TODO: Step 5 - Regress outcome residual on treatment residual
# This is just OLS: theta = (W'W)^(-1) W'V

theta_dml_half = np.dot(W2, V2) / np.dot(W2, W2)

print(f"DML estimate (using sample 2): ${theta_dml_half:.2f}")

# TODO: Step 6 - Compare with naive difference in means
# Naive ATE = E[Y|D=1] - E[Y|D=0]

naive_ate = Y[D == 1].mean() - Y[D == 0].mean()

print(f"Naive difference in means: ${naive_ate:.2f}")
print(f"True ATE: $2,500")

# TODO: Bonus - Implement cross-fitting (use both folds)
# Repeat the above but also:
# - Train on sample 2, predict on sample 1 → get theta1
# - Average theta1 and theta2

# YOUR CODE HERE (optional)
```

---

### Expected Output

```
DML estimate (using sample 2): $2,480.32
Naive difference in means: $3,150.78
True ATE: $2,500
```

**Explanation:**
- **DML** is close to true ATE ($2,500)
- **Naive** is biased upward because 401(k) participants have higher income/age (confounding)

---

### Discussion Questions (5 min)

**Q1:** Why is the naive estimator biased even though we have data on confounders?

**Answer:** Because we didn't control for confounders! Naive just compares treated vs control without adjustment. Those who participate in 401(k) are systematically different (higher income, older, etc.).

---

**Q2:** What would happen if we used OLS with controls instead of ML?

**Answer:** OLS with controls (linear regression $Y = \beta_0 + \beta_1 D + \beta_2' X + \varepsilon$) would work if the true model is linear. But if there are interactions or nonlinear relationships, OLS would be misspecified. DML allows flexible ML methods while maintaining valid inference.

---

**Q3:** When would you NOT need DML? (When is simple regression fine?)

**Answer:**
- When you have few controls (low-dimensional X)
- When linear model is a good approximation
- When you don't need sparsity/regularization

**When DML is helpful:**
- Many controls (high-dimensional X)
- Complex interactions / nonlinear relationships
- Want to use ML for flexibility while getting valid standard errors

---

## Part III: Connection to DoubleML Package

In practice, you don't implement DML by hand. Use the `DoubleML` package:

```python
from doubleml import DoubleMLPLR
from doubleml.datasets import make_plr_CCDDHNR2018

# Load data (or use your own)
data = make_plr_CCDDHNR2018(n_obs=500)

# Specify ML methods
from sklearn.ensemble import RandomForestRegressor
ml_g = RandomForestRegressor(n_estimators=100)
ml_m = RandomForestRegressor(n_estimators=100)

# Create DML object
dml_obj = DoubleMLPLR(data, ml_g, ml_m, n_folds=5)

# Fit
dml_obj.fit()

# Results
print(dml_obj.summary)
```

**Advantages of DoubleML:**
- Automatic cross-fitting
- Correct standard errors
- Supports various models (PLR, IRM, PLIV, etc.)
- Handles multiple treatments

---

## Summary: Key Takeaways

1. ✅ **Naive ML plug-in fails** due to regularization bias
2. ✅ **DML uses sample splitting** to orthogonalize nuisance parameters
3. ✅ **Cross-fitting** uses all data and reduces variance
4. ✅ **DML enables valid inference** with flexible ML methods
5. ✅ **DoubleML package** automates the implementation

---

## Preview: Midterm Project

In the Midterm Project (assigned next week), you will:
- Apply DML to a real economic question
- Use `DoubleML` package for implementation
- Compare different ML methods (Lasso, Random Forest, Boosting)
- Interpret results and discuss economic implications

Start thinking about interesting treatment effect questions!

---

## Additional Resources

**Required Reading:**
- Chernozhukov et al. (2018), "Double/Debiased Machine Learning," *Econometrics Journal*
- DoubleML documentation: https://docs.doubleml.org/

**Optional:**
- Chernozhukov et al. (2024), *Applied Causal Inference Powered by ML and AI*, Chapter 3

---

**For Discussion:**
- What economic applications would benefit from DML?
- When is DML overkill? (When should we just use OLS?)
- How do we choose which ML method to use for nuisance models?
