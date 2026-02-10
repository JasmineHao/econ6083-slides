# In-Class Exercise: Instrumental Variables & DML-IV

**Course:** ECON6083 - Machine Learning in Economics
**Topic:** IV for Endogeneity, LATE, and DML for High-Dimensional IV
**Time:** 25 minutes

---

## Part I: IV Validity (10 minutes)

### Scenario: Returns to College Education

**Research Question:** What is the causal effect of college education on earnings?

**Problem:** Ability bias (from Lecture 7)
- High-ability people choose to attend college
- High-ability people also earn more (even without college)
- OLS overestimates the return to college

**Proposed Solution:** Use **distance to the nearest college** as an instrument.

**Variables:**
- **D** = College degree (1 = college, 0 = no college)
- **Y** = Log earnings
- **Z** = Distance to nearest college (miles)
- **U** = Ability (unobserved)

---

### Question 1: IV Validity Conditions

For Z to be a valid instrument, we need **three assumptions:**

#### **Assumption 1: Relevance (First Stage)**
$$Cov(Z, D) \neq 0$$

**In words:** The instrument must predict treatment.

**For our example:**

Does distance to college predict college attendance?

**Answer:** Yes, if living far from college increases the cost (tuition + living expenses), fewer people attend.

**Empirical check:**
```python
from sklearn.linear_model import LinearRegression

# First stage regression: D ~ Z + X
first_stage = LinearRegression().fit(Z, D)
f_stat = compute_f_statistic(first_stage)

print(f"First stage F-statistic: {f_stat:.2f}")
# Rule of thumb: F > 10 for strong instrument
```

**Expected:** F ≈ 50-100 (strong instrument in Card's 1995 study)

---

#### **Assumption 2: Exclusion Restriction**
$$Z \text{ affects } Y \text{ only through } D$$

**In words:** The instrument affects the outcome ONLY by changing treatment (no direct effect).

**For our example:**

Does distance to college affect earnings EXCEPT through college attendance?

**Potential violations:**
- ❌ Distance correlated with rural/urban (different labor markets)
- ❌ Distance correlated with family wealth (richer families live near colleges)
- ❌ Distance affects social networks (college friends → job connections)

**Question:** Which of these is the most serious threat?

**Answer:** The first two are serious. If distance is correlated with labor market characteristics or family background, it directly affects earnings. This violates exclusion restriction.

**How to address:**
- Control for urban/rural, family income, region fixed effects
- Use variation WITHIN regions (e.g., compare people in same city)

**Key insight:** Exclusion restriction is NOT testable! It's an assumption we must defend with economic reasoning.

---

#### **Assumption 3: Monotonicity (for LATE)**
$$D_i(Z=1) \geq D_i(Z=0) \text{ for all } i$$

**In words:** The instrument moves everyone in the same direction (no "defiers").

**Types of individuals (Imbens & Angrist 1994):**
- **Compliers:** Would attend college if nearby, not attend if far (affected by instrument)
- **Always-takers:** Attend college regardless of distance (not affected)
- **Never-takers:** Never attend college regardless of distance (not affected)
- **Defiers:** Attend college if far, NOT if nearby (violates monotonicity!)

**For our example:**

Are there "defiers"—people who attend college ONLY if it's far away?

**Answer:** Hard to imagine! Why would someone prefer a distant college to a nearby one (assuming quality is comparable)? Monotonicity is plausible here.

---

### Question 2: What Does IV Estimate? (LATE)

If all three assumptions hold, IV estimates the **Local Average Treatment Effect (LATE):**

$$\text{LATE} = E[Y_i(1) - Y_i(0) | \text{Complier}]$$

**In words:** The average effect for people whose treatment status is affected by the instrument (compliers).

---

**Question:** Who are the "compliers" in the college distance example?

**Answer:**
- People who would attend college if nearby, but NOT if far
- Likely: Middle-income families who can't afford to send kids far away
- NOT: Rich families (always-takers) or very poor families (never-takers)

**Implication:** IV estimates the return to college for marginal students (those on the fence about attending). This may differ from the average treatment effect (ATE) across all students.

**External validity concern:** LATE ≠ ATE if treatment effects are heterogeneous.

---

**Example numbers (hypothetical):**

| Group | % of population | Return to college |
|-------|----------------|-------------------|
| Always-takers (rich) | 30% | 15% wage gain |
| Compliers (middle-class) | 40% | **25% wage gain** ← IV estimates this |
| Never-takers (poor) | 30% | 35% wage gain (but constrained) |
| **ATE (overall)** | 100% | 25% |

**In this example:** LATE = ATE, but this is coincidence! In general, LATE can be higher or lower than ATE.

---

## Part II: DML-IV Calculation (15 minutes)

### Why Combine IV with DML?

**Traditional 2SLS:**
- Works well with few controls (low-dimensional X)
- Requires linear models for first stage and reduced form

**DML-IV:**
- Handles high-dimensional controls (many X's)
- Allows flexible ML methods (Lasso, Random Forest, etc.)
- Provides valid inference (correct standard errors)

**When is DML-IV useful?**
- Many control variables (e.g., hundreds of demographic features)
- Complex relationships (interactions, nonlinearities)
- Want to use ML for nuisance models while estimating causal effect

---

### Setup: Partially Linear IV Model

**Model:**
$$Y = \theta D + g(X) + U$$
$$D = m(Z, X) + V$$

**Goal:** Estimate $\theta$ (causal effect of D on Y)

**Problem:** If we use ML to estimate $g(X)$ and $m(Z,X)$, naive plug-in is biased (same issue as in Lecture 5!)

**Solution:** DML-IV with orthogonal scores.

---

### Task: Manual DML-IV Implementation

**Given:** Small dataset (N = 100 for illustration)

**Variables:**
- Y = Earnings (outcome)
- D = College degree (endogenous treatment)
- Z = Distance to college (instrument)
- X = 10 control variables (age, family income, test scores, etc.)

---

**Step 1: Sample Splitting**

```python
from sklearn.model_selection import train_test_split

# Split data in half
idx_1, idx_2 = train_test_split(np.arange(N), test_size=0.5, random_state=42)

# Sample 1
Y1, D1, Z1, X1 = Y[idx_1], D[idx_1], Z[idx_1], X[idx_1]

# Sample 2
Y2, D2, Z2, X2 = Y[idx_2], D[idx_2], Z[idx_2], X[idx_2]
```

---

**Step 2: Fit Nuisance Models on Sample 1**

We need to estimate two models:

**Model 1: Reduced Form** $m_0(Z, X) = E[Y | Z, X]$

```python
from sklearn.ensemble import RandomForestRegressor

# Fit reduced form: Y ~ Z + X
rf_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reduced.fit(np.column_stack([Z1, X1]), Y1)

# Predict on sample 2
Y2_pred = rf_reduced.predict(np.column_stack([Z2, X2]))
```

**Model 2: First Stage** $m_1(Z, X) = E[D | Z, X]$

```python
# Fit first stage: D ~ Z + X
rf_first = RandomForestRegressor(n_estimators=100, random_state=42)
rf_first.fit(np.column_stack([Z1, X1]), D1)

# Predict on sample 2
D2_pred = rf_first.predict(np.column_stack([Z2, X2]))
```

---

**Step 3: Compute Residuals (on Sample 2)**

```python
# Outcome residual: V = Y - m0(Z, X)
V2 = Y2 - Y2_pred

# Treatment residual: W = D - m1(Z, X)
W2 = D2 - D2_pred
```

**Interpretation:**
- V2 = part of Y not explained by (Z, X)
- W2 = part of D not explained by (Z, X) (exogenous variation in treatment)

---

**Step 4: IV Regression on Residuals**

```python
# DML-IV estimator: regress V on W (both are residuals)
theta_dml_half = np.cov(V2, W2)[0, 1] / np.var(W2)

print(f"DML-IV estimate (sample 2): {theta_dml_half:.4f}")
```

**What does this estimate?**

This is the **causal effect of D on Y**, using Z as an instrument, after partialling out the effect of X using machine learning.

---

**Step 5: Cross-Fitting (Full DML)**

To use all data and reduce variance, repeat the above using Sample 2 for training and Sample 1 for estimation, then average:

```python
# Repeat steps 2-4 with roles reversed
# Train on sample 2, predict on sample 1
# ...

theta_dml_1 = # estimate using sample 1
theta_dml_2 = theta_dml_half  # estimate using sample 2

# Average
theta_dml = (theta_dml_1 + theta_dml_2) / 2

print(f"DML-IV final estimate: {theta_dml:.4f}")
```

---

**Step 6: Compare with 2SLS**

```python
from statsmodels.sandbox.regression.gmm import IV2SLS

# Traditional 2SLS (linear first stage and reduced form)
iv_model = IV2SLS(Y, np.column_stack([D, X]), np.column_stack([Z, X]))
iv_result = iv_model.fit()

print(f"2SLS estimate: {iv_result.params[0]:.4f}")
print(f"DML-IV estimate: {theta_dml:.4f}")
```

**Expected pattern:**
- If relationship is linear: 2SLS ≈ DML-IV
- If nonlinear: DML-IV may differ (capturing true causal effect)
- DML-IV typically has larger standard errors (more flexible model)

---

### Question: When Would DML-IV and 2SLS Differ?

**Scenario A:** All relationships are linear, no interactions

**Result:** DML-IV ≈ 2SLS (both estimate the same thing)

---

**Scenario B:** First stage is nonlinear
- E.g., $D = \beta_0 + \beta_1 Z + \beta_2 Z^2 + \beta_3 X + \beta_4 X \cdot Z + V$

**Result:** DML-IV captures nonlinearity (uses flexible ML), 2SLS misses it (assumes linearity)

**Advantage of DML-IV:** Flexibility! No need to manually specify interactions.

---

**Scenario C:** Very high-dimensional X (e.g., 100 control variables)

**Result:** 2SLS may fail (too many parameters relative to sample size), DML-IV uses regularization (Lasso, RF) to handle high-dimensionality

---

## Summary: Key Takeaways

1. ✅ **IV solves endogeneity** when we have a valid instrument
2. ✅ **Three assumptions:** Relevance, exclusion restriction, monotonicity
3. ✅ **LATE ≠ ATE:** IV estimates effect for compliers, not whole population
4. ✅ **DML-IV:** Combines IV with ML for high-dimensional controls
5. ✅ **Orthogonalization:** Same logic as DML for ATE (Lecture 5), but with IV
6. ✅ **Comparison:** DML-IV more flexible than 2SLS, but same when linear

---

## Connection to Economic Research

**Classic IV applications:**
- **Returns to education:** Quarter of birth (Angrist & Krueger 1991), distance to college (Card 1995)
- **Military service and earnings:** Draft lottery (Angrist 1990)
- **Trade and growth:** Geography as instrument (Frankel & Romer 1999)
- **Institutions and development:** Settler mortality (Acemoglu, Johnson, Robinson 2001)

**Modern DML-IV applications:**
- **Demand estimation:** Price endogeneity with many product characteristics
- **Labor supply:** Wages endogenous, use tax reforms as IV with rich controls
- **Technology adoption:** Instruments for adoption with heterogeneous firms

---

## Preview: HW3 (IV-DML Track)

In **HW3 Track A**, you will:
- Estimate price elasticity of demand using cost shifters as instruments
- Compare OLS (biased), 2SLS (traditional IV), and DML-IV (flexible)
- See how DML-IV improves on 2SLS when there are many controls

**Dataset:** Simulated grocery scanner data (5,000 obs, 20 features)

**Deliverable:** Code + 3-4 page report with economic interpretation

---

## Additional Resources

**Required Reading:**
- Chernozhukov et al. (2018), "Double/Debiased ML for Treatment and Structural Parameters"
- Angrist & Pischke (2009), *Mostly Harmless Econometrics*, Chapter 4 (IV)

**Classic IV Papers:**
- Angrist, Imbens, Rubin (1996), "Identification of Causal Effects Using IV" (LATE framework)
- Angrist & Krueger (1991), "Does Compulsory School Attendance Affect Schooling and Earnings?"

**Software:**
- DoubleML: `DoubleMLPLIV` class for DML-IV
- statsmodels: `IV2SLS` for traditional 2SLS

**Optional:**
- Mogstad & Wiswall (2016), "Testing the Exclusion Restriction in IV" (sensitivity analysis)

---

**For Discussion:**
- When is IV "credible"? What makes a good instrument?
- How do we defend exclusion restriction in practice?
- What if LATE differs greatly from ATE—does that limit policy relevance?
- Can we ever test the exclusion restriction? (Answer: No, but we can do sensitivity analysis!)
