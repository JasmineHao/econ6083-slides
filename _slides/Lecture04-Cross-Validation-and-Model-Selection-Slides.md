---
marp: true
theme: gaia
paginate: true
header: ''
footer: 'ECON6083 Lecture 4 | Model Selection, Validation, and Causal Identification'
size: 16:9
style: |
  @import 'default';

  section {
    background: linear-gradient(to bottom, #ffffff 0%, #f8fafc 100%);
    font-family: 'Segoe UI', 'Liberation Sans', sans-serif;
    font-size: 22px;
    padding: 70px 80px;
    color: #1e293b;
    line-height: 1.8;
  }

  section::after {
    font-size: 12px;
    color: #64748b;
  }

  h1 {
    color: #0f172a;
    font-weight: 700;
    font-size: 2em;
    margin-bottom: 0.5em;
    border-bottom: 4px solid #5a3bf6;
    padding-bottom: 0.3em;
  }

  h2 {
    color: #361eaf;
    font-weight: 600;
    font-size: 1.5em;
    margin-top: 0.8em;
    margin-bottom: 0.6em;
  }

  h3 {
    color: #3730a3;
    font-weight: 600;
    font-size: 1.2em;
  }

  strong {
    color: #0f172a;
    font-weight: 600;
  }

  ul, ol {
    margin-left: 1.5em;
    line-height: 2.2;
  }

  li {
    margin-bottom: 0.8em;
  }

  p {
    line-height: 1.9;
  }

  code {
    background: #e0e7ff;
    color: #3730a3;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
  }

  pre {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    font-size: 0.85em;
  }

  pre code {
    background: transparent;
    color: #e2e8f0;
    padding: 0;
  }

  table {
    margin: 25px auto;
    border-collapse: collapse;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    font-size: 0.95em;
  }

  table th {
    background: linear-gradient(to bottom, #3b82f6, #4325eb);
    color: white;
    font-weight: 600;
    padding: 10px 16px;
    text-align: left;
  }

  table td {
    padding: 8px 16px;
    border-bottom: 1px solid #e2e8f0;
  }

  table tr:nth-child(even) {
    background: #f8fafc;
  }

  table tr:hover {
    background: #eff6ff;
  }

  blockquote {
    border-left: 4px solid #3b82f6;
    padding-left: 20px;
    margin: 20px 0;
    font-style: italic;
    color: #475569;
  }

  a {
    color: #2563eb;
    text-decoration: none;
    border-bottom: 1px solid #93c5fd;
  }

  a:hover {
    color: #1d4ed8;
    border-bottom-color: #2563eb;
  }

  section.lead {
    background: linear-gradient(135deg, #1e40af 0%, #6d3bf6 100%);
    color: white;
    text-align: center;
    justify-content: center;
  }

  section.lead h1 {
    color: white;
    border-bottom: 4px solid rgba(255,255,255,0.3);
    font-size: 2.2em;
  }

  section.lead h2 {
    color: #dbeafe;
    font-size: 1.4em;
  }

  section.lead strong {
    color: #fbbf24;
  }
---

<!-- _class: lead -->

# Lecture 4
## Model Selection, Validation, and Causal Identification

**ECON6083: Machine Learning in Economics**

---

## Today's Roadmap

**1. The Prediction Foundation**
   - Fragility of simple splits
   - K-Fold cross-validation best practices
   - Hyperparameter tuning strategies

**2. Advanced Validation Techniques**
   - Evaluation metrics for regression and classification
   - Time series cross-validation

---

## Today's Roadmap (Continued)

**3. The Causal Transition**
   - From prediction to causal identification
   - Nuisance parameters and orthogonality

**4. Causal Identification**
   - Potential outcomes and CIA
   - Propensity score methods
   - Doubly robust estimation

**5. LaLonde Case Study**
   - Re-evaluating NSW program
   - Comparing estimators

---

<!-- _class: lead -->

# Part 1
## The Prediction Foundation

---

## The Fragility of Simple Splits

**Why simple train-test splits fail in economics:**

- **Heterogeneous data:** Single split highly unstable
- **Sample underutilization:** Costly in small $n$, large $p$ settings
- **"Luck of the split":** Performance varies with outliers

**The problem:**
Metrics fluctuate based on which observations go where

---

## K-Fold Cross-Validation

<!-- Suggested image: K-fold cross-validation diagram showing data partitioning into folds -->

**Standard approach for estimating out-of-sample skill:**

- Partition data into $K$ equal-sized folds
- Train on $K-1$ folds, test on remaining fold
- Repeat $K$ times, average performance

**Heuristic for $K$:**
- Economic practice: $K = 5$ or $K = 10$
- Balances bias-variance tradeoff
- LOOCV ($K = n$): nearly unbiased but high variance

---

## Stratified K-Fold

**Essential for imbalanced outcomes:**

**When to use:**
- Credit scoring with rare defaults
- Policy targeting with low treatment prevalence
- Any classification with rare positive class

**Key benefit:**
Maintains original class distribution across all folds

---

## Hyperparameter Tuning Strategies

<!-- Suggested image: Grid search heatmap showing hyperparameter combinations and performance -->

**Grid Search:**
- Exhaustive search over parameter grid
- Slow but thorough
- Best for small parameter spaces

**Random Search:**
- Sample randomly from parameter distributions
- More efficient in high dimensions
- Higher chance of finding good regions

---

## Lasso $\lambda$ Selection

**Cross-validation approach:**
- Standard practice in ML
- Computationally intensive

**Rigorous Lasso (Belloni & Chernozhukov 2013):**
- Plug-in penalty based on noise level
- Avoids heavy CV cost
- Stronger theoretical guarantees for inference

---

## Random Forest Best Practices

<!-- Suggested image: Learning curves showing how training and validation error change with data size -->

**Key hyperparameters:**

- **`n_estimators`:** More trees $\neq$ overfitting
  - Gains plateau after ~100 trees

- **`max_depth`, `min_samples_leaf`:** Control complexity
  - Balance between bias and variance

**Causal contexts:**
Use **honest trees** (split/estimate subsample separation)

---

## Advanced Evaluation Metrics: Regression

**Mean Squared Error (MSE) / RMSE:**
- Standard choice
- Sensitive to outliers

**Mean Absolute Error (MAE):**
- More robust to heavy tails
- Better for income/wealth data

**Trade-off:** MSE penalizes large errors more

---

## Classification Metrics: The Accuracy Trap

<!-- Suggested image: Confusion matrix visualization with TP, TN, FP, FN cells -->

**Accuracy misleads with imbalance:**
- Example: 1% fraud rate
- Trivial classifier achieves 99% accuracy

**Better metrics:**
- **Precision:** Of predicted positives, how many correct?
- **Recall:** Of actual positives, how many found?
- **F1 Score:** Harmonic mean of precision and recall

---

## ROC and AUC

**Receiver Operating Characteristic (ROC):**
- Plots True Positive Rate vs False Positive Rate
- Across all classification thresholds

**Area Under Curve (AUC):**
- Measures ranking ability
- Not tied to single cutoff
- Robust to class imbalance

**Interpretation:** Probability model ranks random positive above random negative

---

## Time Series Cross-Validation

**Why standard CV fails:**
- Random shuffling breaks temporal order
- Creates future leakage
- Model uses future to predict past

**Solution: Forward-chaining (rolling origin)**
- Train on data up to time $t$
- Test on $t+1$
- Expand window and repeat

---

## Time Series: Window Strategies

**Expanding window:**
- Uses all past data
- Emphasizes long-run stability
- Good for stable environments

**Sliding window:**
- Fixed-length window
- More responsive to regime changes
- Better for non-stationary series

---

<!-- _class: lead -->

# Part 2
## From Prediction to Causation

---

## The "Causal Switch"

**Critical distinction:**

Good predictive performance $\neq$ good causal estimates

**Why prediction differs from causation:**
- Prediction: $E[Y \mid X]$
- Causation: $E[Y \mid do(X)]$

**Chernozhukov et al. insight:**
ML excels at prediction, requires careful adaptation for causation

---

## Nuisance Parameters

**In Double Machine Learning (DML):**

ML estimates nuisance models, not causal effects directly

**Key nuisance functions:**
- Treatment model: $m_0(X) = E[D \mid X]$
- Outcome model: $g_0(X) = E[Y \mid X]$

**Role:** Remove confounding to isolate causal effect

---

## Orthogonality and Cross-Fitting

**Two pillars of DML:**

**Neyman-orthogonal scores:**
- Reduce sensitivity to nuisance estimation errors
- First-order bias vanishes

**Cross-fitting (sample splitting):**
- Breaks overfitting link
- Nuisance fits on independent data

---

<!-- _class: lead -->

# Part 3
## Causal Identification Frameworks

---

## Potential Outcomes Framework

**Fundamental setup:**
- $Y(1)$: outcome under treatment
- $Y(0)$: outcome under control
- Only one observed per individual

**The fundamental problem:**
Cannot observe both $Y(1)$ and $Y(0)$ for same person

**Causal effect:**
$\tau_i = Y_i(1) - Y_i(0)$ (individual level, unobserved)

---

## Conditional Independence Assumption

**CIA (Unconfoundedness):**

$$(Y(0), Y(1)) \perp D \mid X$$

**Interpretation:**
Treatment is "as good as random" conditional on $X$

**Also called:**
- Selection on observables
- Ignorability of treatment

---

## Overlap (Positivity)

**Assumption:**

$$0 < P(D = 1 \mid X) < 1$$

**Ensures:** Both treated and controls at every $X$

**Violations:**
- $p(X) \approx 0$ or $1$: no counterfactuals
- Leads to unstable weights
- Requires trimming or restricted samples

---

## Propensity Score: Definition

**Propensity score:**

$$p(X) = P(D = 1 \mid X)$$

**Key property (Rosenbaum & Rubin 1983):**
Balancing score - condenses high-dimensional $X$ into scalar

**Implication:**
$(Y(0), Y(1)) \perp D \mid p(X)$

---

## Propensity Score Matching

**Strategy:**
Pair treated and control units with similar $p(X)$

**Common approach:**
- 1:1 nearest-neighbor matching
- With caliper (maximum allowed distance)

**Advantage:** Intuitive, transparent

**Challenge:** Curse of dimensionality without $p(X)$

---

## Inverse Propensity Weighting (IPW)

**Idea:**
Weight observations by inverse treatment probability

**ATE estimator:**

$$\hat{\tau}_{IPW} = \frac{1}{n}\sum_{i=1}^n \left[\frac{D_i Y_i}{\hat{p}(X_i)} - \frac{(1-D_i)Y_i}{1-\hat{p}(X_i)}\right]$$

**Creates pseudo-population where treatment independent of $X$**

---

## Stratification: Cochran's Rule

**Cochran (1968) insight:**

Dividing sample into 5 strata based on $p(X)$ removes ~90% of bias

**Procedure:**
1. Estimate propensity scores
2. Create quintiles based on $\hat{p}(X)$
3. Estimate treatment effect within each stratum
4. Average across strata

---

## Doubly Robust Estimation: Motivation

**Key insight:**

What if we combine propensity score AND outcome modeling?

**Doubly robust property:**
Consistent if **either** model correct (not necessarily both)

**Best of both worlds:**
- IPW debiases outcome model
- Outcome model reduces IPW variance

---

## AIPW Estimator

**Augmented Inverse Propensity Weighted estimator:**

$$\hat{\tau}_{DR} = \frac{1}{n}\sum_{i=1}^n \left[\frac{D_i(Y_i - \hat{m}_1(X_i))}{\hat{p}(X_i)} - \frac{(1-D_i)(Y_i - \hat{m}_0(X_i))}{1-\hat{p}(X_i)} + \hat{m}_1(X_i) - \hat{m}_0(X_i)\right]$$

**Where:**
- $\hat{m}_d(X)$: predicted outcome under treatment $d$
- $\hat{p}(X)$: predicted propensity score

---

## AIPW Intuition

**Three terms in estimator:**

1. **IPW component:** Reweights to balance treatment groups
2. **Outcome model:** Provides baseline predictions
3. **Correction:** IPW debiases outcome model errors

**Robustness:** If $\hat{p}$ or $\hat{m}$ correct, $\hat{\tau}_{DR}$ consistent

---

<!-- _class: lead -->

# Part 4
## The LaLonde Case Study

---

## LaLonde (1986) Warning

**Research question:**
Do non-experimental methods replicate experimental benchmarks?

**National Supported Work (NSW) Program:**
- Randomized job training experiment
- True experimental benchmark available

**LaLonde's finding:**
Standard econometric methods often fail badly

---

## The Failure of OLS

**LaLonde showed:**

OLS estimates highly sensitive to:
- Choice of control variables
- Functional form specification
- Selection of comparison group

**Result:**
Estimates far from experimental benchmark

---

## PSM Breakthrough

**Dehejia & Wahba (1999):**

Using Propensity Score Matching with careful overlap:
- Replicated experimental results closely
- Demonstrated importance of common support

**Key innovations:**
1. Restrict to region of good overlap
2. Use flexible propensity score estimation
3. Check balance diagnostics

---

## Modern Lessons from LaLonde

**Critical practices:**

**1. Overlap is key**
- Assess covariate distributions carefully
- Check propensity score support
- Trim extreme values

**2. Validation strategies**
- Placebo tests with lagged outcomes
- Balance checks across methods

---

## Modern Estimators

**Evolution beyond PSM:**

Current best practice uses:
- Double Machine Learning (DML)
- Doubly robust estimators
- High-dimensional confounders

**Advantages:**
- Handle complex non-linearities
- Robust to misspecification
- Valid inference in high dimensions

---

## Comparison of Estimators

| Method | ATE Estimate | Std. Error | Key Insight |
|--------|------------:|----------:|-------------|
| Experimental | 886.30 | 277.37 | True benchmark |
| OLS | Varies | High | Sensitive to specification |
| PS Matching | 1,079.13 | 158.59 | Good with overlap |
| DML / DR | 370.94 | 394.68 | Robust, high-dim ready |

---

## LaLonde: Key Takeaways

**What we learned:**

1. **Non-experimental methods require care**
   - Can work but need validation

2. **Overlap crucial for credibility**
   - Common support requirements binding

3. **Multiple robustness checks essential**
   - Sensitivity to specifications
   - Placebo tests valuable

---

<!-- _class: lead -->

# Conclusion

---

## Key Takeaways: Prediction

**Cross-validation best practices:**
- Use K-fold ($K = 5$ or 10)
- Stratify for imbalanced outcomes
- Time series requires forward-chaining

**Hyperparameter tuning:**
- Grid search for small spaces
- Random search for high dimensions
- Consider theory-based selection (rlasso)

---

## Key Takeaways: Metrics

**Choose metrics carefully:**

**Regression:**
- MSE for general use
- MAE for robustness to outliers

**Classification:**
- Never rely solely on accuracy
- Use precision/recall/F1 for imbalance
- ROC/AUC for threshold-free evaluation

---

## Key Takeaways: Causation

**From prediction to causation:**

1. **Prediction $\neq$ causation**
   - Requires different framework

2. **Identification strategies**
   - CIA and overlap required
   - Propensity scores as dimension reduction

3. **Doubly robust methods**
   - Protect against misspecification
   - Combine PS and outcome modeling

---

## The Complete Pipeline

**Model validation + Causal identification:**

1. **Validate predictions:** Proper CV, metrics
2. **Check identification:** Overlap, balance
3. **Robust estimation:** DR/DML methods
4. **Sensitivity analysis:** Multiple specifications

**This lecture bridges ML prediction and causal inference**

---

<!-- _class: lead -->

# Questions?

**Office Hours**: [To be announced]
**Email**: [Your email]
**Course Website**: [Course link]

---

<!-- _class: lead -->

# Thank You!

**Next Lecture**: Double/Debiased Machine Learning (DML)

See you next time!
