---
marp: true
theme: gaia
paginate: true
header: ''
footer: 'ECON6083 Lecture 3 | Tree-Based Methods, Ensembles, and Classification'
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

# Lecture 3
## Tree-Based Methods, Ensembles, and Classification

**ECON6083: Machine Learning in Economics**

---

## Today's Roadmap

**1. Beyond Linearity**
   - Limitations of linear models
   - Non-linear economic realities

**2. Classification Fundamentals**
   - Problem setting and logistic regression

**3. Decision Trees (CART)**
   - Intuition and splitting criteria
   - Regression vs classification trees

---

## Today's Roadmap (Continued)

**4. Random Forests**
   - Bagging and variance reduction
   - Decorrelating trees

**5. Boosting Methods**
   - Gradient boosting algorithms
   - XGBoost and modern implementations

**6. Evaluation & Applications**
   - Classification metrics
   - Credit default case study

---

<!-- _class: lead -->

# Part 1
## The Motivation for Non-Linearity

---

## The Linear Worldview

**Standard linear model:**

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \epsilon$$

**Key assumptions:**
- **Monotonicity:** If $\beta_1 > 0$, increasing $X_1$ always increases $Y$
- **Additivity:** Effect of $X_1$ independent of $X_2$

**Reality:** Economic relationships rarely this simple

---

## Threshold Effects

**Sheepskin effects in education:**
- 11 vs 12 years of schooling (high school diploma)
- Discrete jump in wages beyond linear trend

**Credit score cutoffs:**
- Score of 619 vs 620
- Bank policy creates discontinuity
- Linear model misses this structural break

---

## Complex Interactions

**Interest rate effects on inflation:**

Depend on:
- Current unemployment rate
- Corporate debt levels
- International capital flows

**The curse of dimensionality:**
Specifying all interactions ($X_1 X_2 X_3...$) in OLS impossible

---

## Non-Linear Decision Boundaries

**In classification tasks:**
- Linear models (logit) draw straight line separators
- Real boundaries often "lumpy" or circular

**Example: Credit default**
- High income + high debt = risky
- Low income + low debt = safe
- Circular boundary, not linear

---

<!-- _class: lead -->

# Part 2
## Classification Fundamentals

---

## The Classification Problem

**Outcome:** $Y_i \in \{0, 1, ..., K-1\}$

**Goal:** Estimate $f(X) = P(Y=1 \mid X)$

**Focus today:** Binary classification (0/1)

**Examples:**
- Default vs no default
- Recession vs expansion
- Customer churn vs retention

---

## Why Not Linear Probability Model?

**LPM:** $Y = X\beta + \epsilon$

**Problems:**
- Predictions can be $< 0$ or $> 1$
- Inherent heteroskedasticity: $\text{Var}(\epsilon \mid X) = p(X)(1-p(X))$
- Violates probability axioms

**Solution needed:** Model that constrains output to $(0, 1)$

---

## Logistic Regression: The Baseline

**Models log-odds as linear:**

$$\ln\left(\frac{p}{1-p}\right) = X\beta$$

**Key properties:**
- Predictions guaranteed in $(0, 1)$
- Interpretable odds ratios
- Industry standard for transparency

**Limitation:** Still assumes linear decision boundary

---

<!-- _class: lead -->

# Part 3
## Decision Trees (CART)

---

## The Tree Intuition

<!-- Suggested image: Decision tree diagram showing nodes, splits, and leaf predictions -->

**Decision tree is a step function:**

- Partitions feature space into rectangular regions
- Each region: $R_1, R_2, ..., R_J$

**Prediction rule:**
- **Regression:** Predict mean of training observations in region
- **Classification:** Predict mode (most common class)

---

## Greedy Recursive Splitting

**Algorithm:**

At each step, find best variable $X_j$ and cutoff $s$ to split

**"Greedy" means:**
- Optimize current split only
- Don't look ahead to future splits
- Computationally feasible

**"Recursive" means:**
- Apply same process to each resulting node
- Continue until stopping criterion

---

## Regression Trees: Splitting Criterion

**Goal:** Minimize Residual Sum of Squares (RSS)

$$\min_{j, s} \left[ \sum_{i: x_i \in R_1(j,s)} (y_i - \bar{y}_{R_1})^2 + \sum_{i: x_i \in R_2(j,s)} (y_i - \bar{y}_{R_2})^2 \right]$$

**Economic example: House prices**
- Split 1: Location == 'Downtown'?
- Split 2: SqFt > 2000?

---

## Classification Trees: Impurity Measures

**Cannot use RSS for categorical outcomes**

**Need measure of node purity:**

**Gini Index (standard):**
$$G = \sum_{k=1}^K \hat{p}_{mk} (1 - \hat{p}_{mk})$$

**Cross-Entropy:**
$$D = - \sum_{k=1}^K \hat{p}_{mk} \log \hat{p}_{mk}$$

**Both minimize when node is pure** ($\hat{p} = 0$ or $1$)

---

## Economic Example: Recession Prediction

**Root node:**
- 50% Recession, 50% Expansion (high impurity)

**Split on Yield Curve Inversion:**
- **Left node (inverted):** 80% Recession (purer)
- **Right node (normal):** 10% Recession (purer)

**Result:** Split significantly reduces Gini impurity

---

## The Bias-Variance Trade-off

**Deep tree:**
- Low bias (can fit complex patterns)
- High variance (overfits to training data)
- Can perfectly memorize (1 sample per leaf)

**Shallow tree (stump):**
- High bias (misses patterns)
- Low variance (stable predictions)

**Need to balance!**

---

## Tree Regularization

**Three main approaches:**

**1. Minimum samples per leaf:**
"Don't create rules for fewer than $k$ observations"

**2. Maximum depth:**
"Don't ask more than $d$ questions"

**3. Cost-complexity pruning:**
Grow full tree, then prune branches that don't justify complexity

---

<!-- _class: lead -->

# Part 4
## Random Forests

---

## The Problem with Single Trees

**High variance:**
- Small changes in training data
- Completely different tree structure

**Economic analogy:**
- Single economist: potentially biased
- Survey of 100 economists: consensus more stable

**Solution:** Ensemble methods

---

## Bagging: Bootstrap Aggregating

**Algorithm:**

1. Generate $B$ bootstrap samples (sampling with replacement)
2. Train deep tree on each sample $b$: $\hat{f}^{*b}(x)$
3. Average predictions:

$$\hat{f}_{bag}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}^{*b}(x)$$

**Mathematical insight:**
Averaging $B$ i.i.d. variables with variance $\sigma^2$ reduces variance to $\sigma^2 / B$

---

## The Correlation Problem

**If one predictor dominates (e.g., Credit Score):**
- All bagged trees use it for first split
- Trees highly correlated ($\rho$ high)

**Variance of average:**
$$\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

**If $\rho$ near 1, bagging doesn't help much!**

---

## Random Forest: Decorrelating Trees

<!-- Suggested image: Random Forest ensemble diagram showing multiple diverse trees being aggregated -->

**The RF innovation:**

At each split, consider only random subset of $m$ features

**Typical choice:** $m \approx \sqrt{p}$

**Economic intuition:**
- Force model to explore secondary signals
- Instead of always using Credit Score
- Consider Debt-to-Income, Payment History, etc.

**Result:** Lower $\rho$, greater variance reduction

---

## Out-of-Bag (OOB) Error

**Bootstrap property:**
Each tree uses ~2/3 of data, leaves ~1/3 out

**OOB observations:**
- Weren't used to train that tree
- Can test the tree's performance

**Key benefit:**
No need for separate validation set!
OOB error ≈ unbiased estimate of test error

---

## Random Forest: Practical Considerations

**Hyperparameters:**
- `n_estimators`: Number of trees (more = better, diminishing returns)
- `max_features`: $m$ variables per split
- `max_depth`: Tree depth limit
- `min_samples_leaf`: Minimum leaf size

**Computation:**
Trees grown in parallel (fast on multi-core)

---

<!-- _class: lead -->

# Part 5
## Boosting Methods

---

## Bagging vs Boosting

<!-- Suggested image: Bagging vs Boosting comparison diagram showing parallel vs sequential training -->

**Bagging (Random Forest):**
- Train independent strong learners in parallel
- Reduces variance
- Can't improve bias much

**Boosting:**
- Train weak learners sequentially
- Reduces bias
- Each model learns from previous mistakes

---

## Gradient Boosting: The Algorithm

**Step-by-step process:**

1. Start simple: $F_0(x) = \bar{y}$ (predict mean)
2. Calculate residuals: $r_i = y_i - F_0(x_i)$
3. Fit shallow tree $h_1(x)$ to predict residuals
4. Update: $F_1(x) = F_0(x) + \eta \cdot h_1(x)$
5. Repeat with new residuals

**$\eta$ = learning rate (shrinkage), typically 0.01-0.1**

---

## Boosting: Economic Analogy

**Wage prediction example:**

**Step 1:** OLS with Education explains 60%

**Step 2:** Residuals large for tech workers
→ Next model learns Industry matters

**Step 3:** Add correction, residuals large for older workers
→ Next model learns Age matters

**Sequential refinement of errors**

---

## Why Weak Learners?

**Shallow trees (depth 3-6):**
- Each captures simple pattern
- Won't overfit on their own
- Stable building blocks

**Learning rate $\eta$:**
- Small values (0.01) = slow learning
- More iterations needed
- Better generalization

**Trade-off:** Speed vs accuracy

---

## Modern Boosting Implementations

**XGBoost / LightGBM / CatBoost:**

**Advantages:**
- Optimized for speed (GPU support)
- Handle missing data automatically
- Built-in regularization (L1/L2 penalties)
- Feature importance scores

**State-of-the-art for tabular economic data**

---

## Boosting: Key Hyperparameters

**Number of trees (`n_estimators`):**
- More trees = more learning
- Risk of overfitting (use early stopping)

**Learning rate (`learning_rate`):**
- Lower = slower but smoother
- Typical: 0.01-0.1

**Tree depth (`max_depth`):**
- Shallow trees (3-6) typical
- Controls complexity of each weak learner

---

<!-- _class: lead -->

# Part 6
## Evaluation & Applications

---

## The Confusion Matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | True Negative (TN) | False Positive (FP) |
| **Actual 1** | False Negative (FN) | True Positive (TP) |

**Four possible outcomes for each prediction**

---

## Classification Metrics

**Accuracy:** $(TP + TN) / N$
- Misleading for imbalanced data!

**Precision:** $TP / (TP + FP)$
- "If model says default, is it correct?"

**Recall (Sensitivity):** $TP / (TP + FN)$
- "Did we catch all the defaults?"

**F1 Score:** Harmonic mean of precision and recall

---

## The Accuracy Trap

**Example: Fraud detection**
- 1% of transactions fraudulent
- Trivial classifier: always predict "not fraud"
- Achieves 99% accuracy!

**But misses all fraud**

**Lesson:** Must consider class imbalance

---

## ROC Curve and AUC

<!-- Suggested image: ROC curve example showing TPR vs FPR with AUC calculation -->

**ROC (Receiver Operating Characteristic):**
- Plots True Positive Rate vs False Positive Rate
- Across all classification thresholds

**AUC (Area Under Curve):**
- Single number summary
- AUC = 0.5: Random guessing
- AUC = 1.0: Perfect classifier
- Robust to class imbalance

---

## Case Study: Credit Default Prediction

**Data sources:**
- LendingClub
- Home Mortgage Disclosure Act (HMDA)

**Features:**
- Income, Debt, Age
- Employment length
- Payment history

**Target:** Default (0/1)

---

## Model Comparison: Credit Default

| Model | AUC | Advantages | Disadvantages |
|-------|-----|-----------|---------------|
| Logistic | 0.72 | Interpretable, odds ratios | Linear boundary |
| Decision Tree | 0.68 | Visual, thresholds | High variance |
| Random Forest | 0.79 | Best performance | Black box |
| XGBoost | 0.81 | State-of-art | Complex tuning |

**Random Forest/Boosting typically improve AUC by 5-10%**

---

## The Interpretability Trade-off

**Logistic Regression:**
- Clear equation
- "Odds of default increase 20% per 10% debt increase"
- Easy to explain to stakeholders

**Random Forest/Boosting:**
- No simple equation
- "Black box"

**Solution: Feature importance plots**

---

## Feature Importance

<!-- Suggested image: Feature importance plot showing bar chart of relative feature contributions -->

**Two main approaches:**

**1. Impurity-based (Gini):**
- Built into Random Forest
- Fast to compute
- Can be biased toward high-cardinality features

**2. Permutation importance:**
- Shuffle feature values
- Measure drop in performance
- More reliable but slower

---

## SHAP Values: Individual Explanations

**SHAP (SHapley Additive exPlanations):**

**For individual prediction:**
- Decomposes into feature contributions
- "Why did model predict this person would default?"

**Advantages:**
- Theoretically grounded (game theory)
- Local explanations for each observation
- Compatible with any model

---

<!-- _class: lead -->

# Conclusion

---

## Key Takeaways: Trees vs Linear Models

**When to use trees:**
- Non-linear relationships
- Complex interactions (unknown)
- Threshold effects
- Mixed categorical/continuous features

**When to stick with linear:**
- Need interpretability (policy)
- Small sample size
- Well-understood relationships

---

## Key Takeaways: Ensemble Methods

**Random Forest (Bagging):**
- Reduces variance of unstable trees
- Parallel training (fast)
- Good default choice

**Boosting (XGBoost):**
- Reduces bias through sequential learning
- Often achieves best performance
- Requires more tuning

**Both vastly outperform single trees**

---

## Key Takeaways: Evaluation

**For imbalanced classification:**
- Never trust accuracy alone
- Use precision/recall/F1
- ROC/AUC for threshold-free evaluation

**For model selection:**
- Cross-validation essential
- OOB error convenient for RF
- Early stopping for boosting

---

## Prediction vs Causal Inference

**If goal is prediction (risk management):**
- "Who will default?"
- Random Forest/XGBoost usually best

**If goal is causal inference (policy):**
- "Does interest rate cause default?"
- Need regression + identification strategy
- Or Causal ML (future lectures)

**Different tools for different questions**

---

## Practical Recommendations

**Start simple, build up:**
1. Logistic regression (baseline)
2. Single decision tree (understand structure)
3. Random Forest (reduce variance)
4. XGBoost (squeeze last 1-2% performance)

**Always check:**
- Feature importance for sanity
- Cross-validation for honest performance
- Multiple metrics for robustness

---

## Next Steps in Course

**Building on today:**
- **Lecture 4:** Model selection and validation
- **Lecture 5:** Double Machine Learning
- **Lecture 6:** Causal forests (trees + causality)

**Today's methods foundational for:**
- Nuisance parameter estimation in DML
- Heterogeneous treatment effects
- Conditional average treatment effects (CATE)

---

<!-- _class: lead -->

# Questions?

**Office Hours**: [To be announced]
**Email**: [Your email]
**Course Website**: [Course link]

---

<!-- _class: lead -->

# Thank You!

**Next Lecture**: Cross-Validation, Model Selection & Evaluation

See you next time!
