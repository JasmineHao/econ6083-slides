---
marp: true
theme: gaia
paginate: true
header: ''
footer: 'ECON6083 Lecture 2 | Regularization & High-Dimensional Regression'
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

# Lecture 2
## Regularization & High-Dimensional Regression

**ECON6083: Machine Learning in Economics**

---

## Today's Roadmap

**1. The Failure of OLS in High Dimensions**
   - The "wide data" problem
   - Invertibility and variance explosion
   - Bias-variance trade-off

**2. Ridge Regression (L2)**
   - Closed-form solution
   - Geometric interpretation
   - Shrinkage and stability

---

## Today's Roadmap (Continued)

**3. Lasso Regression (L1)**
   - Soft-thresholding and sparsity
   - Variable selection property
   - Elastic Net

**4. Economic Applications**
   - Growth regressions
   - Post-double selection for causal inference

---

<!-- _class: lead -->

# Part 1
## The Failure of OLS in High Dimensions

---

## The Big Data Problem

**Traditional econometrics: "Tall" data**
- $N \gg p$ (many observations, few variables)
- Example: $N=10,000$ individuals, $p=10$ controls

**Modern reality: "Wide" data**
- $p \approx N$ or $p > N$ (many variables)
- Macroeconomics: Hundreds of GDP indicators (Fred-MD)
- Microeconomics: Thousands of confounders
- Text data: Each word a feature

---

## OLS Estimator Recap

**The classic formula:**

$$\hat{\beta}_{OLS} = (X'X)^{-1}X'Y$$

**Requirements:**
- $X'X$ must be full rank
- Requires $\text{rank}(X) = p$

**What could go wrong?**

---

## The Invertibility Problem

**If $p > N$:**
- Rank of $X$ at most $N$
- $X'X$ is singular (determinant = 0)
- $(X'X)^{-1}$ does not exist!

**Consequence:**
- Infinite solutions for $\hat{\beta}$
- Perfect in-sample fit ($R^2 = 1$)
- Terrible out-of-sample prediction

---

## The Variance Explosion

**Even if $p < N$ but $p$ large:**

Multicollinearity makes variance explode

**Via SVD:** $X = U \Sigma V'$

$$\text{Var}(\hat{\beta}_{OLS}) = \sigma^2 \sum_{j=1}^p \frac{v_j v_j'}{d_j^2}$$

**If features correlated:** $d_j \approx 0$

**Result:** $\frac{1}{d_j^2} \to \infty$ (variance explodes!)

---

## Numerical Instability Example

**Highly correlated features:**
- $\hat{\beta}_j$ might be $+1000$ in one sample
- $\hat{\beta}_j$ might be $-1000$ in another sample

**Classic case:**
- "Housing starts" coefficient: $+50$
- "Construction permits" coefficient: $-49$
- They cancel out, but individually unstable

---

## The Bias-Variance Decomposition

<!-- Suggested image: Bias-variance tradeoff curve showing U-shaped MSE as model complexity increases -->

**Mean Squared Error (MSE) breakdown:**

$$E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2_\epsilon$$

**OLS (Gauss-Markov):**
- Bias = 0 (unbiased)
- Variance = Minimal among unbiased estimators

**The regularization insight:**
Accept small bias to drastically reduce variance!

---

<!-- _class: lead -->

# Part 2
## Ridge Regression (L2 Regularization)

---

## The Ridge Estimator

**Add penalty to loss function:**

$$\min_{\beta} \sum_{i=1}^N (y_i - x_i'\beta)^2 + \lambda \sum_{j=1}^p \beta_j^2$$

**Equivalent form:**

$$\min_{\beta} ||Y - X\beta||_2^2 + \lambda ||\beta||_2^2$$

**$\lambda \geq 0$:** Tuning parameter (hyperparameter)

---

## Closed-Form Solution

**Take gradient and set to zero:**

$$\frac{\partial L}{\partial \beta} = -2X'(Y - X\beta) + 2\lambda \beta = 0$$

**Solve:**

$$\hat{\beta}_{Ridge} = (X'X + \lambda I)^{-1}X'Y$$

**Key difference from OLS:** $+\lambda I$ term

---

## Why Ridge Solves Singularity

**Original problem:** $X'X$ might be singular

**Ridge solution:** $(X'X + \lambda I)$

**Effect on eigenvalues:**
- Original: $d_j^2$ (can be $\approx 0$)
- Ridge: $d_j^2 + \lambda$ (always $> 0$)

**Result:** Matrix always invertible!

---

## Geometric Interpretation: Shrinkage

<!-- Suggested image: Ridge regression geometric interpretation showing circular constraint region -->

**SVD representation:**

$$\hat{y}_{Ridge} = \sum_{j=1}^p u_j \frac{d_j^2}{d_j^2 + \lambda} u_j' Y$$

**Shrinkage factor:** $\frac{d_j^2}{d_j^2 + \lambda} < 1$

**Behavior:**
- Small $d_j$ (noise) → shrunk more
- Large $d_j$ (signal) → shrunk less

---

## Bayesian Interpretation

**Ridge equivalent to:**

OLS with Gaussian prior on coefficients

$$\beta_j \sim N(0, \sigma^2/\lambda)$$

**Interpretation:**
- Prior belief: coefficients near zero
- Strength controlled by $\lambda$
- Data updates this belief

---

## Economic Example: Inflation Forecasting

**Scenario:**
100 correlated macro variables (rates, exchange, housing)

**OLS result:**
- "Housing starts": $\hat{\beta} = +50$
- "Construction permits": $\hat{\beta} = -49$
- Wildly oscillating, canceling out

**Ridge result:**
Both get $\hat{\beta} \approx +0.5$ (shared signal)

---

## Ridge: Pros and Cons

**Advantages:**
- Handles multicollinearity
- Always has solution ($p > N$ OK)
- Stable predictions
- Excellent for dense signals (many small effects)

**Disadvantages:**
- **No variable selection** (all $\beta_j \neq 0$)
- Shrinks all coefficients
- Less interpretable (which variables matter?)

---

<!-- _class: lead -->

# Part 3
## Lasso Regression (L1 Regularization)

---

## The Lasso Estimator

**Least Absolute Shrinkage and Selection Operator (Tibshirani, 1996):**

$$\min_{\beta} \frac{1}{2N} \sum_{i=1}^N (y_i - x_i'\beta)^2 + \lambda \sum_{j=1}^p |\beta_j|$$

**Key difference:** L1 penalty (absolute value)

**No closed-form solution!**
- $|\beta|$ not differentiable at 0
- Use coordinate descent algorithm

---

## Why Lasso Yields Sparsity

**The "zero" property:**

Lasso performs **model selection**

**For orthonormal $X$ ($X'X = I$):**

$$\hat{\beta}_j^{Lasso} = \text{sign}(\hat{\beta}_j^{OLS}) \max(|\hat{\beta}_j^{OLS}| - \lambda, 0)$$

**Soft-thresholding:**
- If $|\hat{\beta}_{OLS}| < \lambda$ → $\hat{\beta}_{Lasso} = 0$
- If $|\hat{\beta}_{OLS}| > \lambda$ → shrink by $\lambda$

---

## Ridge vs Lasso: Key Difference

**Ridge:**
- Multiplies by factor $< 1$
- Never exactly zero
- Shrinks proportionally

**Lasso:**
- Subtracts constant $\lambda$
- Can hit exactly zero
- Performs variable selection

---

## Geometric Intuition

<!-- Suggested image: Ridge vs Lasso geometric interpretation showing circle vs diamond constraint regions with RSS contours -->

**Constraint regions:**

**Ridge:** $\beta_1^2 + \beta_2^2 \leq C$ (circle)

**Lasso:** $|\beta_1| + |\beta_2| \leq C$ (diamond)

**RSS contours:** Ellipses centered at OLS solution

**Lasso diamond has corners on axes**
→ Solution likely at corner (one coefficient = 0)

---

## Lasso for Sparse Models

**Ideal for:**
- Few truly important variables
- Many irrelevant features
- Interpretability needed

**Example: Growth regressions**
- 50 potential predictors
- Only 5-10 truly matter
- Lasso identifies them

---

## Lasso Limitations

**Saturation:**
- If $p > N$, Lasso selects at most $N$ variables
- Might miss important features

**Grouping problem:**
- Correlated variables ($\rho \approx 1$)
- Lasso randomly picks one, drops others
- Economists dislike arbitrary selection

---

## Elastic Net: Best of Both Worlds

**Combined penalty:**

$$\min_{\beta} RSS + \lambda_1 ||\beta||_1 + \lambda_2 ||\beta||_2^2$$

**Properties:**
- Sparsity from Lasso
- Stability from Ridge
- Selects groups of correlated variables

**sklearn parameter:** `l1_ratio`
- 0 = pure Ridge
- 1 = pure Lasso
- 0.5 = equal mix

---

<!-- _class: lead -->

# Part 4
## Economic Applications

---

## Growth Regressions

**Classic problem (Sala-i-Martin, 1997):**

"I Just Ran Two Million Regressions"

**Question:** What determines GDP growth?

**Challenge:**
- Theory suggests "Openness" matters
- How to measure? Tariffs? Quotas? Trade volume?
- 50+ possible proxies

**Lasso solution:** Let data select most predictive features

---

## Reducing Researcher Degrees of Freedom

**Traditional approach:**
- Researcher tries many specifications
- Reports "best" one
- p-hacking risk

**Lasso approach:**
- Pre-specify all candidates
- Algorithmic selection
- Reduces arbitrary choices

---

## Causal Inference with High-Dimensional Controls

**Setup (Belloni, Chernozhukov, Hansen, 2014):**

$$Y_i = \alpha D_i + X_i'\beta + \epsilon_i$$

**Goal:** Estimate treatment effect $\alpha$

**Problem:** High-dimensional confounders $X$

---

## The Naive Lasso Mistake

**Wrong approach:**
Simply run Lasso of $Y$ on $D, X$

**Why dangerous?**

Lasso might drop variable that is:
- Strongly correlated with $D$ (confounder!)
- Weakly correlated with $Y$

**Result:** Omitted variable bias in $\hat{\alpha}$

---

## Post-Double Selection Lasso

**Belloni-Chernozhukov-Hansen procedure:**

**Step 1:** Lasso $Y$ on $X$ → select set $S_1$

**Step 2:** Lasso $D$ on $X$ → select set $S_2$

**Step 3:** OLS of $Y$ on $D$ and $(S_1 \cup S_2)$

**Intuition:**
Include variables that predict $Y$ (reduce variance)
AND variables that predict $D$ (remove bias)

---

## Double Selection: Why It Works

**Set $S_1$:** Controls for outcome variation

**Set $S_2$:** Controls for confounding

**Union $S_1 \cup S_2$:**
- Captures confounders even if weak for $Y$
- Maintains low variance
- Delivers consistent $\hat{\alpha}$

**Modern standard for high-dimensional causal inference**

---

<!-- _class: lead -->

# Conclusion

---

## Key Takeaways: The High-Dimensional Problem

**OLS fails when $p$ large:**
- Singularity ($p > N$)
- Variance explosion (multicollinearity)

**Solution: Regularization**
- Accept small bias
- Drastically reduce variance
- Improve out-of-sample prediction

---

## Key Takeaways: Ridge vs Lasso

| Feature | Ridge (L2) | Lasso (L1) | Elastic Net |
|---------|-----------|-----------|-------------|
| **Penalty** | $\sum \beta_j^2$ | $\sum \|\beta_j\|$ | Both |
| **Sparsity** | No (all $\beta \neq 0$) | Yes (some $\beta = 0$) | Yes |
| **Stability** | High | Lower | High |
| **Use case** | Dense signals | Sparse signals | Mixed |

---

## Key Takeaways: Economic Applications

**Prediction tasks:**
- Use Ridge for correlated features (macro forecasting)
- Use Lasso for sparse features (growth regressions)
- Elastic Net as default compromise

**Causal inference:**
- **Never** naive Lasso on $Y \sim D + X$
- Always use post-double selection
- Critical for valid treatment effects

---

## Practical Guidelines

<!-- Suggested image: Lambda selection via cross-validation curve showing MSE vs regularization parameter -->

**Workflow:**
1. **Always standardize** features first
2. Use **cross-validation** to select $\lambda$
3. Compare Ridge, Lasso, Elastic Net
4. Check **coefficient stability** across folds
5. Interpret **selected variables** economically

**Red flags:**
- Unstable coefficient signs
- Sensitivity to single observation
- Perfect in-sample fit

---

## When to Use Each Method

**Ridge:**
- Many correlated predictors
- All potentially relevant
- Prediction focus

**Lasso:**
- Sparse true model
- Interpretability needed
- Variable selection important

**Elastic Net:**
- Uncertain about sparsity
- Want robustness
- Default safe choice

---

## Next Steps in Course

**Building on regularization:**
- **Lecture 3:** Tree-based methods (non-linear)
- **Lecture 4:** Model selection and validation
- **Lecture 5:** Double ML (combines Lasso + causality)

**Today's methods foundational for:**
- High-dimensional causal inference
- Nuisance parameter estimation
- Modern econometric practice

---

<!-- _class: lead -->

# Questions?

**Office Hours**: [To be announced]
**Email**: [Your email]
**Course Website**: [Course link]

---

<!-- _class: lead -->

# Thank You!

**Next Lecture**: Classification & Nonlinear Methods (Trees, Random Forests, Boosting)

See you next time!
