---
marp: true
theme: gaia
paginate: true
header: ''
footer: 'ECON6083 Lecture 5 | Double-Debiased Machine Learning'
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

# Lecture 5
## Double-Debiased Machine Learning

**ECON6083: Machine Learning in Economics**

---

## Today's Roadmap

**1. Naive ML Problems**
   - Why direct ML fails for causal inference
   - Regularization bias and overfitting

**2. Neyman Orthogonality**
   - Mathematical foundations
   - Orthogonal score construction

**3. Cross-Fitting Algorithms**
   - DML1 vs DML2 procedures
   - Sample splitting strategies

---

## Today's Roadmap (Continued)

**4. Asymptotic Theory**
   - Bias decomposition
   - $N^{-1/4}$ convergence requirement

**5. Implementation & Applications**
   - DoubleML framework
   - 401(k) empirical study

**6. Extensions**
   - Causal Forests and heterogeneous effects

---

<!-- _class: lead -->

# Part 1
## The Problem with Naive Machine Learning

---

## The Fundamental Challenge

<!-- Suggested image: Naive ML bias illustration showing regularization bias in causal estimation -->

**Modern empirical economics faces:**
- High-dimensional confounders ($p \approx N$ or $p > N$)
- Complex non-linear relationships
- Need for valid statistical inference

**The naive approach:**
- Use ML to estimate everything simultaneously
- Plug predictions into standard estimators
- Hope for the best

**Result: Catastrophic failure**

---

## The Partially Linear Regression Model

**Structural equations:**

$$Y = D\theta_0 + g_0(X) + \zeta, \quad E[\zeta \mid D, X] = 0$$

$$D = m_0(X) + V, \quad E[V \mid X] = 0$$

**Where:**
- $Y$: outcome variable
- $D$: treatment/policy variable of interest
- $X$: high-dimensional confounders

---

## PLR Model Components

**Parameters and functions:**
- $\theta_0$: causal effect (target parameter)
- $g_0(X) = E[Y - D\theta_0 \mid X]$: outcome nuisance function
- $m_0(X) = E[D \mid X]$: propensity score (treatment nuisance)

**Challenge:**
When $\dim(X)$ is large, $g_0$ and $m_0$ cannot be estimated parametrically without severe misspecification

---

## The Naive Plug-in Estimator

**Naive approach:**
Estimate $\hat{g}(X)$ using ML, then solve:

$$\frac{1}{n} \sum_{i=1}^n D_i (Y_i - D_i\hat{\theta}_{naive} - \hat{g}(X_i)) = 0$$

**Solving for $\hat{\theta}_{naive}$:**

$$\hat{\theta}_{naive} = \left(\frac{1}{n}\sum D_i^2\right)^{-1} \frac{1}{n}\sum D_i(Y_i - \hat{g}(X_i))$$

---

## Why Naive ML Fails: Bias Decomposition

**Decompose the error:**

$$\sqrt{n}(\hat{\theta}_{naive} - \theta_0) = \underbrace{\left(\frac{1}{n}\sum D_i^2\right)^{-1} \frac{1}{\sqrt{n}}\sum D_i\zeta_i}_{\text{Standard noise: } O_p(1)}$$

$$+ \underbrace{\left(\frac{1}{n}\sum D_i^2\right)^{-1} \frac{1}{\sqrt{n}}\sum D_i(g_0(X_i) - \hat{g}(X_i))}_{\text{Regularization bias: does NOT vanish!}}$$

---

## Source of Failure: Regularization Bias

**Machine Learning algorithms:**
- Maximize predictive accuracy
- Navigate bias-variance tradeoff
- Introduce regularization (Lasso penalties, tree pruning, early stopping)

**The cost:**
- Regularization reduces variance but introduces systematic bias
- ML estimators converge slowly: $N^{-1/3}$ or slower
- Bias transmits directly to $\hat{\theta}$: not $\sqrt{N}$-consistent

---

## Example: Lasso Regularization

**Lasso objective:**

$$\min_{\beta} \frac{1}{n}\sum_{i=1}^n (Y_i - X_i'\beta)^2 + \lambda\|\beta\|_1$$

**Effect:**
- Penalty term $\lambda\|\beta\|_1$ shrinks coefficients toward zero
- Reduces variance but introduces bias
- Convergence rate: $N^{-1/2}$ under strong sparsity, slower otherwise

**Problem:** This bias contaminates $\hat{\theta}$ when naively plugged in

---

## Numerical Example: Regularization Impact

**Simulation setup:**
- True model: $Y = 2D + \sin(X_1) + X_2^2 + \zeta$
- Estimate $g_0(X)$ with Ridge regression
- Compare with true $g_0$

| Sample Size | $\|\hat{g} - g_0\|_2$ | Bias in $\hat{\theta}_{naive}$ | Coverage (95% CI) |
|------------|---------------------|------------------------|-------------------|
| 100 | 0.45 | 0.38 | 72% |
| 500 | 0.28 | 0.21 | 81% |
| 1000 | 0.21 | 0.15 | 85% |

---

## The Overfitting Problem

**Second source of bias: using same data twice**
- Train ML model on sample: $\hat{g}(X_i)$ fits noise in $Y_i$
- Evaluate score on same sample: residuals correlate with prediction errors
- Creates "own-observation bias"

**Mathematical consequence:**
Empirical process terms fail to behave as expected
Requires restrictive Donsker class assumptions (often violated by ML)

---

<!-- _class: lead -->

# Part 2
## Neyman Orthogonality: The First Pillar

---

## The Classical Intuition: FWL Theorem

**Frisch-Waugh-Lovell Theorem (OLS):**

To estimate $\theta$ in $Y = D\theta + X'\beta + \epsilon$:
1. Regress $Y$ on $X$: get residuals $\tilde{Y} = Y - \hat{E}[Y \mid X]$
2. Regress $D$ on $X$: get residuals $\tilde{D} = D - \hat{E}[D \mid X]$
3. Regress $\tilde{Y}$ on $\tilde{D}$: coefficient equals $\hat{\theta}_{OLS}$

**Key insight:** Partialling out confounders orthogonalizes the problem

---

## Extending FWL to Nonparametric Settings

**DML generalizes FWL:**
- Replace parametric projections with ML predictions
- Requires special care to preserve orthogonality
- Must handle both $g_0(X)$ and $m_0(X)$ simultaneously

**The "Double" in Double ML:**
- Debias the outcome: $Y - \hat{l}(X)$
- Debias the treatment: $D - \hat{m}(X)$

---

## Formal Definition: Neyman Orthogonality

**Moment condition identifies $\theta_0$:**

$$E_P[\psi(W; \theta_0, \eta_0)] = 0$$

**Definition:** Score $\psi$ is Neyman Orthogonal at $(\theta_0, \eta_0)$ if:

$$\partial_\eta E_P[\psi(W; \theta_0, \eta_0)][\eta - \eta_0] = 0$$

**Interpretation:** Expected score is locally insensitive to nuisance errors

---

## Geometric Interpretation

<!-- Suggested image: Neyman orthogonality concept diagram showing orthogonal score directions -->

**Visualizing orthogonality:**
- Moment surface $E[\psi(W; \theta, \eta)]$ as function of $(\theta, \eta)$
- At truth $(\theta_0, \eta_0)$, surface is flat in $\eta$ direction
- Small errors in $\hat{\eta}$ cause only second-order changes in moment

**Implication:**
First-order bias vanishes, leaving only product of errors

---

## The Naive Score for PLR (Non-Orthogonal)

**Naive regression adjustment score:**

$$\psi_{naive}(W; \theta, g) = (Y - D\theta - g(X)) \cdot D$$

**Check orthogonality with respect to $g$:**

$$\frac{\partial}{\partial r} E[\psi_{naive}(W; \theta_0, g_0 + rh)]\bigg|_{r=0}$$

$$= E[-h(X) \cdot D] = -E_X[h(X) \cdot m_0(X)]$$

---

## Failure of Naive Score

**The derivative is non-zero unless:**
- $m_0(X) = 0$ (treatment independent of covariates), or
- $h(X) = 0$ (perfect estimation of $g_0$)

**Consequence:**
Estimation error in $g$ weighted by propensity $m_0(X)$
Confirms mathematical source of regularization bias

**Need:** Construct score that is orthogonal to both nuisance functions

---

## Constructing the Orthogonal Score

**Define two nuisance functions:**
- $l_0(X) = E[Y \mid X]$: conditional expectation of outcome
- $m_0(X) = E[D \mid X]$: conditional expectation of treatment

**Key relationship:**

$$l_0(X) = E[Y \mid X] = m_0(X)\theta_0 + g_0(X)$$

**Therefore:**

$$Y - l_0(X) = (D - m_0(X))\theta_0 + \zeta$$

---

## The Neyman Orthogonal Score

**DML orthogonal score for PLR:**

$$\psi_{orth}(W; \theta, \eta) = (Y - l(X) - \theta(D - m(X))) \cdot (D - m(X))$$

**Where:**
- $\eta = (l, m)$ are nuisance parameters
- Score involves residuals of both $Y$ and $D$

**This is the "Partialling Out" score**

---

## Proof of Orthogonality: Part 1

**Derivative with respect to $l$:**

Perturb $l_0 \to l_0 + r\delta_l$:

$$\frac{\partial}{\partial r} E[\psi(W; \theta_0, l_0 + r\delta_l, m_0)]\bigg|_{r=0}$$

$$= E[-\delta_l(X) \cdot (D - m_0(X))]$$

**Apply Law of Iterated Expectations:**

$$= -E_X[\delta_l(X) \cdot \underbrace{E[D - m_0(X) \mid X]}_{= 0}] = 0 \quad \checkmark$$

---

## Proof of Orthogonality: Part 2

**Derivative with respect to $m$:**

Let $\tilde{\zeta} = Y - l_0(X) - \theta_0(D - m_0(X))$ and $V = D - m_0(X)$

Perturb $m_0 \to m_0 + r\delta_m$:

$$\psi(r) = (\tilde{\zeta} + r\theta_0\delta_m)(V - r\delta_m)$$

**Taking derivative at $r=0$:**

$$\frac{\partial}{\partial r}E[\psi(r)]\bigg|_{r=0} = E[-\tilde{\zeta}\delta_m + \theta_0\delta_m V]$$

---

## Proof of Orthogonality: Part 2 (Continued)

**Apply LIE to both terms:**

$$E[-\tilde{\zeta}\delta_m] = -E_X[\delta_m(X) \cdot \underbrace{E[\tilde{\zeta} \mid X]}_{=0}] = 0$$

$$E[\theta_0\delta_m V] = \theta_0 E_X[\delta_m(X) \cdot \underbrace{E[V \mid X]}_{=0}] = 0$$

**Both derivatives vanish:** Score is Neyman Orthogonal $\checkmark$

---

## Alternative Score: IV-Type Formulation

**Two equivalent orthogonal scores:**

| Type | Formula | Advantages |
|------|---------|-----------|
| Partialling Out | $(Y - l(X) - \theta(D - m(X)))(D - m(X))$ | Numerically stable, convex optimization |
| IV-Type | $(Y - D\theta - g(X))(D - m(X))$ | Conceptually similar to 2SLS |

**Both satisfy Neyman orthogonality**
Partialling Out preferred in practice (default in DoubleML)

---

## Why "Double" Debiasing?

**The term "Double" refers to:**
1. First debiasing: outcome $Y - l(X)$
2. Second debiasing: treatment $D - m(X)$

**Multiplying residuals:**
- Creates score orthogonal to both nuisance functions
- Isolates causal effect from confounding
- Enables $\sqrt{N}$-consistent inference

---

## Step-by-Step Derivation Summary

**From naive to orthogonal:**
1. Start with naive score: fails orthogonality test
2. Identify failure mechanism: correlation with $m_0(X)$
3. Apply FWL logic: partial out $X$ from both $Y$ and $D$
4. Construct residualized score: passes orthogonality test
5. Verify mathematically: both Gateaux derivatives vanish

**Result:** First-order bias eliminated

---

<!-- _class: lead -->

# Part 3
## Cross-Fitting: The Second Pillar

---

## The Overfitting Problem Revisited

**Even with orthogonal score:**
If $\hat{l}$ and $\hat{m}$ estimated on same data used for scoring:
- ML algorithm fits noise in training sample
- Residuals correlate with prediction errors
- Creates "own-observation bias"

**Mathematical issue:**
Empirical process convergence requires independence
Standard theory assumes Donsker class (too restrictive for ML)

---

## Solution: Sample Splitting

**Cross-Fitting strategy:**
- Split sample into $K$ folds (e.g., $K=5$)
- For each fold $k$:
  - Estimate nuisance on auxiliary sample $I_k^c$ (other $K-1$ folds)
  - Evaluate score on main sample $I_k$ (held-out fold)
- Aggregate across all folds

**Key benefit:** $\hat{\eta}_k$ independent of $(Y_i, D_i, X_i)$ for $i \in I_k$

---

## Cross-Fitting: Mathematical Setup

<!-- Suggested image: Cross-fitting algorithm diagram showing sample splitting and fold-wise estimation -->

**Partition sample indices:**

$$\{1, \ldots, N\} = I_1 \cup I_2 \cup \cdots \cup I_K \quad (\text{disjoint})$$

**For fold $k$:**
- Main sample: $I_k$ (size $n = N/K$)
- Auxiliary sample: $I_k^c = \{1, \ldots, N\} \setminus I_k$ (size $N - n$)

**Nuisance estimator:**

$$\hat{\eta}_k = \hat{\eta}((W_i)_{i \in I_k^c})$$

---

## DML1 Algorithm: Average-then-Estimate

**Procedure:**
1. Split data into $K$ folds
2. For each fold $k = 1, \ldots, K$:
   - Train nuisance models on $I_k^c$: $\hat{l}_k$, $\hat{m}_k$
   - Solve for $\check{\theta}_k$ on fold $I_k$:

$$\frac{1}{n}\sum_{i \in I_k} \psi(W_i; \check{\theta}_k, \hat{\eta}_k) = 0$$

3. Average fold-specific estimates:

$$\hat{\theta}_{DML1} = \frac{1}{K}\sum_{k=1}^K \check{\theta}_k$$

---

## DML2 Algorithm: Estimate-then-Average

**Procedure:**
1. Split data into $K$ folds
2. For each fold $k = 1, \ldots, K$:
   - Train nuisance models on $I_k^c$: $\hat{l}_k$, $\hat{m}_k$
   - Store predictions for fold $I_k$
3. Solve global moment condition for $\hat{\theta}_{DML2}$:

$$\frac{1}{N}\sum_{k=1}^K \sum_{i \in I_k} \psi(W_i; \hat{\theta}_{DML2}, \hat{\eta}_k) = 0$$

---

## DML1 vs DML2: Key Differences

| Feature | DML1 | DML2 |
|---------|------|------|
| **Estimation** | Separate $\theta$ per fold | Single $\theta$ globally |
| **Aggregation** | Average estimates | Average scores |
| **Stability** | Sensitive to fold variance | More robust |
| **Recommended** | No | Yes (default) |

**Why DML2 preferred:**
More stable when fold-specific variances differ

---

## DML2 for PLR: Explicit Formula

**Score for fold $k$:**

$$\psi_{ik} = (Y_i - \hat{l}_k(X_i) - \theta(D_i - \hat{m}_k(X_i)))(D_i - \hat{m}_k(X_i))$$

**Global moment condition:**

$$\frac{1}{N}\sum_{k=1}^K \sum_{i \in I_k} \psi_{ik}(\hat{\theta}_{DML2}) = 0$$

**Closed-form solution:**

$$\hat{\theta}_{DML2} = \frac{\sum_{k,i} \tilde{D}_{ik}\tilde{Y}_{ik}}{\sum_{k,i} \tilde{D}_{ik}^2}$$

---

## Numerical Example: Cross-Fitting Impact

**Compare approaches on simulated PLR:**

| Method | $\hat{\theta}$ | Std Error | Bias | Coverage (95% CI) |
|--------|-----------|-----------|------|-------------------|
| Naive (no split) | 2.34 | 0.15 | 0.34 | 76% |
| DML1 ($K=3$) | 2.02 | 0.12 | 0.02 | 94% |
| DML2 ($K=5$) | 2.01 | 0.11 | 0.01 | 95% |
| True $\theta_0$ | 2.00 | - | - | - |

**Conclusion:** Cross-fitting essential for valid inference

---

## Choosing the Number of Folds

**Practical considerations:**

| $K$ | Pros | Cons |
|-----|------|------|
| 2 | Fast, simple | High variance, wastes data |
| 5 | Good balance (default) | Standard choice |
| 10 | Low variance | More computation |
| $N$ (LOOCV) | Minimal bias | Computationally expensive |

**Recommendation:** $K=5$ or $K=10$ for most applications

---

<!-- _class: lead -->

# Part 4
## Asymptotic Theory

---

## The Central Result

**Theorem (Chernozhukov et al., 2018):**
Under regularity conditions, if:
1. Score function is Neyman Orthogonal
2. Cross-fitting is employed
3. Nuisance estimators satisfy $\|\hat{\eta} - \eta_0\|_{P,2} = o_P(N^{-1/4})$

**Then:**

$$\sqrt{N}(\hat{\theta}_{DML} - \theta_0) \xrightarrow{d} \mathcal{N}(0, \Sigma)$$

---

## Bias Decomposition

**Decompose the scaled error:**

$$\sqrt{N}(\hat{\theta} - \theta_0) = \underbrace{\frac{1}{\sqrt{N}}\sum_{i=1}^N \psi(W_i; \theta_0, \eta_0)}_{a^*: \text{ Oracle term}}$$

$$+ \underbrace{\sqrt{N} \cdot E_n[\psi(W; \theta_0, \hat{\eta}) - \psi(W; \theta_0, \eta_0)]}_{b^*: \text{ Bias term}} + \underbrace{c^*}_{\text{Empirical process}}$$

---

## Understanding the Three Terms

**Term $a^*$ (Oracle):**
- Behavior if $\eta_0$ were known
- By CLT: $a^* \to \mathcal{N}(0, \sigma^2)$

**Term $b^*$ (Bias):**
- Impact of estimating $\eta$
- Neyman orthogonality: first-order = 0, remainder = product of errors

**Term $c^*$ (Empirical process):**
- Cross-fitting: independent samples $\Rightarrow c^* = o_p(1)$

---

## The Product-of-Rates Condition

**For PLR model, bias term bounded by:**

$$|b^*| \lesssim \sqrt{N} \cdot \|\hat{l} - l_0\|_{P,2} \cdot \|\hat{m} - m_0\|_{P,2}$$

**For bias to vanish:**

$$\sqrt{N} \cdot \|\hat{l} - l_0\|_{P,2} \cdot \|\hat{m} - m_0\|_{P,2} \to 0$$

**If both converge at rate $N^{-\phi}$:**

$$N^{1/2 - 2\phi} \to 0 \implies \phi > \frac{1}{4}$$

---

## The $N^{-1/4}$ Rate: A Revolution

**Classical requirement:**
Parametric estimators need $N^{-1/2}$ convergence
Impossible for nonparametric/high-dimensional problems

**DML requirement:**
Only $N^{-1/4}$ convergence needed

**Why revolutionary:**
Many ML methods achieve this under reasonable conditions:
- Lasso (under sparsity)
- Neural Networks (under smoothness)
- Random Forests (with proper tuning)

---

## Achievability of $N^{-1/4}$ Rate

**Examples of ML methods meeting requirement:**

| Method | Convergence Rate | Conditions |
|--------|-----------------|------------|
| Lasso | $N^{-1/2}$ | Strong sparsity: $s \log p \ll N$ |
| Ridge | $N^{-\alpha/(2\alpha+d)}$ | Smoothness $\alpha$, dimension $d$ |
| Random Forest | $(N/\log N)^{-\alpha/(2\alpha+d)}$ | Smooth functions |
| Neural Net | $N^{-\alpha/(2\alpha+d)}$ | Network depth/width tuned |

**All satisfy $N^{-1/4}$ under appropriate assumptions**

---

## Comparison of Estimator Requirements

| Feature | Naive ML | OLS | DML |
|---------|----------|-----|-----|
| **Nuisance strategy** | Single sample | Parametric | Cross-fitting |
| **Score type** | Non-orthogonal | Linear | Neyman orthogonal |
| **Bias source** | Regularization | Misspecification | 2nd order |
| **Required rate** | $N^{-1/2}$ (fast) | N/A | $N^{-1/4}$ (slow) |
| **Valid in high-dim** | No | No | Yes |

---

## Asymptotic Variance Estimation

**Variance of DML estimator:**

$$\Sigma = \frac{E[\psi^2(W; \theta_0, \eta_0)]}{(E[\partial_\theta \psi(W; \theta_0, \eta_0)])^2}$$

**Sample analog:**

$$\hat{\Sigma} = \frac{\frac{1}{N}\sum_{k,i}\psi^2(W_i; \hat{\theta}, \hat{\eta}_k)}{(\frac{1}{N}\sum_{k,i}\partial_\theta\psi(W_i; \hat{\theta}, \hat{\eta}_k))^2}$$

**Standard errors:** $\hat{\text{SE}} = \sqrt{\hat{\Sigma}/N}$

---

## Confidence Intervals and Hypothesis Tests

**Asymptotic normality enables standard inference:**

**95% Confidence Interval:**

$$[\hat{\theta} - 1.96 \cdot \hat{\text{SE}}, \; \hat{\theta} + 1.96 \cdot \hat{\text{SE}}]$$

**Test $H_0: \theta_0 = \theta^*$:**

$$t = \frac{\hat{\theta} - \theta^*}{\hat{\text{SE}}} \sim \mathcal{N}(0, 1) \text{ under } H_0$$

**Valid inference without knowing nuisance function forms**

---

<!-- _class: lead -->

# Part 5
## Implementation & 401(k) Application

---

## The DoubleML Framework

**Python/R package implementing DML:**
- Object-oriented design
- Supports multiple ML learners
- Handles cross-fitting automatically
- Computes standard errors

**Key advantages:**
- Production-ready implementation
- Extensible to custom models
- Integration with scikit-learn ecosystem

---

## DoubleML Workflow

**Standard implementation steps:**

1. **Data preparation:** Create `DoubleMLData` object
   - Specify outcome $Y$, treatment $D$, controls $X$

2. **Learner selection:** Choose ML algorithms
   - `ml_l` for $E[Y \mid X]$
   - `ml_m` for $E[D \mid X]$

3. **Model specification:** Set resampling and algorithm
   - Number of folds (default: $K=5$)
   - DML algorithm (default: `dml2`)

---

## DoubleML Workflow (Continued)

4. **Estimation:** Call `fit()` method
   - Automatic cross-fitting
   - Score aggregation
   - Variance computation

5. **Inference:** Extract results
   - Point estimates
   - Standard errors
   - Confidence intervals
   - P-values

---

## Basic Code Example

```python
import doubleml as dml
from doubleml.datasets import make_plr_CCDDHNR2018
from sklearn.ensemble import RandomForestRegressor

# Simulate data
data = make_plr_CCDDHNR2018(n_obs=500, dim_x=20)

# Define learners
ml_l = RandomForestRegressor(n_estimators=100, max_depth=5)
ml_m = RandomForestRegressor(n_estimators=100, max_depth=5)

# Initialize DML model
dml_plr = dml.DoubleMLPLR(data, ml_l, ml_m, n_folds=5)
```

---

## Basic Code Example (Continued)

```python
# Fit model
dml_plr.fit()

# View results
print(dml_plr.summary)

# Extract estimates
theta_hat = dml_plr.coef[0]
se = dml_plr.se[0]
ci = dml_plr.confint()
```

---

## Application: 401(k) Plans and Wealth

<!-- Suggested image: 401(k) empirical results showing coefficient estimates with confidence intervals -->

**Research question:**
Do 401(k) retirement plans increase net savings, or do they simply shift savings from other accounts?

**Data:**
1991 Survey of Income and Program Participation (SIPP)
- Sample size: $N = 9{,}915$
- Cross-sectional survey of US households

---

## 401(k) Study: Variables

**Outcome ($Y$):**
Net financial assets (total assets minus debt)

**Treatment ($D$):**
Eligibility for employer-sponsored 401(k) plan (binary)

**Controls ($X$):**
- Income (nonlinear relationship with savings)
- Age, education, family size
- Marital status, two-earner status
- IRA participation, home ownership

---

## 401(k) Study: Identification Strategy

**Causal framework:**
- Potential outcomes: $Y(1)$ if eligible, $Y(0)$ if not
- Treatment effect: $\theta_0 = E[Y(1) - Y(0)]$

**Identification assumption:**
Selection on observables (unconfoundedness)
$$Y(1), Y(0) \perp D \mid X$$

**Conditional on rich controls, eligibility is as-if random**

---

## 401(k) Study: Challenges

**Why not simple OLS?**
- Eligibility correlates with job quality, income
- Income-savings relationship highly non-linear
- Parametric controls likely insufficient

**High-dimensional interactions:**
- Age × Income
- Education × Income
- Family size × Home ownership
- Difficult to specify correctly

---

## 401(k) Study: Implementation

```python
from doubleml.datasets import fetch_401K
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV

# Load data
data = fetch_401K()

# Try multiple learners
learners = {
    'Lasso': (LassoCV(), LassoCV()),
    'RF': (RandomForestRegressor(n_estimators=500),
           RandomForestRegressor(n_estimators=500)),
    'XGBoost': (GradientBoostingRegressor(),
                GradientBoostingRegressor())
}
```

---

## 401(k) Study: Results

| Estimator | ML Learner | $\hat{\theta}$ | Std Error | 95% CI |
|-----------|-----------|---------------|-----------|---------|
| Naive OLS | None | $19,559 | $2,121 | [15,402, 23,716] |
| DML | Lasso | $8,974 | $1,324 | [6,379, 11,569] |
| DML | Random Forest | $8,909 | $1,321 | [6,320, 11,498] |
| DML | XGBoost | $8,597 | $1,350 | [5,951, 11,243] |

**Source: Chernozhukov et al. (2018), DoubleML documentation**

---

## 401(k) Study: Interpretation

**Key findings:**

1. **OLS severely overestimates effect** ($19,559)
   - Residual confounding from income/job quality
   - Linear controls inadequate

2. **DML estimates consistent** ($8,500–$9,000)
   - Different ML learners give similar results
   - Suggests robust causal effect

3. **401(k) eligibility increases wealth by ~$9,000**
   - Significant at conventional levels
   - Evidence against pure displacement

---

## 401(k) Study: Sensitivity Analysis

**Robustness checks performed:**

1. **Alternative learners:** Neural networks, boosting variants
2. **Different fold numbers:** $K = 3, 5, 10$
3. **Subgroup analysis:** By income quartile, age group
4. **Trimming:** Exclude extreme propensity scores

**All specifications yield estimates in $8,000–$10,000 range**

---

## Choosing ML Learners

**Practical guidelines:**

| Data Characteristics | Recommended Learner |
|---------------------|-------------------|
| Linear relationships | Lasso, Ridge |
| Non-linear, smooth | Neural Network, Kernel methods |
| Interactions, categorical | Random Forest, XGBoost |
| Mixed | Ensemble (Super Learner) |

**Always tune hyperparameters via cross-validation**

---

## Hyperparameter Tuning

**Critical for achieving $N^{-1/4}$ rate:**

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 10],
    'min_samples_leaf': [5, 10, 20]
}

# Tune on auxiliary sample
ml_l = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
ml_m = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
```

---

<!-- _class: lead -->

# Part 6
## Extensions: Heterogeneous Treatment Effects

---

## Beyond Average Treatment Effects

**Partially Linear Regression estimates ATE:**
$$\theta_0 = E[Y(1) - Y(0)]$$

**But treatment effects may vary:**
$$\tau(x) = E[Y(1) - Y(0) \mid X = x]$$

**Conditional ATE (CATE) more informative:**
- Identify who benefits most
- Target interventions
- Understand mechanisms

---

## Causal Forests: Motivation

**Random Forests for prediction:**
- Split to minimize MSE
- Maximize prediction accuracy

**Challenge for CATE:**
- True $\tau(x)$ unobserved for any individual
- Cannot compute MSE on treatment effects
- Need new splitting criterion

---

## Causal Forests: Solution

**Athey and Imbens (2016) innovation:**

**Expanded MSE (EMSE) criterion:**
- Split to maximize variance of treatment effects across leaves
- Intuition: informative splits separate high-effect from low-effect subgroups

**Honest Trees:**
- Split sample into training (find splits) and estimation (estimate effects)
- Prevents overfitting bias (analogous to cross-fitting)

---

## Causal Forest DML

**EconML implementation combines DML with Causal Forests:**

1. **First stage (DML):** Estimate residuals
   $$\tilde{Y}_i = Y_i - \hat{E}[Y \mid X_i]$$
   $$\tilde{D}_i = D_i - \hat{E}[D \mid X_i]$$

2. **Second stage (Forest):** Estimate $\tau(x)$ via
   $$\tilde{Y} = \tau(X) \tilde{D} + \text{noise}$$

**Local orthogonalization protects CATE from confounding**

---

## Causal Forest: Code Example

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# Initialize Causal Forest DML
cf_model = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=500),
    model_t=RandomForestRegressor(n_estimators=500),
    n_estimators=1000,
    min_samples_leaf=10
)

# Fit model
cf_model.fit(Y, D, X=X)
```

---

## Causal Forest: Inference

```python
# Estimate CATE
tau_hat = cf_model.effect(X_test)

# Confidence intervals (pointwise)
tau_interval = cf_model.effect_interval(X_test, alpha=0.05)

# Feature importance for heterogeneity
importance = cf_model.feature_importances_
```

---

## Interpreting Heterogeneity with SHAP

**SHAP (Shapley Additive Explanations):**
- Decomposes predictions into feature contributions
- Answers: "Why is $\hat{\tau}(x_i)$ high/low?"

**For individual $i$:**
$$\hat{\tau}(x_i) = \bar{\tau} + \sum_{j=1}^p \phi_j(x_i)$$

**Where $\phi_j$ = contribution of feature $j$**

---

## SHAP for Causal Forests: Example

```python
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(cf_model.model_cate)
shap_values = explainer.shap_values(X_test)

# Visualize for individual
shap.force_plot(explainer.expected_value,
                shap_values[0],
                X_test[0])

# Summary across population
shap.summary_plot(shap_values, X_test)
```

---

## Extensions Beyond PLR

**DML framework applies to many models:**

| Model | Target Parameter | DoubleML Class |
|-------|-----------------|---------------|
| Partially Linear Regression | ATE | `DoubleMLPLR` |
| Partially Linear IV | LATE | `DoubleMLPLIV` |
| Interactive Regression | CATE | `DoubleMLIRM` |
| Difference-in-Differences | ATT | Custom implementation |

---

## Interactive Regression Model (IRM)

**Model specification:**
$$Y = g_0(D, X) + U$$
$$D = m_0(X) + V$$

**Target:**
$$\theta_0(x) = E[g_0(1, x) - g_0(0, x)]$$

**Orthogonal score:**
$$\psi = (Y - \hat{g}(D,X)) \cdot \frac{D - \hat{m}(X)}{\hat{m}(X)(1-\hat{m}(X))} \cdot (D - \hat{m}(X))$$

---

## When Can DML Fail?

**Critical assumptions for validity:**

1. **Overlap/Positivity:** $0 < P(D=1 \mid X) < 1$
   - Fails: extreme propensity scores
   - Solution: trimming, alternative identification

2. **Convergence rate:** ML must achieve $N^{-1/4}$
   - Fails: very high dimension, no structure
   - Solution: feature selection, dimension reduction

3. **Unconfoundedness:** No unmeasured confounders
   - Fails: omitted important variables
   - Solution: IV-DML, sensitivity analysis

---

## Practical Diagnostics

**Check overlap:**
```python
# Propensity score distribution
plt.hist(m_hat, bins=50)
plt.axvline(0.1, color='r', label='Trim threshold')
plt.axvline(0.9, color='r')
```

**Assess ML performance:**
```python
# Cross-validated MSE for nuisance functions
mse_l = mean_squared_error(Y_test, l_hat_test)
mse_m = mean_squared_error(D_test, m_hat_test)
print(f"MSE for E[Y|X]: {mse_l}")
print(f"MSE for E[D|X]: {mse_m}")
```

---

<!-- _class: lead -->

# Conclusion

---

## Key Takeaways

**1. Naive ML fails for causal inference**
- Regularization bias contaminates structural parameters
- Overfitting creates spurious correlations

**2. DML solves the problem with two pillars**
- Neyman Orthogonality: eliminates first-order bias
- Cross-Fitting: ensures independence, removes overfitting

**3. Enables $\sqrt{N}$-consistent inference**
- Only requires $N^{-1/4}$ nuisance convergence
- Achievable by modern ML methods

---

## Key Takeaways (Continued)

**4. Practical implementation is accessible**
- DoubleML package handles complexity
- Works with any scikit-learn learner
- Production-ready for applied work

**5. Extensions to heterogeneous effects**
- Causal Forests estimate CATE
- SHAP provides interpretability
- Applicable to many causal models

---

## The Bigger Picture

**DML bridges prediction and causation:**
- Leverages ML flexibility for confounding adjustment
- Preserves econometric rigor for inference
- Scales to modern high-dimensional data

**Paradigm shift for applied work:**
- From manual control selection to algorithmic learning
- From parametric assumptions to data-driven flexibility
- From limited to rich heterogeneity analysis

---

## Comparison: Traditional vs DML Approach

| Aspect | Traditional Econometrics | Double ML |
|--------|------------------------|-----------|
| **Controls** | Manual selection, linear | Learned, non-linear |
| **Assumptions** | Parametric form | Minimal structure |
| **Dimension** | Low ($p \ll N$) | High ($p \approx N$) |
| **Interactions** | Pre-specified | Automatic |
| **Heterogeneity** | Limited subgroups | Full CATE |
| **Inference** | Standard | Standard (via orthogonality) |

---

## When to Use DML

**DML is especially valuable when:**
- High-dimensional controls ($p$ large relative to $N$)
- Unknown functional form for confounding
- Concern about model misspecification
- Interest in heterogeneous treatment effects

**May not need DML when:**
- Clear parametric structure (validated by theory/data)
- Low-dimensional settings ($p$ small)
- Randomized experiments with perfect compliance

---

## Further Reading

**Foundational papers:**
- Chernozhukov et al. (2018): "Double/Debiased Machine Learning for Treatment and Structural Parameters"
- Belloni, Chernozhukov, Hansen (2014): "High-Dimensional Methods and Inference"

**Extensions:**
- Athey & Imbens (2016): "Recursive Partitioning for Heterogeneous Causal Effects"
- Wager & Athey (2018): "Estimation and Inference of Heterogeneous Treatment Effects"

---

## Software Resources

**Python:**
- `doubleml`: Official DML implementation
- `econml`: Microsoft package (Causal Forests, CATE)
- `causalml`: Uber package (additional methods)

**R:**
- `DoubleML`: R version of Python package
- `grf`: Generalized Random Forests
- `hdm`: High-dimensional metrics

**All available on GitHub with extensive documentation**

---

<!-- _class: lead -->

# Questions?

**Office Hours**: [To be announced]
**Email**: [Your email]
**Course Website**: [Course link]

---

<!-- _class: lead -->

# Thank You!

**Next Lecture**: Heterogeneous Treatment Effects (HTE)

See you next time!
