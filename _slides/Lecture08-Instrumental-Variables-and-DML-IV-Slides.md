---
marp: true
theme: gaia
paginate: true
header: ''
footer: 'ECON6083 Lecture 8 | Instrumental Variables & Double Machine Learning'
size: 16:9
style: |
  @import 'default';

  section {
    background: linear-gradient(to bottom, #ffffff 0%, #f8fafc 100%);
    font-family: 'Segoe UI', 'Liberation Sans', sans-serif;
    font-size: 22px;
    padding: 70px 80px;
    color: #1e293b;
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
    line-height: 1.7;
  }

  li {
    margin-bottom: 0.4em;
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

# Lecture 8
## Instrumental Variables & DML-IV

**ECON6083: Machine Learning in Economics**

---

## Today's Roadmap

**1. Endogeneity Problem** - Why causal inference is hard

**2. Classical IV & LATE** - Compliers framework and proof

**3. High-Dimensional Challenge** - Post-double selection

**4. DML-IV Theory** - Neyman orthogonality and cross-fitting

**5. Applications** - Institutions, 401(k), AI demand

**6. Implementation** - Software, diagnostics, best practices

---

<!-- _class: lead -->

# Part 1
## The Endogeneity Problem

---

## What is Endogeneity?

<!-- Suggested image: Endogeneity illustration showing correlation between treatment and error term -->

**Structural equation**: $Y = \theta_0 D + X'\beta + \epsilon$

**Endogeneity**: $D$ correlated with $\epsilon$

**Three sources**:
1. **Omitted variables**: Unobserved confounders
2. **Simultaneity**: $Y$ and $D$ jointly determined
3. **Measurement error**: In $D$, correlated with $\epsilon$

**Consequence**: OLS biased, cannot interpret as causal

---

## The Era of Big Data

**Modern empirical landscape**:
- Not just volume ($n$) but dimensionality ($p$)
- Personalized medicine, targeted marketing, institutional analysis
- Thousands of potential confounders

**When $p \approx n$ or $p > n$**:
- Traditional OLS/2SLS break down
- Overfitting becomes severe
- Cannot select correct functional form

**Need new tools**: Combine ML prediction with causal inference

---

## The ML-Causality Tension

**ML excels at**:
- Prediction with flexible models
- High-dimensional data
- Automatic feature selection

**But for causal inference**:
- Regularization causes systematic bias
- Destroys $\sqrt{n}$-consistency
- Invalidates standard errors

**DML-IV solution**: Marry ML prediction with econometric rigor via orthogonality and cross-fitting

---

<!-- _class: lead -->

# Part 2
## Classical IV & LATE

---

## The Potential Outcomes Framework

**Notation** for individual $i$:
- **Instrument**: $Z_i \in \{0, 1\}$ (e.g., draft lottery)
- **Treatment**: $D_i(z)$ is potential treatment if $Z = z$
- **Observed treatment**: $D_i = Z_i D_i(1) + (1 - Z_i) D_i(0)$
- **Outcome**: $Y_i(d)$ is potential outcome if treatment were $d$
- **Observed outcome**: $Y_i = D_i Y_i(1) + (1 - D_i) Y_i(0)$

**Framework**: Rubin Causal Model with potential outcomes

---

## IV Assumptions

<!-- Suggested image: IV DAG showing Z -> D -> Y with exclusion restriction -->

**Four core assumptions**:

**1. Independence**: $\{Y(d), D(z)\} \perp Z$ (random assignment)

**2. Exclusion**: $Z \to Y$ only through $D$ (most controversial)

**3. Relevance**: $E[D(1)] \neq E[D(0)]$ (strong first stage)

**4. Monotonicity**: $D(1) \geq D(0)$ (no defiers)

---

## Population Types

<!-- Suggested image: LATE compliers framework showing four population types with D(0) and D(1) outcomes -->

**Monotonicity defines four types**:
1. **Compliers**: $D(1)=1, D(0)=0$ (respond to instrument)
2. **Always-takers**: $D(1)=1, D(0)=1$ (take treatment regardless)
3. **Never-takers**: $D(1)=0, D(0)=0$ (never take treatment)
4. **Defiers**: $D(1)=0, D(0)=1$ (ruled out by monotonicity)

**Key insight**: IV identifies effect for **compliers only**
- Not average treatment effect for whole population
- Only for subpopulation affected by instrument

---

## LATE Theorem: Statement

**Local Average Treatment Effect**:

$$\beta_{IV} = \frac{E[Y | Z=1] - E[Y | Z=0]}{E[D | Z=1] - E[D | Z=0]} = E[Y(1) - Y(0) | \text{Complier}]$$

**What IV does NOT identify**:
- Average Treatment Effect for whole population
- Effect on always-takers or never-takers

**What IV DOES identify**:
- Effect on "marginal" individuals moved by instrument
- Often the policy-relevant margin

---

## LATE Proof: Step 1 (Denominator)

**Decompose the denominator (first stage)**:

$$E[D | Z=1] - E[D | Z=0]$$

By independence: $E[D | Z=z] = E[D(z)]$

$$= E[D(1)] - E[D(0)] = E[D(1) - D(0)]$$

Under monotonicity: $D(1) - D(0) \in \{0, 1\}$
- Equals 1 for compliers, 0 for others

**Result**: Denominator = $P(\text{Complier})$

---

## LATE Proof: Step 2 (Numerator)

**Decompose the numerator (reduced form)**:

$$E[Y | Z=1] - E[Y | Z=0]$$

Use $Y = D Y(1) + (1-D) Y(0)$ and independence:

$$= E[D(1) Y(1) + (1-D(1)) Y(0)] - E[D(0) Y(1) + (1-D(0)) Y(0)]$$

Rearranging:
$$= E[(D(1) - D(0))(Y(1) - Y(0))]$$

---

## LATE Proof: Step 3 (Isolate Compliers)

**The term $(D(1) - D(0))$ equals**:
- 0 for always-takers and never-takers
- 1 for compliers

Therefore:
$$\text{Numerator} = E[(Y(1) - Y(0)) \cdot \mathbb{1}\{\text{Complier}\}]$$
$$= P(\text{Complier}) \times E[Y(1) - Y(0) | \text{Complier}]$$

**Final ratio**:
$$\beta_{IV} = \frac{P(\text{Complier}) \times LATE}{P(\text{Complier})} = LATE$$

---

<!-- _class: lead -->

# Part 3
## High-Dimensional Challenge

---

## Conditional IV Assumptions

**In practice**: Assumptions hold conditionally
$$\{Y(d), D(z)\} \perp Z \mid X$$

**Examples**:
- Institution instrument valid conditional on geography
- Draft lottery valid conditional on birth cohort
- 401(k) eligibility valid conditional on income

**When $X$ high-dimensional**:
- Text features, geographic indicators
- Demographic interactions, image embeddings
- Traditional 2SLS breaks down

---

## The Curse of Dimensionality

**Problem with including all variables**:

**When $p \approx n$**: Variance explodes
- Standard errors unreliable
- Weak identification problems multiply

**When $p > n$**: Model not identified
- More parameters than observations
- Cannot invert matrices

**Must perform variable selection or regularization**

---

## Naive Lasso Selection Fails

**Naive approach**: Lasso select predictors of $Y$, then 2SLS

**Why this fails**: Omitted variable bias

**Belloni et al. (2014) insight**:
- Single selection misses important confounders
- Variables weakly correlated with $Y$ but strongly with $D$ or $Z$
- These "moderate" variables are crucial
- Excluding them violates exclusion restriction

---

## Post-Double Selection Lasso

**Solution**: Select from ALL equations

**PDS Algorithm**:
1. Lasso: $Y$ on $X$ → select variables $S_1$
2. Lasso: $D$ on $X$ → select variables $S_2$
3. Lasso: $Z$ on $X$ → select variables $S_3$
4. Union: $S = S_1 \cup S_2 \cup S_3$
5. Run standard 2SLS with variables in $S$

**Key**: Any confounder of outcome, treatment, or instrument is included

**DML generalizes beyond Lasso to any ML method**

---

<!-- _class: lead -->

# Part 4
## DML-IV Theory

---

## Partially Linear IV Model

**DML-IV structural model**:

$$Y = \theta_0 D + g_0(X) + \zeta$$
$$D = r_0(X) + V$$
$$Z = m_0(X) + U$$

**Components**:
- $\theta_0$ = causal effect (target parameter)
- $g_0(X), r_0(X), m_0(X)$ = nuisance functions

**Goal**: Estimate $\theta_0$ with valid inference despite high-dim $X$

---

## Why Naive ML Plugin Fails

**Naive approach**: Estimate $\hat{g}(X)$ with ML, then run IV

**The problem**: Regularization bias

**Scaled error decomposition**:
$$\sqrt{n}(\hat{\theta} - \theta_0) = \text{Normal Limit} + \text{Regularization Bias}$$

**For complex ML**:
- Nuisances converge at $n^{-1/4}$ (not $n^{-1/2}$)
- Bias term: $\sqrt{n} \cdot n^{-1/4} = n^{1/4} \to \infty$
- Destroys $\sqrt{n}$-consistency and invalidates inference

---

## Neyman Orthogonality: Core Idea

**Solution**: Use score function locally insensitive to nuisance errors

**Definition (Neyman Orthogonality)**:
$$\frac{\partial}{\partial \eta} E[\psi(W; \theta_0, \eta)] \bigg|_{\eta=\eta_0} = 0$$

**Interpretation**:
- Moment condition "flat" w.r.t. nuisance parameters
- Small errors in $\hat{\eta}$ have only second-order effect

**Consequence**: Bias becomes second-order
- If $\hat{\eta}$ converges at $n^{-1/4}$
- Then bias: $(\hat{\eta} - \eta_0)^2$ converges at $n^{-1/2}$
- Ensures $\sqrt{n}$-consistency of $\hat{\theta}$

---

## The Orthogonal Score for IV

**For partially linear IV**, the orthogonal score is:

$$\psi(W; \theta, g, m) = (Y - \theta D - g(X)) (Z - m(X))$$

**Intuition**:
- $(Y - \theta D - g(X))$ = residualized outcome
- $(Z - m(X))$ = residualized instrument
- Product-of-residuals forms orthogonal moment

**Robinson (1988) transformation generalized to IV**

---

## Proof of Orthogonality (1/2)

**Claim**: Score $\psi = (Y - \theta D - g(X))(Z - m(X))$ is Neyman orthogonal

**Part A: Derivative w.r.t. $g$**

Perturb $g_0 \to g_0 + \delta h(X)$:

$$\frac{\partial}{\partial \delta} E[(Y - \theta_0 D - g_0 - \delta h)(Z - m_0)] \bigg|_{\delta=0}$$
$$= E[-h(X) (Z - m_0(X))]$$

Let $U = Z - m_0(X)$. By definition: $E[U | X] = 0$

$$= E[E[-h(X) U | X]] = E[-h(X) \cdot 0] = 0 \quad \checkmark$$

---

## Proof of Orthogonality (2/2)

**Part B: Derivative w.r.t. $m$**

Perturb $m_0 \to m_0 + \delta k(X)$:

$$\frac{\partial}{\partial \delta} E[(Y - \theta_0 D - g_0)(Z - m_0 - \delta k)] \bigg|_{\delta=0}$$
$$= E[(Y - \theta_0 D - g_0) (-k(X))]$$

Let $\zeta = Y - \theta_0 D - g_0(X)$. By definition: $E[\zeta | Z, X] = 0$

This implies $E[\zeta | X] = 0$:
$$= E[E[\zeta | X] \cdot (-k(X))] = 0 \quad \checkmark$$

**Conclusion**: Both derivatives vanish → score is orthogonal

---

## Cross-Fitting: Breaking Overfitting

**Remaining problem**: Even with orthogonal scores
- If train on same data used for $\hat{\theta}$
- ML captures sample-specific noise
- Correlation between residuals and errors

**Solution**: Cross-Fitting (Sample Splitting)

**Key idea**:
- Always estimate nuisances on independent sample
- Predict on held-out data
- Breaks correlation between estimation errors and residuals

---

## DML2 Algorithm for IV (1/2)

<!-- Suggested image: DML-IV cross-fitting diagram showing three-stage residualization process -->

**Step 1: Split**
- Randomly partition sample into $K$ folds (e.g., $K=5$)
- Let $I_k$ = observations in fold $k$
- Let $I_k^c$ = complementary training sample

**Step 2: Train** (for each fold $k$)
- Use $I_k^c$ to train ML models:
  - $\hat{g}_k(X)$ predicting $Y$ from $X$
  - $\hat{r}_k(X)$ predicting $D$ from $X$
  - $\hat{m}_k(X)$ predicting $Z$ from $X$

**Step 3: Predict** (on held-out fold $I_k$)
- Apply trained models to data in $I_k$

---

## DML2 Algorithm for IV (2/2)

**Step 4: Residualize** (for each $i \in I_k$)

$$\tilde{Y}_i = Y_i - \hat{g}_k(X_i)$$
$$\tilde{D}_i = D_i - \hat{r}_k(X_i)$$
$$\tilde{Z}_i = Z_i - \hat{m}_k(X_i)$$

**Step 5: Estimate** (pool all residuals)

$$\hat{\theta} = \frac{\sum_{k=1}^K \sum_{i \in I_k} \tilde{Z}_i \tilde{Y}_i}{\sum_{k=1}^K \sum_{i \in I_k} \tilde{Z}_i \tilde{D}_i}$$

**Result**: $\sqrt{n}$-consistent with asymptotically normal distribution

---

## DML: The Complete Solution

**Two innovations work together**:

**1. Neyman Orthogonal Score**
- Immunizes against regularization bias
- Product-of-errors: $n^{-1/4} \times n^{-1/4} = n^{-1/2}$
- Restores parametric convergence rate

**2. Cross-Fitting**
- Breaks overfitting correlation
- Ensures independence between nuisance estimates and residuals
- Uses all data efficiently

**Resulting properties**: $\sqrt{n}$-consistency, valid confidence intervals, works with any ML method

---

<!-- _class: lead -->

# Part 5
## Empirical Applications

---

## Application Overview

**Three applications demonstrating DML-IV**:

**1. Institutions and Growth (AJR 2001)**
- Classic development economics
- High-dimensional geography controls
- Weak instrument diagnostics

**2. 401(k) Eligibility and Wealth**
- Household finance application
- Non-linear income confounding
- LATE interpretation for marginal savers

**3. AI-Driven Demand Analysis**
- Frontier application with unstructured data
- Image/text embeddings as controls
- Price elasticity estimation

---

## Case 1: Institutions and Growth (1/3)

**Context**: Acemoglu, Johnson, Robinson (2001)
- Does geography or institutions drive long-run growth?
- One of most influential development papers

**Identification strategy**:
- Outcome ($Y$): Log GDP per capita
- Treatment ($D$): Institutional quality
- Instrument ($Z$): Log settler mortality
- Controls ($X$): Geography variables

**Intuition**: High mortality → extractive institutions → low growth

---

## Case 1: Institutions and Growth (2/3)

**The high-dimensional challenge**:
- "Geography" is complex and multifaceted
- Latitude, distance to coast, soil, temperature
- Potentially infinite interactions and polynomials
- Linear controls likely insufficient
- Sample size: $n = 64$ countries

**DML-IV application**:
- Generate high-dim geography features ($p \approx 20-100$)
- Latitude polynomials, continent interactions, distance measures
- Use Random Forest and Lasso for flexible modeling

---

## Case 1: Institutions and Growth (3/3)

**Results**:

| Method | $\hat{\theta}$ | SE | First Stage t-stat |
|--------|---------------|-----|-------------------|
| OLS (naive) | 0.52 | 0.06 | — |
| Original 2SLS | 0.94 | 0.16 | 4.18 |
| DML (RF) | 0.88 | 0.32 | -1.86 |
| DML (Lasso) | 0.78 | 0.28 | -2.88 |

**Key findings**:
- DML estimates close to original (~0.9), reinforces hypothesis
- But weak instrument after aggressive controls (t-stats borderline)
- AR-robust CI: [0.44, 1.74] (wider but still excludes zero)

---

## Case 2: 401(k) and Wealth (1/3)

**Context**: Do tax-deferred savings plans increase net wealth?
- Or merely shift savings across accounts?
- Classic LATE problem in household finance

**Setup**:
- Outcome ($Y$): Net financial assets
- Treatment ($D$): 401(k) participation
- Instrument ($Z$): 401(k) eligibility
- Confounder ($X$): Income (primary confounder)

**Why income confounds**: Eligibility and savings both correlate with high income (non-linearly)

---

## Case 2: 401(k) and Wealth (2/3)

**DML-IV implementation**:
- Model nuisances with Neural Networks and Gradient Boosting
- Capture non-linear income-wealth relationship
- Flexible modeling of eligibility-income relationship

**Results comparison**:

| Method | Estimated Effect | Interpretation |
|--------|-----------------|----------------|
| Naive OLS | $19,559 | Severely biased (selection) |
| Standard 2SLS | $9,000-10,000 | Assumes linearity |
| DML-IV | $8,000-9,000 | Flexible non-linear controls |

---

## Case 2: 401(k) and Wealth (3/3)

**LATE interpretation**:
- DML-IV estimates effect for **compliers**
- Those induced to save by eligibility
- NOT effect on always-participants

**Compliers in this context**:
- Marginal savers, moderate income
- Less financially sophisticated
- Most policy-relevant population

**Policy implications**: $8,000-9,000 increase suggests subsidies effective for marginal savers with limited crowd-out

---

## Case 3: AI Demand Analysis (1/4)

**Context**: Chernozhukov et al. (2024) - "Adventures in Demand"
- Frontier application to unstructured data
- Estimate price elasticity for Amazon products
- Use AI to control for unobserved quality

**Endogeneity problem**:
- Classic: $\text{Price} \leftarrow \text{Quality} \rightarrow \text{Demand}$
- High-quality products command higher prices
- Naive elasticity biased toward zero

**Innovation**: Use product images and text as quality proxies

---

## Case 3: AI Demand Analysis (2/4)

**Methodology**:

**Step 1**: Generate embeddings from unstructured data
- Product images → Vision Transformer (CLIP)
- Text descriptions → Language Model embeddings
- High-dimensional vector $V$ (768-dim)

**Step 2**: Treat embeddings as controls $X = V$
- Captures aesthetic appeal, features, brand
- Dimensions not individually interpretable
- But collectively proxy for quality

**Step 3**: Apply DML-IV with embeddings

---

## Case 3: AI Demand Analysis (3/4)

**Results**:

**Prediction performance**:
- Without embeddings: $R^2 \approx 50\%$ for sales rank
- With embeddings: $R^2 \approx 56-60\%$
- Embeddings capture substantial quality variation

**Elasticity estimates**:
- Naive (no quality controls): $\epsilon \approx -0.8$
- DML with embeddings: $\epsilon \approx -2.3$ to $-2.8$
- Standard models severely underestimate price sensitivity

**Why**: Omitting quality conflates price effect with quality effect

---

## Case 3: AI Demand Analysis (4/4)

**Heterogeneity analysis**:
- Massive heterogeneity across product types
- "Luxury" goods: $\epsilon \approx -1.5$
- "Generic" goods: $\epsilon \approx -3.5$
- Embeddings enable rich heterogeneity analysis

**Broader implications**:
- DML can utilize non-tabular data (images, text)
- Solves classic identification problems with modern data
- Applications: Real estate (images), labor (resumes), trade (appearance)

---

<!-- _class: lead -->

# Part 6
## Implementation Guide

---

## Choosing the Machine Learner

**Lasso** (L1-regularized regression):
- Best for sparse signals (few variables matter)
- Fast and interpretable
- Good default for tabular data

**Random Forests / Gradient Boosting**:
- Best for dense, non-linear signals
- Captures complex interactions automatically
- Computationally intensive

**Neural Networks**:
- Essential for unstructured data (images, text)
- Requires large sample sizes
- Pre-trained models extend applicability

---

## The Role of Stacking

**Problem**: Different learners excel in different settings

**Solution**: Stacking (Ensemble Learning)
- Train multiple learners (Lasso, RF, OLS)
- Combine predictions via weighted average
- Weights chosen via cross-validation

**Advantages**:
- Robust to learner misspecification
- If Lasso fails but RF succeeds, stack leans on RF
- Recommended as safe default

**Software**: `DoubleML` supports stacking out-of-box

---

## Weak Instruments in DML

<!-- Suggested image: Weak instrument diagnostic plot showing F-statistic distribution -->

**Challenge**: Flexible controls can weaken instruments
- Partialling out $X$ aggressively removes variation
- Classic weak IV problems re-emerge

**Diagnostic**: Residual First Stage
1. Obtain residuals $\tilde{D}$ and $\tilde{Z}$ from cross-fitting
2. Regress $\tilde{D}$ on $\tilde{Z}$: $\tilde{D} = \gamma \tilde{Z} + \nu$
3. Check F-statistic (rule: $F < 10$ indicates weak)

**If weak detected**: Use Anderson-Rubin confidence intervals

---

## Anderson-Rubin Confidence Intervals

**AR method**: Inverts test statistic to find plausible $\theta$ values

**Test statistic** for hypothesis $H_0: \theta = \theta^*$:
$$AR(\theta^*) = \frac{(\tilde{Y} - \theta^* \tilde{D})' P_{\tilde{Z}} (\tilde{Y} - \theta^* \tilde{D})}{\hat{\sigma}^2}$$

**AR confidence interval**:
- Set of all $\theta^*$ such that $AR(\theta^*) < c_\alpha$
- Robust to weak instruments
- Typically wider than standard intervals

**Software**: `DoubleML` and `ivmodel` packages support AR inference

---

## Software Ecosystem

**R packages**:
- `DoubleML`: Comprehensive DML implementation
- `hdm`: High-dimensional metrics (Lasso-focused)
- `grf`: Generalized Random Forests with causal variants
- `ivmodel`: Anderson-Rubin and weak IV diagnostics

**Python packages**:
- `DoubleML`: scikit-learn compatible, full DML suite
- `EconML` (Microsoft): Broader causal ML toolkit
- Integrates with sklearn, xgboost

**Stata**: `ddml`, `pdslasso` (less flexible but user-friendly)

---

## Practical Workflow Checklist

**1. Data preparation**
- Split $Y$, $D$, $Z$, $X$; check missing values; standardize

**2. Choose learners**
- Multiple candidates (Lasso, RF, Boosting)
- Consider stacking for robustness

**3. Cross-fitting setup**
- Choose $K$ (typically 5 or 10)
- Ensure sufficient sample size per fold

**4. Estimation**
- Run DML algorithm
- Obtain $\hat{\theta}$ and standard error

---

## Workflow Checklist (continued)

**5. Diagnostics**
- Check nuisance prediction quality ($R^2$, RMSE)
- Poor prediction → consider different learner
- Residual first-stage check for weak instruments

**6. Inference**
- If first-stage strong: Standard DML confidence intervals
- If first-stage weak: Anderson-Rubin intervals
- Report both for transparency

**7. Robustness**
- Vary learner choice, number of folds
- Sensitivity analysis

---

## Common Pitfalls to Avoid

**1. Not checking overlap**: Require $0 < P(Z=1|X) < 1$

**2. Ignoring weak instruments**: Always check residual F-stat

**3. Using wrong score function**: PLR vs. LATE vs. IRM differ

**4. Insufficient sample size**: Rule of thumb $n > 200$, ideally $> 1000$

**5. Not validating nuisance predictions**: Check out-of-sample quality

---

## Key Takeaways

**1. IV identifies LATE** (compliers), not ATE
- Requires Independence, Exclusion, Relevance, Monotonicity

**2. High-dimensional controls create tension**
- Need flexible modeling to avoid bias
- Post-double selection addresses this

**3. DML-IV bridges ML and causal inference**
- Neyman orthogonality eliminates regularization bias
- Cross-fitting breaks overfitting correlation

---

## Key Takeaways (continued)

**4. Applications span classical and frontier**
- Classical: Institutions-growth, 401(k)-wealth
- Frontier: AI embeddings for demand estimation
- Unstructured data now usable as controls

**5. Implementation requires care**
- Choose learner appropriate to data structure
- Always diagnose weak instruments
- Use Anderson-Rubin for robust inference when needed

**6. DML is not a silver bullet**
- Still requires valid instrument and conditional independence

---

<!-- _class: lead -->

# Questions?

**Office Hours**: [To be announced]
**Email**: [Your email]
**Course Website**: [Course link]

---

<!-- _class: lead -->

# Thank You!

**Next Lecture**: Difference-in-Differences & RDD with ML

See you next time!
