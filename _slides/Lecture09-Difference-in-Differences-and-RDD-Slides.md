---
marp: true
theme: gaia
paginate: true
header: ''
footer: 'ECON6083 Lecture 9 | Advanced DiD and ML-RDD'
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
    margin-bottom: 1em;
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

# Lecture 9
## Advanced Difference-in-Differences with Heterogeneity and Machine Learning in Regression Discontinuity Designs

**ECON6083: Machine Learning in Economics**

---

## Today's Roadmap

**1. Metrics 2.0 Paradigm Shift**: From TWFE to heterogeneity-robust estimation
**2. TWFE Failure**: Goodman-Bacon decomposition and forbidden comparisons
**3. Callaway & Sant'Anna**: Clean group-time effects and aggregation
**4. DML for DiD**: High-dimensional controls with Neyman orthogonality
**5. ML in RDD**: Covariate adjustment and RDFlex algorithm
**6. Application**: Green Subsidies and Innovation

**Goal**: Understand why traditional methods fail and how ML fixes them

---

<!-- _class: lead -->

# Part 1
## The TWFE Crisis

---

## The Second Credibility Revolution

**Traditional wisdom (1980-2018)**: TWFE regression as default DiD tool
- Unit and time fixed effects control for confounders
- Coefficient interpreted as ATT under parallel trends
- Dominated applied empirical work for 40 years

**The crisis (2018-2021)**: TWFE fundamentally flawed in modern settings
- Staggered adoption + heterogeneous effects = bias
- Non-convex weights, even negative weights possible
- Catastrophic sign reversals documented in published papers

---

## Why This Matters

**New paradigm requirements**:
- Explicit heterogeneity-robust estimators (not implicit weights)
- ML integration for high-dimensional controls
- Transparent aggregation from building blocks to summaries

**Practical impact**:
- Many published DiD results may be invalid
- Need to re-evaluate classic findings with new methods
- Researchers must adopt modern toolkit

---

## The Canonical 2×2 Case

**Setup**: Two periods $t \in \{0, 1\}$, two groups $D \in \{0, 1\}$
**Model**: $Y_{it} = \alpha + \beta D_i + \gamma T_t + \delta (D_i \times T_t) + \varepsilon_{it}$
**Estimator**:
$$\hat{\delta}_{2\times2} = (\bar{Y}_{1,1} - \bar{Y}_{1,0}) - (\bar{Y}_{0,1} - \bar{Y}_{0,0})$$

---

## 2×2 Case: Numerical Example

**Observed means**:
- Treated pre: $\bar{Y}_{1,0} = 100$, Treated post: $\bar{Y}_{1,1} = 120$
- Control pre: $\bar{Y}_{0,0} = 90$, Control post: $\bar{Y}_{0,1} = 95$

**Estimation**: $(120-100) - (95-90) = 20 - 5 = 15$
**Interpretation**: Treatment effect is 15 units (net of common trend of 5)
**Key assumption**: Parallel trends—absent treatment, treated group would have grown by 5 (same as control)

---

## The Staggered Setup

<!-- Suggested image: Staggered adoption timeline showing different treatment cohorts over time -->

**Real-world complications**:
- Treatment timing varies (early, late, never adopters)
- Effects evolve dynamically over time since treatment
- Different cohorts may have different effect sizes

**TWFE model**:
$$Y_{it} = \alpha_i + \lambda_t + \beta^{TWFE} D_{it} + \varepsilon_{it}$$
where $\alpha_i$ = unit FE, $\lambda_t$ = time FE, $D_{it}$ = treatment indicator

---

## What TWFE Actually Estimates

**Historical interpretation**: $\beta^{TWFE}$ as weighted average ATT
- Assumption: Weights are positive and meaningful
- Assumption: All comparisons are valid

**Reality**:
- Weights can be negative (subtracts some effects!)
- Weights are mechanically determined by variance, not economics
- Already-treated units serve as controls (contamination)
- Sign reversals are mathematically possible

---

<!-- _class: lead -->

# Part 2
## Goodman-Bacon Decomposition

---

## The Anatomy of TWFE

<!-- Suggested image: TWFE decomposition diagram showing different 2x2 comparison components -->

**Goodman-Bacon (2021) decomposition**:
$$\hat{\beta}^{TWFE} = \sum_{k \neq U} s_{kU} \hat{\beta}_{kU}^{2 \times 2} + \sum_{k \neq U} \sum_{j > k} \left[ s_{kj} \hat{\beta}_{kj}^{2 \times 2} + s_{jk} \hat{\beta}_{jk}^{2 \times 2} \right]$$

**Key insight**: TWFE is weighted average of all possible 2×2 comparisons

---

## Decomposition Components

**Three types of comparisons**:

**1. Treated vs. Never-Treated** ($\hat{\beta}_{kU}^{2\times2}$):
- Clean comparison (control never contaminated)
- This is what we want!

**2. Early vs. Late-as-Control** ($\hat{\beta}_{kj}^{2\times2}$, $k<j$):
- Late group not yet treated, so valid control
- Still clean

**3. Late vs. Early-as-Control** ($\hat{\beta}_{jk}^{2\times2}$, $j>k$):
- Early group already treated when used as "control"
- **This is the forbidden comparison!**

---

## Variance Weights

**Weight formula**: $s_{kj} \propto n_k n_j \bar{D}(1 - \bar{D})$

**Where weight comes from**:
- Variance of binary variable maximized at mean = 0.5
- Units treated mid-panel have $\bar{D} \approx 0.5$
- Early/late adopters have $\bar{D}$ near 0 or 1

**Implications**:
- Maximum weight on mid-panel treatment groups
- Purely mechanical, unrelated to economic importance
- **Problem**: This weighting scheme favors forbidden comparisons!

---

## Why Variance Weights are Problematic

**Example**: Three groups treated at $t=3, 6, 9$ in 10-period panel
- Group 1: $\bar{D} = 0.7$ (treated 7/10 periods)
- Group 2: $\bar{D} = 0.4$ (treated 4/10 periods)
- Group 3: $\bar{D} = 0.1$ (treated 1/10 period)

**Variance**: $\bar{D}(1-\bar{D})$ = $0.21, 0.24, 0.09$
**Result**: Group 2 gets highest weight despite no economic reason
**Danger**: Group 2 vs Group 1 comparison (forbidden) gets heavy weight

---

## Forbidden Comparisons

**Setup**: Early group $k$ (treated at $t_k$) vs. Late group $l$ (treated at $t_l > t_k$)

**Standard DiD**:
$$\hat{\beta}_{lk}^{2\times2} = (\bar{Y}_{l}^{post} - \bar{Y}_{l}^{pre}) - (\bar{Y}_{k}^{post} - \bar{Y}_{k}^{pre})$$

**Critical issue**: For "control" group $k$, both periods are post-treatment!
- Pre-period for group $l$ is already post-treatment for group $k$
- Group $k$ is evolving due to its own treatment dynamics

---

## Why Forbidden Comparisons Fail

**With static effects** ($\tau_k$ constant over time):
- $\bar{Y}_{k}^{post} - \bar{Y}_{k}^{pre} = \Delta \lambda$ (just time trend)
- DiD correctly recovers $\tau_l^{post}$

**With dynamic effects** ($\tau_k$ grows over time):
- Early group evolves: $\Delta \tau_k = \tau_k^{post} - \tau_k^{pre} > 0$
- $\bar{Y}_{k}^{post} - \bar{Y}_{k}^{pre} = \Delta \lambda + \Delta \tau_k$
- Estimate becomes: $\tau_l^{post} - \Delta \tau_k$ (biased downward!)

---

## The Bias Algebra

**Potential outcomes framework**:
$$Y_{it}(0) = \alpha_i + \lambda_t + \nu_{it}$$
$$Y_{it}(1) = Y_{it}(0) + \tau(e)$$
where $e = t - t^*$ is event time (time since treatment)

**Late vs. Early comparison**:
$$E[\hat{\beta}_{lk}^{2\times2}] = \tau_l^{post} - \Delta \tau_k$$

---

## Numerical Example of Bias

**Setup**: Early treated at $t=2$, Late treated at $t=6$
**Effects evolve**: $\tau(e) = 10 + 5e$ (grows by 5 per year)

**Calculations**:
- Early at $t=5$: $\tau_k = 10 + 5(3) = 25$
- Early at $t=7$: $\tau_k = 10 + 5(5) = 35$, so $\Delta \tau_k = 10$
- Late at $t=7$: $\tau_l = 10 + 5(1) = 15$ (true effect)

**TWFE estimate**: $15 - 10 = 5$ (underestimates by 10!)
**Sign reversal**: If weight is large enough, aggregate $\beta^{TWFE}$ can be negative

---

## Diagnostic in Practice

**Bacon decomposition plot**: Scatter plot of each 2×2 estimate vs. its weight

**Good scenario**:
- Most weight on "Treated vs. Never-Treated"
- Minimal forbidden contamination
- Estimates similar across comparison types

**Bad scenario**:
- Substantial weight on "Late vs. Early" (forbidden)
- Estimates differ systematically across types
- Negative estimates from forbidden comparisons

---

## Example: Green Subsidies

**Context**: Staggered rollout of green finance zones across provinces
**Mechanism**: Subsidies drive R&D, which slowly converts to patents
**Dynamic effects**: Early provinces see accelerating patent growth

**TWFE problem**:
- Using early provinces as controls for late provinces
- Subtracts their accelerating growth from late province effects
- Biases late province effects toward zero or negative
- Aggregate $\beta^{TWFE}$ may show "no effect" or even negative

**Reality**: All provinces benefit, but TWFE masks this truth

---

<!-- _class: lead -->

# Part 3
## Callaway & Sant'Anna Framework

---

## Defining the Target Parameter

**Group-Time ATT**:
$$ATT(g,t) = E[Y_t(1) - Y_t(0) | G_g = 1]$$

where:
- $g$ = cohort first treated at time $g$
- $t$ = calendar time
- $G_g = 1$ indicates unit belongs to cohort $g$

---

## Interpreting $ATT(g,t)$

**Post-treatment** ($t \geq g$):
- Measures effect at time $t$ for cohort $g$
- Example: $ATT(2017, 2020)$ = effect in 2020 for 2017 cohort

**Pre-treatment** ($t < g$):
- Should be zero under parallel trends
- Non-zero values indicate violation
- Serves as placebo test

**Advantage**: Transparent building block (know exactly what you're estimating!)

---

## From Building Blocks to Aggregation

**Philosophy**: Start with elementary $ATT(g,t)$, then aggregate

**Event studies**: Average across cohorts at same event time
**Group-specific**: Average across time for same cohort
**Overall ATT**: Weighted average across all $(g,t)$ pairs

**Key benefit**: Can diagnose heterogeneity before aggregating
- Which cohorts have larger effects?
- How do effects evolve over time?

---

## Identification Assumptions

<!-- Suggested image: Parallel trends visualization showing treated vs control group trajectories -->

**Assumption 1 - No Anticipation**:
$$E[Y_t(0) | G_g = 1] = E[Y_t(0) | G_g = 0] \quad \forall t < g$$

**Meaning**: Units don't change behavior before treatment starts
**Example violation**: Firms boost R&D before subsidy announcement

**Assumption 2 - Conditional Parallel Trends**:
$$E[Y_t(0) - Y_{g-1}(0) | G_g = 1, X] = E[Y_t(0) - Y_{g-1}(0) | C = 1, X]$$

**Meaning**: After conditioning on $X$, counterfactual trends are parallel

---

## Choosing Control Groups

**Option 1: Never-Treated** ($C_{ne}$)
- Cleanest comparison (never contaminated by treatment)
- **Pro**: No concern about treatment dynamics in control
- **Con**: May be small or non-existent in universal rollout

**Option 2: Not-Yet-Treated** ($C_{ny}$)
- Units untreated at $t$ (even if treated later at $g' > t$)
- **Pro**: Maximizes sample size and power
- **Con**: Requires parallel trends between cohorts

**Critical**: Both explicitly avoid already-treated units as controls!

---

## The Doubly Robust Estimator

**Three estimation approaches**:
1. **Outcome Regression (OR)**: Model $E[\Delta Y | X, C=1]$, impute for treated
2. **Inverse Probability Weighting (IPW)**: Reweight by propensity score
3. **Doubly Robust (DR)**: Combine both (recommended)

**DR advantage**: Only need one model correct for consistency

---

## DR Derivation - Step 1

**Propensity score**:
$$p_g(X) = P(G_g=1 \mid X, G_g=1 \cup C=1)$$

**Purpose**: Balance covariate distribution between treated and control
**Estimation**: Logit/Probit on subsample (group $g$ vs. control $C$)
**Output**: $\hat{p}_g(X_i)$ for each unit $i$

---

## DR Derivation - Step 2

**Outcome regression**:
$$m_{g,t}(X) = E[Y_t - Y_{g-1} | C = 1, X]$$

**Purpose**: Predict counterfactual trend for treated units
**Estimation**: OLS of $\Delta Y$ on $X$ using control group only
**Output**: $\hat{m}_{g,t}(X_i)$ predicts trend for treated units

---

## DR Derivation - Step 3

**DR identification formula**:
$$ATT(g,t)^{DR} = E \left[ \frac{G_g}{E[G_g]} (Y_t - Y_{g-1} - m_{g,t}(X)) - \frac{\frac{p_g(X)C}{1-p_g(X)}}{E[\frac{p_g(X)C}{1-p_g(X)}]} (Y_t - Y_{g-1} - m_{g,t}(X)) \right]$$

**First term**: Treated group residuals (actual - predicted)
**Second term**: Control group residuals (reweighted by PS)

---

## Why "Doubly Robust"?

**Scenario 1 - Outcome model correct** ($m(X)$ perfect):
- Residuals $(Y - m(X))$ for controls have conditional mean zero
- Even if propensity score is wrong, DR still consistent
- IPW weights become irrelevant

**Scenario 2 - Propensity score correct** ($p(X)$ perfect):
- Weighting perfectly balances treated/control on $X$
- Even if $m(X)$ misspecified, errors cancel after weighting
- OR specification errors wash out

---

## DR Intuition

**What DR does**:
- Combines regression adjustment and reweighting
- Each protects against failure of the other
- Get "two chances" at consistency

**Practical implication**:
- Robust to moderate misspecification in nuisance parameters
- More credible than relying on single model
- Standard in modern causal inference

**Caveat**: Still fails if both models badly misspecified (DML helps here!)

---

## Aggregation: Event Studies

<!-- Suggested image: Callaway-Sant'Anna event study plot showing dynamic treatment effects over time -->

**Event time aggregation**:
$$\theta_{ES}(e) = \sum_{g} \sum_{t} \mathbf{1}(t-g = e) \cdot P(G=g | t-g=e) \cdot ATT(g,t)$$

**What this does**: Average across all cohorts at same elapsed time $e$
**Weights**: Proportional to group size at that event time

---

## Uses of Event Studies

**1. Visualize dynamic effects**:
- Plot $\theta_{ES}(e)$ against $e$
- See how effect evolves after treatment
- Identify lag, growth, or decay patterns

**2. Test pre-trends**:
- Null hypothesis: $H_0: \theta_{ES}(-1) = \theta_{ES}(-2) = \cdots = 0$
- Wald test for joint significance
- Validates parallel trends assumption

**3. Identify effect maturation**:
- Does effect kick in immediately or gradually?
- Does it plateau or keep growing?

---

## Group-Specific Effects

**Cohort-level aggregation**:
$$\theta(g) = \frac{1}{T - g + 1} \sum_{t=g}^T ATT(g,t)$$

**Interpretation**: Average effect for cohort $g$ over all post-treatment periods

**Why this matters**:
- Test whether early/late adopters have different effects
- Identify saturation or selection effects
- Prioritize groups for policy targeting

---

## Detecting Heterogeneity Across Cohorts

**Example patterns**:

**Early > Late**: Early adopters more responsive
- Possible reason: Selection into early treatment
- Policy implication: Focus on recruiting early

**Late > Early**: Later adopters benefit more
- Possible reason: Learning spillovers from early cohorts
- Policy implication: Stagger rollout to maximize learning

**No difference**: Effect stable across cohorts
- Interpretation: Treatment works uniformly

---

## Comparison: TWFE vs CS2021

| Feature | TWFE | CS2021 |
|---------|------|--------|
| **Target** | Implicit weighted avg | Explicit $ATT(g,t)$ |
| **Controls** | Includes treated | Never/not-yet only |
| **Weights** | Can be negative | Always non-negative |
| **Heterogeneity** | Assumed away | Explicitly modeled |
| **Pre-trends** | Hard to test | Easy (placebo $e<0$) |
| **Dynamics** | Single coefficient | Full event study |

---

## Common Misconceptions

**Myth 1**: "Adding unit and time FE is enough"
- **Reality**: FE don't fix forbidden comparison problem
- Solution requires explicit restriction on control groups

**Myth 2**: "TWFE is conservative (Type I error safe)"
- **Reality**: Can have sign reversals (Type II and Type III errors!)
- Not just low power, but wrong direction

**Myth 3**: "CS2021 only for staggered designs"
- **Reality**: Works for any design, including 2×2
- Always at least as good as TWFE, often better

---

<!-- _class: lead -->

# Part 4
## Double Machine Learning for DiD

---

## The High-Dimensional Challenge

**CS2021 uses parametric models** (logit/OLS) for nuisance parameters
**Modern data realities**:
- Text data from annual reports (TF-IDF features)
- Patent embeddings (hundreds of dimensions)
- Complex interactions between firm characteristics
- $p \approx n$ or even $p > n$

**Traditional approach fails**:
- Logit/OLS doesn't converge or overfits badly
- Cannot capture non-linear relationships

---

## Why Not Just "Plug in" ML?

**Naive approach**: Replace logit with Random Forest, OLS with Lasso

**Problem 1 - Regularization bias**:
- ML methods shrink coefficients to reduce variance
- This bias in nuisance parameters ($\hat{g}, \hat{m}$) propagates to $\hat{\theta}$
- Final estimate systematically biased

**Problem 2 - Overfitting**:
- Train and test on same data means learning noise
- Inference invalid (confidence intervals too narrow)
- Slow convergence: $n^{-1/4}$ instead of parametric $n^{-1/2}$

---

## DML Solution Overview

**Double/Debiased Machine Learning (DML)** solves both problems:

**Solution to regularization bias**:
- Construct Neyman-orthogonal score function
- Makes $\hat{\theta}$ locally insensitive to errors in nuisances
- Achieves $\sqrt{n}$-consistency despite slow ML convergence

**Solution to overfitting**:
- Cross-fitting (like cross-validation)
- Train on one fold, predict on another
- Out-of-sample predictions break overfitting

---

## Neyman Orthogonality

**Core innovation**: Construct score function orthogonal to nuisance parameters

**Orthogonal score for DiD**:
$$\psi(W; \theta, \eta) = \frac{D}{\pi} ( \Delta Y - \theta ) - \frac{D - g(X)}{\pi (1 - g(X))} ( \Delta Y - \ell(X) )$$

where:
- $g(X) = P(D=1|X)$ (propensity score)
- $\ell(X) = E[\Delta Y | D=0, X]$ (outcome regression)
- $\eta = (g, \ell)$ are nuisance parameters

---

## What is Orthogonality?

**Mathematical condition**:
$$\frac{\partial}{\partial \eta} E[\psi(W; \theta_0, \eta)] \bigg|_{\eta = \eta_0} = 0$$

**Intuitive meaning**:
- Score function locally insensitive to nuisance errors
- First-order effect of $\hat{\eta} - \eta_0$ on $\hat{\theta}$ is zero
- Only second-order effects remain (product of errors)

**Why this matters**: Converts slow ML rates into fast parametric rates!

---

## Why Orthogonality Works

**Error decomposition**:
- Without orthogonality: $\hat{\theta} - \theta_0 \approx (\hat{g} - g_0) + (\hat{\ell} - \ell_0)$
- Rate: $O_p(n^{-1/4})$ (slow ML rate)

**With orthogonality**:
- First-order terms cancel by design
- $\hat{\theta} - \theta_0 \approx (\hat{g} - g_0) \times (\hat{\ell} - \ell_0)$ (product!)
- Rate: $n^{-1/4} \times n^{-1/4} = n^{-1/2}$ (parametric!)

---

## Orthogonality: The Magic Trick

**Convergence comparison**:

| Method | $\hat{g}$ rate | $\hat{\ell}$ rate | $\hat{\theta}$ rate |
|--------|---------|----------|----------|
| Plug-in ML | $n^{-1/4}$ | $n^{-1/4}$ | $n^{-1/4}$ (slow) |
| DML (orthogonal) | $n^{-1/4}$ | $n^{-1/4}$ | $n^{-1/2}$ (fast!) |

**Result**: $\sqrt{n}$-consistency and normal CLT for inference
**Practical implication**: Can use ML aggressively without sacrificing valid inference

---

## Cross-Fitting Algorithm

**Purpose**: Break correlation between estimation errors and residuals
**Analogy**: Like cross-validation, but for inference not prediction

**Algorithm**:

**Step 1**: Split data into $K$ folds (e.g., $K=5$)

**Step 2**: For each fold $k$:
- Train $\hat{g}_k(X)$ on other $K-1$ folds (Random Forest/XGBoost)
- Train $\hat{\ell}_k(X)$ on other $K-1$ folds (Lasso/Neural Net)

---

## Cross-Fitting (continued)

**Step 3**: Generate out-of-sample predictions
- For units in fold $k$, predict using models from Step 2
- Get $\hat{g}_k(X_i)$ and $\hat{\ell}_k(X_i)$ for each $i$ in fold $k$

**Step 4**: Calculate orthogonal scores
- Use out-of-sample predictions: $\psi_i(\theta; \hat{g}_k, \hat{\ell}_k)$
- Crucially: prediction errors uncorrelated with residuals

**Step 5**: Solve for $\hat{\theta}$
- Find $\hat{\theta}$ such that $\frac{1}{N} \sum_{i=1}^N \psi_i(\hat{\theta}) = 0$
- Often has closed-form solution

---

<!-- _class: lead -->

# Part 5
## Machine Learning in RDD

---

## The Power Problem in RDD

<!-- Suggested image: RDD sharp discontinuity plot showing outcome jump at cutoff -->

**Standard sharp RDD**:
$$\tau = E[Y(1) | S=c^+] - E[Y(0) | S=c^-]$$

**Advantage**: Robust identification at cutoff
- Requires only local continuity
- Minimal assumptions

**Disadvantage**: Low statistical power
- Estimation is local (uses only data near $c$)
- Discards most of the data
- Large standard errors
- Many true effects appear insignificant

---

## Can Covariates Help in RDD?

**Key question**: Can we use covariates $Z$ to improve precision?

**Challenge**: RDD identifies effect at cutoff only
- Covariates could introduce bias if used naively
- Need careful theory to know what's safe

**Traditional approach** (Calonico, Cattaneo, Titiunik 2019):
- Include $Z$ linearly in regression
- Reduces residual variance if $Y$ and $Z$ correlated
- **Limitation**: Assumes linear relationship

---

## ML Innovation for RDD

**Traditional CCT approach**:
$$Y = \tau D + f(S) + \gamma Z + \epsilon$$

**Benefit**: Reduces residual variance if $Y$ and $Z$ correlated
**Limitation**: Restricts to linear $\gamma Z$ (may miss complex patterns)

**Noack, Olma, Rothe (2024) innovation**:
- Use ML (Random Forest, Neural Net) to capture non-linear relationship
- Optimally "denoise" outcome variable
- No bias from flexible functional form!

---

## Optimal Adjustment Theory

**Define adjusted outcome**:
$$Y_{\eta} = Y - \eta(Z)$$

**Question**: What function $\eta(Z)$ minimizes asymptotic variance of RDD estimator?

**Answer** (Noack et al. 2024):
$$\eta^*(z) = E[Y | Z = z]$$

**Interpretation**: Optimal adjustment is simply conditional expectation!

---

## Intuition for Optimal Adjustment

**What we're doing**:
1. Predict $Y$ using $Z$ globally (using all data, not just near cutoff)
2. Subtract prediction: $\tilde{Y} = Y - \hat{E}[Y|Z]$
3. Apply RDD to residuals $\tilde{Y}$

**Why this works**:
- Removes variation in $Y$ explained by $Z$
- Leaves only "surprise" component
- Jump at cutoff is clearer (higher signal-to-noise)
- Variance of RDD estimator strictly reduced

---

## The "Free Lunch" of ML-RDD

**Key question**: Can we use ML to estimate $\hat{\eta}(Z)$ without bias?

**Theorem (Noack et al. 2024)**: Yes, under cross-fitting!

**Why it works**:
- RDD estimator is local (depends on data at $S \approx c$)
- ML model estimated using global data (all $S$)
- Estimation error in $\hat{\eta}$ is governed by global properties
- This error averages out locally at cutoff
- Effectively orthogonal to local jump estimation

---

## Free Lunch Guarantees

**Result 1 - Consistency**:
- $\hat{\tau}_{ML-RDD} \to \tau$ as $n \to \infty$
- No bias from using ML for $\hat{\eta}(Z)$

**Result 2 - Variance reduction**:
- Asymptotic variance strictly lower than standard RDD
- Can be 50%+ reduction in practice

**Result 3 - Valid inference**:
- Confidence intervals have correct coverage
- Can use standard RDD inference formulas on residuals

**Bottom line**: We get precision gains for free (no cost in bias or validity)!

---

## RDFlex Algorithm

**Step 1**: Split data into Folds A and B (50-50 split)

**Step 2**: Train ML model in Fold A
- Predict $Y$ from $Z$ only (exclude treatment $D$ and running variable $S$!)
- Use Random Forest, XGBoost, Neural Net, etc.
- Output: $\hat{\eta}_A(Z)$

**Why exclude $D$ and $S$?** If we included them, ML would learn the jump at cutoff, removing the very effect we want to estimate!

---

## RDFlex Algorithm (continued)

**Step 3**: Residualize in Fold B
$$\tilde{Y}_i = Y_i - \hat{\eta}_A(Z_i)$$

**Step 4**: Estimate RDD on residuals
- Apply standard local linear regression to $\tilde{Y}_i$ vs. $S_i$ in Fold B
- Estimate jump $\hat{\tau}_B$ at cutoff

**Step 5**: Swap folds and average
- Train in B, test in A, get $\hat{\tau}_A$
- Final estimate: $\hat{\tau}_{ML-RDD} = \frac{1}{2}(\hat{\tau}_A + \hat{\tau}_B)$

---

<!-- _class: lead -->

# Part 6
## Application: Green Subsidies

---

## Empirical Setting

**Research question**: Do government green subsidies drive corporate green innovation?

**Data source**: Chinese manufacturing firms (2010-2023)
- Panel of 5,000+ publicly listed firms
- Annual observations (70,000+ firm-years)

**Treatment**: Firm located in "Green Finance Pilot Zone"
- Zones designated by central government
- Firms in zones receive preferential green loans, tax breaks

---

## Treatment Rollout

**Staggered adoption**:
- **2017**: 5 provinces (Zhejiang, Jiangxi, Guangdong, Guizhou, Xinjiang)
- **2019**: 3 additional provinces (Gansu, Chongqing, Lanzhou)
- **2020**: Further expansion to other regions

**Outcome**: Log(Green Patent Citations + 1)
- Green patents: Environment-related innovations
- Citations: Quality-weighted measure

**Design**: Staggered DiD with high-dimensional controls

---

## Covariates

**Standard firm characteristics**:
- Size (log assets), Leverage, ROA, Age
- State ownership (SOE dummy)
- Industry and region fixed effects

**High-dimensional features**:
- Text from annual reports (TF-IDF of sustainability keywords)
- Patent embeddings (past innovation patterns)
- Network features (supply chain connections)
- **Total**: 200+ covariates

**Challenge**: Traditional methods cannot handle this dimensionality

---

## Step 1: Diagnosing TWFE

**Standard TWFE regression**:
```stata
xtreg green_patents treated i.year, fe cluster(firm)
```

**Result**: $\hat{\beta}^{TWFE} = -0.05$ (SE = 0.02)
- Negative and statistically significant
- Interpretation: Subsidies reduce green innovation?!

**This contradicts theory and case studies**
- Pilot zones designed to boost innovation
- Anecdotal evidence shows positive effects

---

## Bacon Decomposition Results

**Run diagnostic**:
```stata
bacondecomp green_patents treated, ddetail
```

**Findings**:
- 40% of weight from "Late vs. Early" comparisons (forbidden!)
- 35% from "Early vs. Late" (clean)
- 25% from "Treated vs. Never-Treated" (clean)

**Patterns**:
- Clean comparisons show positive effects (+0.15)
- Forbidden comparisons show negative effects (-0.30)
- Early zones have steep patent growth (dynamic effects)

---

## Why TWFE Failed Here

**Mechanism**:
- Subsidies drive cumulative R&D investments
- Patents emerge with 2-3 year lag
- Early zones (2017) show accelerating growth by 2020
- Using them as controls for 2020 cohort subtracts this growth
- Creates spurious negative effect

**Conclusion**: TWFE estimate is invalid
**Action**: Move to heterogeneity-robust methods

---

## Step 2: Callaway & Sant'Anna

**Implementation**:
```R
library(did)
result <- att_gt(yname = "green_patents", tname = "year",
                 idname = "firm_id", gname = "first_treat",
                 control_group = "notyettreated")
```

**Estimation**: $ATT(g,t)$ for each cohort-time pair
**Control group**: Not-yet-treated (maximizes power)

---

## CS2021 Results: Pre-Trends

**Test parallel trends assumption**:

| Event Time | Estimate | Std. Error | p-value |
|------------|----------|------------|---------|
| $e = -3$ | 0.01 | 0.03 | 0.74 |
| $e = -2$ | -0.02 | 0.03 | 0.50 |
| $e = -1$ | 0.00 | 0.02 | 0.99 |

**Interpretation**: Pre-trends flat and insignificant
**Conclusion**: Parallel trends assumption plausible

---

## CS2021 Results: Event Study

**Dynamic effects**:

| Event Time | Estimate | Std. Error | Significance |
|------------|----------|------------|--------------|
| $e = 0$ | 0.01 | 0.03 | ns |
| $e = 1$ | 0.02 | 0.03 | ns |
| $e = 2$ | 0.08 | 0.03 | ** |
| $e = 3$ | 0.15 | 0.04 | *** |
| $e = 4$ | 0.20 | 0.05 | *** |

**Pattern**: Effects emerge with 2-year lag, then grow

---

## Interpreting CS2021 Findings

**Key insights**:

**1. Lag structure**: Immediate impact near zero, kicks in after 2 years
- Consistent with R&D-to-patent pipeline
- Firms need time to invest and innovate

**2. Growing effects**: Year 3+ show strong positive effects
- Cumulative nature of innovation
- Learning and capacity building

**3. TWFE failure explained**: Dynamic effects violated homogeneity
- Early cohorts' growth created forbidden comparison bias
- CS2021 cleanly separates cohorts and event times

---

## Step 3: DML-DiD Robustness

**Concern**: Selection on observables
- Pilot zones chosen based on complex factors
- Technology potential, political connections, past performance
- Linear controls in CS2021 may miss non-linear selection patterns

**Solution**: Apply DML-DiD (Chang 2020)
- 200+ covariates (including text features from reports)
- Random Forest for propensity score $\hat{g}(X)$
- Gradient Boosting for outcome regression $\hat{\ell}(X)$
- Orthogonal score with 5-fold cross-fitting

---

## DML-DiD Results

**Comparison**:

| Method | Estimate | Std. Error | 95% CI |
|--------|----------|------------|---------|
| CS2021 (linear) | 0.15 | 0.04 | [0.07, 0.23] |
| DML-DiD (ML) | 0.12 | 0.03 | [0.06, 0.18] |

**Interpretation**:
- DML estimate slightly smaller but still positive and significant
- Suggests some positive selection, but effect robust
- High-dimensional non-linear controls don't eliminate effect

**Conclusion**: Causal effect persists even after aggressive control for confounding

---

## Step 4: ML-RDD for Subsidy Eligibility

**Additional design element**: Within pilot zones, sharp RDD
- Firms must achieve "Green Score" $S \geq 80$ to receive cash subsidy
- Score based on environmental compliance metrics
- Provides complementary identification strategy

**Standard RDD**:
- Estimate discontinuity at $S = 80$
- Confidence interval: [-0.1, 0.5] (wide, insignificant)
- Large standard error due to noisy outcome

---

## ML-RDD Implementation

**Step 1**: Train XGBoost to predict patents
- Use firm characteristics: size, age, sector, past R&D
- Exclude treatment and running variable
- Get $\hat{\eta}(Z)$ from cross-fitted predictions

**Step 2**: Residualize outcome
$$\tilde{Y}_i = Y_i - \hat{\eta}(Z_i)$$

**Step 3**: Run RDD on residuals
- Local linear regression of $\tilde{Y}$ on $S$ near 80
- Use standard RDD bandwidth selection

---

## ML-RDD Results

**Comparison**:

| Method | Estimate | Std. Error | 95% CI | Significant? |
|--------|----------|------------|---------|--------------|
| Standard RDD | 0.22 | 0.15 | [-0.1, 0.5] | No |
| ML-RDD | 0.25 | 0.08 | [0.1, 0.4] | Yes |

**Standard error reduction**: 47% (from 0.15 to 0.08)
**Power gain**: Detects effect that was hidden by noise

**Interpretation**: Subsidy has localized positive effect at threshold
- ML denoising reveals signal in noisy data

---

<!-- _class: lead -->

# Conclusion

---

## Empirical Synthesis

**Four complementary methods applied to green subsidy question**:

**1. TWFE diagnostic**: Revealed negative bias from forbidden comparisons
**2. CS2021**: Showed positive, growing effects with 2-year lag
**3. DML-DiD**: Confirmed robustness to high-dimensional confounding
**4. ML-RDD**: Provided sharp evidence at eligibility threshold

**Consistent story**: Subsidies work, but require modern methods to detect

---

## Summary of Methods

| Method | Application | Key Innovation | Math Mechanism |
|--------|-------------|----------------|----------------|
| **TWFE (Static)** | Panel Data | Legacy Method | Variance-weighted avg (biased) |
| **CS2021** | Staggered DiD | Heterogeneity-Robust | $ATT(g,t)$ aggregation |
| **DML-DiD** | High-Dim Controls | Bias Correction | Neyman orthogonality |
| **ML-RDD** | RDD Precision | Noise Reduction | Optimal residualization |

---

## When to Use Each Method

**TWFE**: Never use for staggered designs with potential dynamics
- Only valid if: (1) Single treatment timing, or (2) Constant effects
- Always run Bacon decomposition as diagnostic

**CS2021**: Default for staggered DiD
- Use when treatment timing varies
- Provides transparent $ATT(g,t)$ building blocks
- Test pre-trends, visualize dynamics

---

## When to Use Each Method (continued)

**DML-DiD**: Add when you have high-dimensional covariates
- Text data, embeddings, many interactions
- Worried about non-linear confounding
- Want robustness to model misspecification

**ML-RDD**: Use for precision gains in RDD
- When standard RDD has large standard errors
- Have rich covariates correlated with outcome
- Want to maximize power without sacrificing validity

---

## Key Takeaways

**1. TWFE is obsolete for staggered designs**:
- Goodman-Bacon decomposition proves mechanical failure
- Forbidden comparisons create negative weights
- Sign reversals are real, not rare

**2. Callaway & Sant'Anna is the new standard**:
- Transparent $ATT(g,t)$ building blocks
- Clean comparisons (no forbidden controls)
- Valid aggregation to event studies

---

## Key Takeaways (continued)

**3. Machine learning is essential, not optional**:
- DML handles high-dimensional confounding in DiD
- ML-RDD provides safe precision gains in RDD
- **Critical requirement**: Proper orthogonalization and cross-fitting
- Cannot just "plug in" ML naively

**4. Theory enables ML, not replaces it**:
- Modern theory tells us when ML is safe
- Cross-fitting and orthogonality are the keys
- Allows flexible non-parametric modeling with valid inference

---

## Practical Workflow for Staggered DiD

**Phase 1 - Diagnostic**:
1. Run Bacon decomposition on TWFE
2. Plot estimates vs. weights by comparison type
3. Assess contamination severity (% weight on forbidden comparisons)
4. Document why TWFE may fail in your context

**Phase 2 - Heterogeneity-Robust Estimation**:
1. Use CS2021 (`csdid` or `did` package)
2. Choose control group (never-treated vs. not-yet-treated)
3. Estimate $ATT(g,t)$ matrix

---

## Practical Workflow (continued)

**Phase 3 - Validation and Aggregation**:
1. Test pre-trends (joint significance test on $e < 0$)
2. Aggregate to event study plot
3. Check for heterogeneity across cohorts
4. Report overall ATT with appropriate weights

**Phase 4 - Robustness (if high-dimensional covariates)**:
1. Apply DML-DiD with ML for nuisances
2. Use 5-fold cross-fitting
3. Try multiple ML algorithms (RF, XGBoost, Lasso)
4. Verify stability of estimate across algorithms

---

## Practical Workflow for RDD

**Standard RDD**:
1. Plot outcome vs. running variable
2. Test for discontinuity in covariates (placebo)
3. Estimate RDD with optimal bandwidth
4. Report estimate and confidence interval

**ML-RDD (if power is low)**:
1. Identify covariates $Z$ correlated with outcome
2. Train ML model (exclude $D$ and $S$!) with cross-fitting
3. Residualize: $\tilde{Y} = Y - \hat{\eta}(Z)$
4. Run standard RDD on residuals
5. Report both standard and ML-RDD results for transparency

---

## The Metrics 2.0 Revolution

**Old paradigm (1980-2018)**:
- TWFE as default panel DiD estimator
- Linear controls only
- Implicit homogeneity assumptions
- "Black box" regression

**New paradigm (2018-present)**:
- Explicit heterogeneity modeling with $ATT(g,t)$
- Clean comparison units only (no forbidden comparisons)
- ML integration with proper orthogonalization
- Transparent aggregation from building blocks

---

## Why This Matters for Applied Work

**Credibility revolution continues**:
- Not enough to identify variation (quasi-experiment)
- Must also estimate correctly (heterogeneity-robust methods)

**Practical implications**:
- Many published DiD results may need re-evaluation
- Reviewers increasingly demand modern methods
- Replication studies finding opposite signs

**For researchers**:
- Mastering these tools is essential
- Produces robust, transparent causal inference
- Enables credible evidence in age of big data

---

## Software Implementation

**R packages**:
- `did`: Callaway & Sant'Anna implementation
- `bacondecomp`: Goodman-Bacon decomposition
- `DoubleML`: General DML framework
- `rdrobust`: Standard RDD with covariate adjustment

**Stata commands**:
- `csdid`: CS2021 estimator
- `ddtiming`: Event study plots
- `bacondecomp`: TWFE decomposition

**Python**:
- `DoubleML`: Most comprehensive DML package
- `linearmodels`: Panel data with diagnostics

---

## Further Reading

**TWFE Failure**:
- Goodman-Bacon (2021, JoE): "Difference-in-Differences with Variation in Treatment Timing"
- de Chaisemartin & D'Haultfoeuille (2020, AER): "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects"

**Heterogeneity-Robust DiD**:
- Callaway & Sant'Anna (2021, JoE): "Difference-in-Differences with Multiple Time Periods"
- Sun & Abraham (2021, JoE): "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects"

---

## Further Reading (continued)

**DML Framework**:
- Chernozhukov et al. (2018, Econometrics Journal): "Double/Debiased Machine Learning for Treatment and Structural Parameters"
- Chang (2020, arXiv): "Double/Debiased Machine Learning for Difference-in-Differences Models"

**ML in RDD**:
- Noack, Olma, & Rothe (2024, arXiv): "Flexible Covariate Adjustments in Regression Discontinuity Designs"
- Calonico et al. (2019, JASA): "Regression Discontinuity Designs Using Covariates"

---

## Final Thoughts

**The paradigm shift is real**:
- TWFE failure is not a technicality, it's fundamental
- Modern methods are not optional refinements, they're necessary
- ML integration requires theory (orthogonality, cross-fitting)

**What to remember**:
- Always run diagnostics (Bacon decomposition) before trusting TWFE
- Use CS2021 as default for staggered designs
- Apply DML when dimensionality is high
- Leverage ML-RDD for precision gains in RDD

**The future**: More complex treatments, richer data, smarter methods

---

<!-- _class: lead -->

# Questions?

**Office Hours**: [To be announced]
**Email**: [Your email]
**Course Website**: [Course link]

---

<!-- _class: lead -->

# Thank You!

**Next Lecture**: Optimal Policy Learning & Text as Data

See you next time!
