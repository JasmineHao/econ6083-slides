---
marp: true
theme: gaia
paginate: true
header: ''
footer: 'ECON6083 Lecture 6 | Heterogeneous Treatment Effects'
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

# Lecture 6
## Heterogeneous Treatment Effects and Causal Machine Learning

**ECON6083: Machine Learning in Economics**

---

## Today's Roadmap

**1. Why Heterogeneity?**: Limitations of ATE and policy targeting
**2. Causal Trees**: EMSE criterion and honest estimation
**3. Causal Forests**: Asymptotic normality and inference
**4. GRF Framework**: Generalized moment equations
**5. DML Integration**: R-Learner and Neyman orthogonality
**6. Implementation**: EconML, SHAP, and policy learning

**Goal**: Move from average effects to precision policy design

---

<!-- _class: lead -->

# Part 1
## Why Care About Heterogeneity?

---

## The Limits of Average Treatment Effects

<!-- Suggested image: Heterogeneity motivation example showing distribution of individual treatment effects -->

**Traditional causal inference goal**: Estimate ATE
$$\tau = E[Y_i(1) - Y_i(0)]$$

**What ATE tells us**: Population-level summary of intervention impact
**What ATE hides**: Individual variability and subgroup differences

**Problem**: Single "average" effect masks crucial heterogeneity
- Who benefits most from job training?
- Which patients respond to the drug?
- Where should we target the policy?

---

## Policy Targeting and Triage

<!-- Suggested image: Policy targeting example showing optimal allocation based on predicted treatment effects -->

**Resource allocation problem**: Budget constraint with heterogeneous effects

**Example: Training vouchers**
- ATE = $2,000 earnings gain (cost = $5,000)
- Seems inefficient at population level
- But some workers gain $10,000, others gain $0

**Optimal policy**: Target high-effect individuals
- Improves cost-effectiveness
- Requires estimating $\tau(x)$ not just $\tau$

---

## The CATE: Our Target Parameter

**Conditional Average Treatment Effect**:
$$\tau(x) \equiv E[Y_i(1) - Y_i(0) \mid X_i = x]$$

where $x$ is a vector of pre-treatment covariates

**Interpretation**: Average effect for subpopulation with characteristics $x$
**Example**: Effect of drug for males aged 40-50 with high blood pressure

---

## Simpson's Paradox: A Cautionary Tale

**Medical study**: Open surgery (A) vs. Closed surgery (B) for kidney stones

| Stone Size | Treatment A | Treatment B | Winner |
|------------|-------------|-------------|--------|
| Small | 93% (81/87) | 87% (234/270) | A better |
| Large | 73% (192/263) | 69% (55/80) | A better |
| Combined | 78% (273/350) | 83% (289/350) | B better! |

**Paradox**: A wins in both subgroups but loses overall
**Source**: Charig et al. (1986), analyzed by Athey & Imbens

---

## Why Simpson's Paradox Happens

**Mechanism**: Stone size is a confounder
- Doctors assign severe cases (large stones) to Treatment A
- Treatment A handles harder cases, lowering overall success rate
- Treatment B benefits from easier case mix

**Within groups**: Treatment A consistently superior
**Aggregate**: Treatment B appears superior (misleading!)

**Lesson**: $\tau(x)$ is not just refinement, it's necessary for correctness
- Failing to condition can reverse sign of effect
- ATE may be fundamentally wrong without heterogeneity analysis

---

## The Missing Data Problem

**Fundamental problem of causal inference**: Cannot observe both outcomes

**What we want**: $\tau_i = Y_i(1) - Y_i(0)$ for each unit
**What we observe**: $Y_i^{obs} = W_i Y_i(1) + (1-W_i) Y_i(0)$

**In supervised learning**: Loss function $E[(Y_i - \hat{\mu}(X_i))^2]$
- We observe true label $Y_i$
- Can compute and minimize prediction error

**In causal inference**: Cannot compute $(\tau_i - \hat{\tau}(X_i))^2$
- Individual effects $\tau_i$ never observed
- Standard ML algorithms fail completely

---

## Why Standard ML Fails

**Standard approach**: Use data to find patterns maximizing predictive accuracy
- Random Forest: Split to minimize outcome variance
- Gradient Boosting: Iteratively reduce residuals

**Problem in causal setting**: Adaptivity creates spurious heterogeneity
- Algorithm searches for subgroups with large $\hat{\tau}$
- Inevitably finds noise masquerading as signal
- "Winner's curse": Discovered effects don't replicate

---

## The Failure of Adaptivity

**Two fatal flaws**:

**1. Systematic bias**: Estimates biased away from zero
- Partitions selected because they show extreme values in training data
- Sampling noise treated as real signal
- Estimates don't generalize to population

**2. Invalid inference**: Confidence intervals assume fixed model structure
- Don't account for search over partition space
- Coverage rates far below nominal (e.g., 95% CI covers only 70%)
- False positives in detecting heterogeneity

**Solution**: Honest estimation (next part)

---

<!-- _class: lead -->

# Part 2
## Causal Trees and Honest Estimation

---

## From Standard Trees to Causal Trees

<!-- Suggested image: Causal tree split example showing treatment effect heterogeneity across nodes -->

**Standard CART**: Minimize outcome variance
$$\text{Criterion} = \min \sum_{\ell \in \Pi} \sum_{i \in \ell} (Y_i - \bar{Y}_\ell)^2$$

**Problem**: Splits based on $Y$ correlation, not treatment effect

**Causal Tree**: Maximize treatment effect heterogeneity
$$\text{Criterion} = \max \text{Variance of } \hat{\tau}(x) \text{ across leaves}$$

**Innovation**: Split to find where effects differ, not where outcomes differ

---

## EMSE Splitting Criterion: Motivation

**Goal**: Minimize Mean Squared Error of treatment effect estimates
$$MSE_\tau = E[(\hat{\tau}(X_i) - \tau(X_i))^2]$$

**Problem**: $\tau(X_i)$ is unobserved, cannot compute MSE directly

**Solution**: Derive unbiased estimator for this risk
- Expected Mean Squared Error (EMSE)
- Computable from observed data
- Accounts for estimation variance

---

## EMSE Derivation: Step 1

**Expand the quadratic**:
$$(\tau_i - \hat{\tau}(X_i))^2 = \tau_i^2 - 2\tau_i \hat{\tau}(X_i) + \hat{\tau}(X_i)^2$$

**Key insight**: $\tau_i^2$ doesn't depend on our partition $\Pi$

**Equivalent objective**: Maximize
$$2\tau_i \hat{\tau}(X_i) - \hat{\tau}(X_i)^2$$

This is the starting point for deriving a feasible criterion

---

## EMSE Derivation: Step 2

**Address the cross-term**: $\tau_i \hat{\tau}(X_i)$ contains unobservable $\tau_i$

**Under honest estimation**: $\hat{\tau}(X_i)$ estimated on independent sample

**Key property**: Within leaf $\ell$, assuming unbiased estimation
$$E[\tau_i \cdot \hat{\tau}_\ell] = \tau_\ell \cdot \hat{\tau}_\ell$$

**This gives us**:
$$E[2\tau_i \hat{\tau}_\ell - \hat{\tau}_\ell^2] = 2\hat{\tau}_\ell^2 - \hat{\tau}_\ell^2 = \hat{\tau}_\ell^2$$

So maximizing $\hat{\tau}^2$ is approximately correct

---

## EMSE Derivation: Step 3

**Problem with naive $\hat{\tau}^2$ criterion**: Overfitting
- Leaves with high variance randomly produce large $\hat{\tau}^2$
- Algorithm exploits noise

**Solution**: Penalize estimation variance

**Within leaf $\ell$**, variance of difference-in-means:
$$\hat{V}_\ell = \frac{S^2_{treat}}{N_{treat}} + \frac{S^2_{control}}{N_{control}}$$

where $S^2$ is sample variance of outcome

---

## Final EMSE Criterion

**Unbiased estimator for EMSE**:
$$-\widehat{EMSE}_\tau(\Pi) = \frac{1}{N} \sum_{i} \hat{\tau}^2(X_i) - \left(\frac{1}{N_{tr}} + \frac{1}{N_{est}}\right) \sum_{\ell \in \Pi} \hat{V}_\ell$$

**First term**: Reward heterogeneity (large treatment effects)
**Second term**: Penalize variance (imprecise estimates)

**Interpretation**: Only split if heterogeneity gain exceeds precision loss

---

## EMSE: Numerical Example

**Parent node**: $N=100$, $\hat{\tau}_{parent} = 0.1$, $\hat{V}_{parent} = 0.04$

**Proposed split into two children**:
- Left: $N=40$, $\hat{\tau}_L = 0.3$, $\hat{V}_L = 0.06$
- Right: $N=60$, $\hat{\tau}_R = -0.05$, $\hat{V}_R = 0.05$

**Heterogeneity term**: $0.4 \times 0.3^2 + 0.6 \times 0.05^2 = 0.0375$
**Parent**: $0.1^2 = 0.01$
**Gain**: $0.0375 - 0.01 = 0.0275$

**Variance penalty**: $0.06 + 0.05 - 0.04 = 0.07$
**Net**: $0.0275 - 0.005 \times 0.07 < 0$ (don't split if weight too high)

---

## Honest Estimation: The Core Concept

**Honesty**: Separation of information for structure vs. estimation

**Random sample split**:
- **Splitting sample** ($S_{tr}$): Determine tree structure (where to split)
- **Estimation sample** ($S_{est}$): Estimate $\hat{\tau}$ in each leaf

**Critical property**: Estimation data never influences partition choice
- No data snooping
- No selection bias

---

## Why Honesty Eliminates Bias

**Adaptive (dishonest) estimation**:
- Use all data for both splitting and estimation
- Algorithm selects split $j$ because it showed large $\hat{\tau}$ in sample
- Includes noise component $\epsilon$ in selection criterion
- Same data used for estimation inherits this noise
- Result: $E[\hat{\tau} | \text{split selected}] \neq \tau$ (biased!)

**Honest estimation**:
- Noise $\epsilon^{tr}$ influences structure selection
- Noise $\epsilon^{est}$ in estimation sample is independent
- Conditional on partition $\Pi$: $E[\hat{\tau}_\ell | \Pi] = \tau_\ell$ (unbiased!)

---

## Honest vs. Adaptive: Visual Analogy

**Adaptive approach**: Like a student who
- Sees exam questions
- Studies only those topics
- Takes the same exam
- Appears to know everything (overfits!)

**Honest approach**: Like a student who
- Sees practice problems (splitting sample)
- Studies relevant topics
- Takes different exam (estimation sample)
- True knowledge assessed fairly

**Result**: Honest estimates have valid confidence intervals

---

## The Cost of Honesty

**Bias-variance tradeoff**:

**Honesty benefits**:
- Unbiased estimates conditional on structure
- Valid confidence intervals
- Nominal coverage (95% CI actually covers 95%)

**Honesty costs**:
- Discards half the data for structure learning
- Shallower trees (less adaptive)
- Higher MSE in finite samples (potentially)

**Wager & Athey conclusion**: Bias reduction essential for inference
- Honesty is default for inference-focused applications

---

<!-- _class: lead -->

# Part 3
## Causal Forests

---

## From Trees to Forests

**Single tree problem**: High variance
- Small change in data can alter entire structure
- Unstable predictions

**Causal Forest solution**: Ensemble of honest causal trees
1. Bootstrap: Random sample with replacement
2. Random features: Each split considers random subset of features
3. Honesty: Each tree uses honest splitting
4. Aggregation: Average predictions across trees

**Result**: Smooth, stable, powerful estimates

---

## Causal Forests as Adaptive Kernels

**Forest prediction**: Weighted average of training outcomes
$$\hat{\tau}(x) = \sum_{i=1}^n \alpha_i(x) Y_i^{obs}$$

**Weight definition**: Frequency of co-occurrence in leaves
$$\alpha_i(x) = \frac{1}{B} \sum_{b=1}^B \frac{\mathbb{I}(X_i \in L_b(x))}{|L_b(x)|}$$

where $L_b(x)$ is leaf containing $x$ in tree $b$

**Interpretation**: Forest learns "causal proximity" kernel
- Points are similar if they share treatment effect structure
- Unlike Euclidean distance (standard kernels)

---

## Asymptotic Normality: The Goal

**Key theoretical result (Wager & Athey 2018)**:
$$\hat{\tau}(x) \xrightarrow{d} N(\tau(x), \sigma^2(x))$$

**Why this matters**:
- Valid confidence intervals around predictions
- Hypothesis testing for heterogeneity
- Moves Random Forests from black-box to rigorous statistical tool

**Requirements**: Three technical conditions

---

## Condition 1: Honesty

**Requirement**: Trees must use honest splitting
- Outcome $Y$ not used for splitting if used for estimation
- Decouples dependency structure

**Why necessary**: Without honesty, estimates biased
- Asymptotic distribution incorrect
- Confidence intervals invalid

**Implementation**: Sample splitting within each tree

---

## Condition 2: Regularity

**Requirement**: Tree structure must "localize" around $x$

**Technical definition**: As sample grows, probability that a leaf doesn't get further split must decrease

**Intuition**: Leaves shrink in diameter
- Target increasingly local neighborhoods
- Captures smoothness of $\tau(x)$

**Practical implication**: Don't set `min_leaf_size` too large

---

## Condition 3: Subsampling Rate

**Most critical technical condition**:
$$s_n = n^\beta \quad \text{where } \beta < 1$$

**Meaning**: Subsample size must grow slower than full sample
- Implies $s_n / n \to 0$ as $n \to \infty$

**Why necessary**: Ensures trees sufficiently uncorrelated
- Overlap between subsamples decreases
- Variance decays at correct rate for CLT

**If $s_n \approx n$**: Trees too correlated, CLT fails

---

## Why Subsampling Enables Gaussianity

**Standard bootstrap** ($s_n = n$):
- Trees highly correlated
- Variance reduction from averaging limited
- Central Limit Theorem doesn't apply

**Subsampling** ($s_n = n^{0.7}$):
- Trees more independent
- Variance of forest average: $\frac{\sigma^2_{tree}}{B} + \text{correlation terms}$
- Correlation terms vanish as $n \to \infty$
- CLT applies!

**Practical choice**: GRF uses $s_n = n/2$ (works well in practice)

---

## Asymptotic Normality: Proof Sketch

**Key tool**: Hoeffding (ANOVA) decomposition of U-statistic

**Decompose forest estimator**:
$$\hat{\theta} = E[\hat{\theta}] + \underbrace{\sum_{i} (E[\hat{\theta} | Z_i] - E[\hat{\theta}])}_{\text{Main effects (linear)}} + \underbrace{\text{Higher-order terms}}_{\text{Interactions}}$$

**Hájek projection** ($\dot{T}$): Keep only linear terms
- Sum of independent variables
- Standard CLT applies: $\dot{T} \xrightarrow{d} N(\mu, \sigma^2)$

---

## Proof Sketch: The Key Step

**Show that higher-order terms vanish**:
- Variance of interaction terms scales as $O(s_n^2 / n^2)$
- Under $s_n / n \to 0$: This goes to zero faster than $1/n$
- Complex forest estimator $\hat{\theta}$ asymptotically equivalent to linear projection $\dot{T}$

**Result**: Forest inherits Gaussianity from its linear approximation
$$\hat{\theta} - \theta_0 \approx \dot{T} - \theta_0 \sim N(0, \sigma^2/n)$$

**Convergence rate**: $\sqrt{n}$ (parametric rate despite non-parametric method!)

---

## Infinitesimal Jackknife for Variance

**Problem**: Need to estimate $\sigma^2(x)$ for confidence intervals
- Bootstrap-of-forests computationally prohibitive
- Need fast, accurate variance estimator

**Solution**: Infinitesimal Jackknife (IJ) from Efron (2014)

**IJ formula**:
$$\hat{V}_{IJ}(x) = \sum_{i=1}^n \text{Cov}_*(\hat{\tau}_b^*(x), N_{bi}^*)^2$$

where $N_{bi}^*$ counts appearances of observation $i$ in tree $b$'s subsample

---

## IJ Intuition

**What IJ measures**: Sensitivity to individual observations
- If including observation $i$ consistently shifts predictions
- Then $i$ contributes substantially to variance

**Calculation**:
- For each tree $b$: Record prediction $\hat{\tau}_b(x)$ and presence $N_{bi}$
- Compute covariance across trees
- Square and sum over observations

**Advantage**: Single forest run, no refitting
**Output**: Valid variance estimates for confidence intervals

---

<!-- _class: lead -->

# Part 4
## Generalized Random Forests

---

## Beyond CATE: The GRF Framework

**Causal Forest**: Specific instance of general framework

**Generalized Random Forests (GRF)**: Estimate any parameter $\theta(x)$ defined by moment equations

**Examples**:
- CATE: $\theta(x) = E[Y(1) - Y(0) | X=x]$
- Quantile: $\theta(x) = Q_\tau(Y | X=x)$
- IV: $\theta(x) = \text{Local IV effect}$

**Unified algorithm**: Same forest machinery, different moment conditions

---

## Local Moment Equations

**General form**:
$$E[\psi_{\theta(x), \nu(x)}(O_i) \mid X_i = x] = 0$$

where:
- $\psi$ is a score function
- $\theta(x)$ is target parameter
- $\nu(x)$ are nuisance parameters
- $O_i$ are observables

**Example (CATE)**: Score function
$$\psi = (Y_i - \tau(x) W_i - m(x))$$
where $m(x) = E[Y|X=x]$ is nuisance

---

## GRF Estimation Procedure

**Step 1**: Grow forest to obtain weights $\alpha_i(x)$
- Standard tree algorithm with gradient-based splitting

**Step 2**: Solve weighted moment equation
$$\sum_{i=1}^n \alpha_i(x) \psi(O_i, \hat{\theta}(x), \hat{\nu}(x)) = 0$$

**Step 3**: Invert to get $\hat{\theta}(x)$
- Often has closed-form solution
- Example: For CATE, solve weighted regression

---

## Gradient-Based Splitting

**Problem**: EMSE criterion specific to CATE
- Need general criterion for arbitrary $\theta$

**Solution**: Gradient-based pseudo-outcomes

**Compute pseudo-outcome**: For each observation $i$
$$\rho_i \approx -H^{-1} \psi(O_i, \hat{\theta}_{parent})$$

where $H$ is Hessian of moment condition

**Split trees**: Maximize variance of $\rho_i$ across children
- Finds directions of maximal parameter heterogeneity

---

## GRF Examples

**Quantile Forest**: $\theta(x) = Q_\tau(Y | X=x)$
- Score: $\psi = (Y \leq \theta) - \tau$
- Gradient: $\rho \approx \mathbb{I}(Y \leq \hat{Q})$

**IV Forest**: $\theta(x) = $ Local IV effect
- Score: Structural moment condition
- Allows heterogeneous instrumental variable estimation

**Survival Forest**: $\theta(x) = $ Hazard rate
- Score: Log-likelihood derivative

**Advantage**: Same forest engine, different applications

---

<!-- _class: lead -->

# Part 5
## Double Machine Learning Integration

---

## The Confounding Challenge

**Standard Causal Forest**: Assumes randomized/unconfounded treatment
$$Y_i(w) \perp W_i \mid X_i \quad \text{(Unconfoundedness)}$$

**Real-world problem**: Treatment assignment depends on covariates
- Doctors assign drugs based on patient characteristics
- Firms select into programs based on profitability

**Need**: Method to handle confounding in high dimensions

---

## Why Not Just Control Linearly?

**Naive approach**: Include covariates $X$ in forest

**Problem 1**: Regularization bias
- Forests shrink coefficients to reduce variance
- Bias in nuisance parameters propagates to $\hat{\tau}(x)$

**Problem 2**: Functional form misspecification
- Linear controls miss non-linear confounding
- High-dimensional interactions

**Solution**: Double Machine Learning (DML) framework

---

## The R-Learner: Overview

**Two-stage approach** (Nie & Wager 2021):

**Stage 1**: Residualize outcome and treatment
- Estimate $m(x) = E[Y_i | X_i = x]$ using ML
- Estimate $e(x) = E[W_i | X_i = x]$ using ML
- Compute residuals: $\tilde{Y}_i = Y_i - \hat{m}(X_i)$, $\tilde{W}_i = W_i - \hat{e}(X_i)$

**Stage 2**: Run Causal Forest on residuals
- Outcome: $\tilde{Y}$
- Treatment: $\tilde{W}$

---

## R-Learner: Stage 1 Details

**Nuisance estimation with cross-fitting**:

**Outcome regression**:
$$\hat{m}^{(-i)}(X_i) = E[Y_i | X_i] \quad \text{(trained on fold excluding } i \text{)}$$

**Propensity score**:
$$\hat{e}^{(-i)}(X_i) = E[W_i | X_i] \quad \text{(trained on fold excluding } i \text{)}$$

**Superscript** $(-i)$: Out-of-sample prediction
- Crucial for avoiding overfitting bias

---

## R-Learner: Stage 2 Objective

**Minimize R-loss**:
$$\hat{\tau}(\cdot) = \arg\min_\tau \sum_{i=1}^n (\tilde{Y}_i - \tau(X_i) \tilde{W}_i)^2$$

**Implementation**: Causal Forest solves this implicitly
- Forest weights provide local smoothing
- Solves weighted least squares in each leaf

**Local moment condition** (GRF form):
$$E[\tilde{W}_i (\tilde{Y}_i - \tilde{W}_i \tau(X_i)) \mid X_i = x] = 0$$

---

## Neyman Orthogonality: The Key Property

**Orthogonality condition**:
$$\frac{\partial}{\partial \eta} E[\psi(W; \theta_0, \eta)] \bigg|_{\eta = \eta_0} = 0$$

**Meaning**: Score function locally insensitive to nuisance errors
- First derivative w.r.t. $(m, e)$ is zero at truth
- Small errors in $\hat{m}, \hat{e}$ have negligible first-order effect on $\hat{\tau}$

**Why it matters**: Converts slow ML rates to fast parametric rates

---

## Orthogonality: Error Decomposition

**Without orthogonality**:
$$\hat{\theta} - \theta_0 \approx \underbrace{(\hat{m} - m_0)}_{\text{Error 1}} + \underbrace{(\hat{e} - e_0)}_{\text{Error 2}}$$
- Rate: $O_p(n^{-1/4})$ (slow ML convergence)

**With orthogonality**:
$$\hat{\theta} - \theta_0 \approx (\hat{m} - m_0) \times (\hat{e} - e_0)$$
- Product of errors!
- Rate: $n^{-1/4} \times n^{-1/4} = n^{-1/2}$ (parametric!)

**Result**: $\sqrt{n}$-consistency despite using slow ML methods

---

## Why Residualization Creates Orthogonality

**R-Learner score**:
$$\psi = \tilde{W}_i (\tilde{Y}_i - \tilde{W}_i \tau)$$

**Expand**:
$$\psi = (W_i - e(X_i))(Y_i - m(X_i) - (W_i - e(X_i))\tau)$$

**Check derivative w.r.t. $m$ at truth**:
$$\frac{\partial}{\partial m} E[\psi] = -E[W_i - e(X_i)] = 0$$

**By unconfoundedness**: $E[W | X] = e(X)$, so expectation is zero!
**Similar for $e$**: Derivative is zero

**Conclusion**: R-Learner score is Neyman orthogonal by construction

---

## Convergence Rate Comparison

| Method | Nuisance Rate | Target Rate | Inference |
|--------|---------------|-------------|-----------|
| Plug-in ML | $n^{-1/4}$ | $n^{-1/4}$ | Invalid |
| DML (orthogonal) | $n^{-1/4}$ | $n^{-1/2}$ | Valid |
| Oracle (known nuisances) | — | $n^{-1/2}$ | Valid |

**Key insight**: DML achieves oracle rate despite estimating nuisances!
**Practical implication**: Can use aggressive ML without sacrificing inference

---

<!-- _class: lead -->

# Part 6
## Implementation and Interpretation

---

## EconML: Causal ML Toolkit

**Microsoft EconML**: State-of-the-art Python library

**Key features**:
- `CausalForestDML`: Causal Forest with DML residualization
- Multiple meta-learners (T-Learner, X-Learner, DR-Learner)
- Built-in SHAP integration
- Policy learning utilities

**Installation**:
```python
pip install econml
```

---

## CausalForestDML: Basic Usage

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# Initialize estimator
est = CausalForestDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestRegressor(),
    n_estimators=4000,
    min_samples_leaf=5,
    max_depth=50,
    random_state=123
)

# Fit: Y=outcome, T=treatment, X=features, W=controls
est.fit(Y, T, X=X, W=W)
```

---

## Inference and Prediction

```python
# Predict CATE for new observations
tau_hat = est.effect(X_test)

# Confidence intervals (95%)
lb, ub = est.effect_interval(X_test, alpha=0.05)

# Summary statistics
print(f"Mean CATE: {tau_hat.mean():.3f}")
print(f"Std CATE: {tau_hat.std():.3f}")
print(f"Range: [{tau_hat.min():.3f}, {tau_hat.max():.3f}]")

# Test for heterogeneity
# If all CIs overlap zero: No detectable heterogeneity
```

---

## SHAP for CATE Interpretation

<!-- Suggested image: SHAP values visualization showing feature contributions to treatment effect heterogeneity -->

**Problem**: Forest is black-box, which $X$ drives heterogeneity?

**Solution**: SHAP (Shapley Additive Explanations)

**SHAP for CATE**: Decompose treatment effect
$$\hat{\tau}(x_i) = \bar{\tau} + \sum_{j=1}^p \phi_{ij}$$

where $\phi_{ij}$ is contribution of feature $j$ to unit $i$'s effect

**Interpretation**: $\phi_{Age} = +0.05$ means being this age increases effect by 0.05

---

## SHAP Implementation

```python
import shap

# Compute SHAP values for treatment effect
shap_values = est.shap_values(X)

# Summary plot: Which features matter?
shap.summary_plot(shap_values['Y0']['T0_1'], X)

# Dependence plot: How does Age affect treatment effect?
shap.dependence_plot('Age', shap_values['Y0']['T0_1'], X)

# Force plot: Explain single prediction
shap.force_plot(
    base_value=tau_hat.mean(),
    shap_values=shap_values['Y0']['T0_1'][0],
    features=X.iloc[0]
)
```

---

## SHAP vs. Feature Importance

| Method | Information | Interpretation |
|--------|-------------|----------------|
| **Feature Importance** | How often feature splits tree | "Age is important" |
| **SHAP** | Direction and magnitude | "Age +10 years → effect +0.05" |

**Feature importance**: Global relevance, no direction
**SHAP**: Local, signed, additive contributions

**Example**: SHAP reveals
- Young workers: Large training effect
- Old workers: Small training effect
- Feature importance only tells us "Age matters"

---

## Case Study: Pricing Elasticity

**Business problem**: Which customers are price-sensitive?

**Setup**:
- Treatment $P$: Price (continuous)
- Outcome $Y$: Purchase quantity
- Features $X$: Customer demographics, history
- Confounding: Prices set by algorithm based on predicted demand

**Goal**: Estimate $\theta(x) = \frac{\partial Y}{\partial P}$ (price elasticity)

---

## DML Strategy for Pricing

**Stage 1: Residualize**
- Model 1: Predict price from $X$ → Residual $\tilde{P} = P - \hat{P}(X)$
- Model 2: Predict quantity from $X$ → Residual $\tilde{Y} = Y - \hat{Y}(X)$

**Stage 2: Causal Forest**
- Run forest on $\tilde{Y}$ vs. $\tilde{P}$ with features $X$
- Output: $\hat{\theta}(x)$ for each customer

**Policy**: Target discounts to high-elasticity customers
- $\hat{\theta}(x) < -2$: Offer coupon (price-sensitive)
- $\hat{\theta}(x) > -0.5$: No discount (inelastic)

---

## Policy Learning from CATE

**Optimal policy problem**: Given $\hat{\tau}(x)$, whom to treat?

**Constrained optimization**:
$$\pi^* = \arg\max_\pi E[\tau(X) \cdot \pi(X)] \quad \text{s.t. } E[\pi(X)] \leq b$$

where $b$ is budget constraint

**Simple rule**: Treat top $b \times 100\%$ by predicted $\hat{\tau}(x)$

**Caution**: Estimation uncertainty
- Don't blindly rank by point estimates
- Consider confidence intervals

---

## Policy Learning: Thompson Sampling

**Problem**: Ranking by $\hat{\tau}(x)$ ignores uncertainty
- Units with noisy estimates may rank high by chance

**Solution**: Thompson Sampling
1. For each unit, draw $\tau_i^{(s)} \sim N(\hat{\tau}(x_i), \hat{\sigma}^2(x_i))$
2. Rank by sampled values
3. Repeat $S$ times and average policies

**Result**: Accounts for epistemic uncertainty
- More robust to estimation noise
- Better out-of-sample performance

---

<!-- _class: lead -->

# Conclusion

---

## From ATE to Precision Policy

**Traditional approach**: Estimate single effect, apply uniformly

**Modern approach**: Estimate heterogeneous effects, target optimally

**Key enablers**:
1. Causal Trees: Honest splitting for valid inference
2. Causal Forests: Asymptotic normality and confidence intervals
3. GRF: Unified framework for diverse parameters
4. DML: Robustness to high-dimensional confounding

**Result**: Data-driven policy targeting with statistical guarantees

---

## The Three Pillars

**1. Machine Learning**: Recursive partitioning handles interactions
- Finds heterogeneity in high dimensions
- No pre-specification of subgroups

**2. Classical Statistics**: Honesty and subsampling ensure validity
- Unbiased estimates
- Gaussian confidence intervals
- Nominal coverage

**3. Econometrics**: Orthogonalization handles confounding
- DML for nuisance parameters
- Parametric rates despite non-parametrics

---

## Practical Workflow

**Step 1: Diagnostics**
- Check for potential heterogeneity (descriptive stats)
- Assess confounding severity

**Step 2: Estimation**
- Use `CausalForestDML` with cross-fitting
- Choose suitable ML models for nuisances
- Set forest parameters (trees=4000, min_leaf=5)

**Step 3: Inference**
- Compute confidence intervals
- Test for significant heterogeneity
- Validate on held-out data

---

## Practical Workflow (continued)

**Step 4: Interpretation**
- SHAP values to identify drivers
- Dependence plots for key features
- Report effect distribution (mean, SD, range)

**Step 5: Policy Design**
- Formulate targeting rule
- Simulate counterfactual policies
- Estimate welfare gains

**Step 6: Robustness**
- Try alternative ML models
- Check sensitivity to hyperparameters
- Compare to simpler methods (linear interactions)

---

## Common Pitfalls

**1. Ignoring honesty**: Adaptive trees have invalid inference
- Always use honest estimation for causal questions

**2. Small samples**: Forests need $n > 1000$ ideally
- Below that, consider parametric interactions

**3. Overfitting to noise**: Declaring heterogeneity without testing
- Use confidence intervals, not just point estimates

**4. Ignoring confounding**: Causal Forest assumes unconfoundedness
- Use DML integration if treatment non-random

---

## When to Use Causal Forests

**Good scenarios**:
- Large sample ($n > 2000$)
- Many potential moderators (high-dimensional $X$)
- Non-linear interactions likely
- Need data-driven subgroup discovery

**Poor scenarios**:
- Small sample ($n < 500$)
- Few covariates with known interactions
- Strong prior knowledge about heterogeneity
- Inference not required (pure prediction)

---

## Alternative Methods

| Method | Strengths | Limitations |
|--------|-----------|-------------|
| **Linear interactions** | Simple, interpretable | Misses non-linearity |
| **Causal Forest** | Flexible, automatic | Requires large $n$ |
| **Meta-learners** | Plug-and-play | Less efficient |
| **Bayesian CART** | Uncertainty quantification | Computationally intensive |

**Recommendation**: Start with Causal Forest, validate with alternatives

---

## Software Ecosystem

**Python**:
- `econml`: Comprehensive, production-ready
- `causalml` (Uber): Alternative implementations

**R**:
- `grf`: Fastest implementation (C++ backend)
- `causalweight`: Additional methods

**Key advantage of `grf`**: Speed and memory efficiency
**Key advantage of `econml`**: Integration with scikit-learn ecosystem

---

## Extensions and Frontiers

**Current research directions**:

**1. Dynamic treatment regimes**: Sequential decisions
**2. Network interference**: Spillovers between units
**3. Continuous treatments**: Beyond binary interventions
**4. Survival outcomes**: Censored data
**5. Federated learning**: Privacy-preserving CATE estimation

**Open problems**: Theory lags practice
- Multi-armed bandits with forests
- Optimal policy with constraints

---

## Key Takeaways

**1. Heterogeneity matters**: ATE often insufficient for policy
**2. Honesty is crucial**: Eliminates bias, enables inference
**3. Forests achieve normality**: With subsampling and regularity
**4. Orthogonality is magic**: Parametric rates from non-parametric ML
**5. Interpretation is essential**: Use SHAP to understand drivers
**6. Policy learning is endpoint**: Go from $\hat{\tau}(x)$ to decisions

---

## Further Reading

**Foundational papers**:
- Athey & Imbens (2016): "Recursive Partitioning for Heterogeneous Causal Effects" (PNAS)
- Wager & Athey (2018): "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" (JASA)
- Athey, Tibshirani & Wager (2019): "Generalized Random Forests" (Annals of Statistics)

**DML integration**:
- Nie & Wager (2021): "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" (Biometrika)
- Chernozhukov et al. (2018): "Double/Debiased Machine Learning" (Econometrics Journal)

---

## Further Reading (continued)

**Applications**:
- Davis & Heller (2017): "Using Causal Forests to Predict Treatment Heterogeneity" (PNAS)
- Knaus, Lechner & Strittmatter (2021): "Machine Learning Estimation of Heterogeneous Causal Effects" (Econometrics Journal)

**Software**:
- Athey et al.: `grf` R package documentation
- Microsoft: `econml` Python package documentation

**Review**:
- Künzel et al. (2019): "Metalearners for Estimating Heterogeneous Treatment Effects" (PNAS)

---

<!-- _class: lead -->

# Questions?

**Office Hours**: [To be announced]
**Email**: [Your email]
**Course Website**: [Course link]

---

<!-- _class: lead -->

# Thank You!

**Next Lecture**: DAGs and Structural Causal Models

See you next time!
