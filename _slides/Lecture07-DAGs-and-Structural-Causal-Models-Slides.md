---
marp: true
theme: gaia
paginate: true
header: ''
footer: 'ECON6083 Lecture 7 | Structural Causal Models & Applied Identification'
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

# Lecture 7
## Structural Causal Models, Directed Acyclic Graphs, and Applied Identification in Economics

**ECON6083: Machine Learning in Economics**

---

## Today's Roadmap

**1. The Modern Identification Synthesis**
   - Bridging structure and design

**2. Structural Causal Models & DAGs**
   - Four elemental structures
   - The topology of causality

**3. Bad Controls**
   - M-bias and common pitfalls
   - Sample selection as collider bias

**4. Double Machine Learning**
   - High-dimensional confounding
   - Neyman orthogonality and cross-fitting

**5. Structural Identification**
   - Endogenous entry in strategic settings

**6. Applied Workflow**

---

<!-- _class: lead -->

# Part 1
## The Modern Identification Synthesis

---

## The Historical Divide

### Structuralist School (Cowles Commission)
- Write down "deep parameters" (preferences, technology, constraints)
- Explicit modeling of decision-making processes
- **Promise**: Counterfactual policy analysis
- **Cost**: Strong parametric assumptions, often implausible

### Credibility Revolution (Reduced-Form)
- Seek "natural experiments" and clean instrumental variables
- Emulate randomization of clinical trials
- **Promise**: Credible estimates of specific treatment effects
- **Cost**: Limited external validity, lacks theoretical richness

---

## The New Synthesis

**Modern paradigm integrates three pillars**:

**1. Rigorous causal language**
   - Structural Causal Models (SCMs) and Directed Acyclic Graphs (DAGs)

**2. Flexible estimation**
   - Machine Learning methods (Double/Debiased ML)

**3. Theoretical foundations**
   - Structural Industrial Organization models

**Key insight**: No longer forced to choose between "clean design" and "structural mechanisms"

**Today's goal**: Master this integrated toolkit for modern applied identification

---

<!-- _class: lead -->

# Part 2
## Structural Causal Models and DAGs

---

## From Algebra to Topology

<!-- Suggested image: Basic DAG structures showing nodes, edges, and causal flow -->

**Traditional approach**: Systems of simultaneous equations
- Obscures directionality and independence assumptions
- Difficult to reason about identification

**DAG approach**: Explicit causal flow via graphs
- **Nodes** = Variables
- **Directed edges** = Causal influence
- **Acyclic** = No feedback loops

**Advantage**: Forces explicit causal assumptions
- Transparent identification strategy
- Systematic reasoning about confounding

---

## The Topology of Association

**Core question**: When does statistical association flow between variables?

**Think of DAGs as pipe networks**:
- Association flows like water through pipes
- Specific arrangements act as "valves"
- Determine whether paths are "open" (transmitting) or "blocked" (independent)

**Four fundamental structures** govern the flow of association:
1. Fork (common cause)
2. Chain (mediator)
3. Collider (common effect)
4. Descendant (child of collider)

---

<!-- _class: lead -->

# The Four Elemental Structures

---

## Structure 1: The Fork (Common Cause)

**Configuration**: $Z \rightarrow X$ and $Z \rightarrow Y$

Example: Age ($Z$) → Income ($X$) and Health ($Y$)

**Mechanism**: $Z$ causes both $X$ and $Y$ (spurious correlation)
**Path**: OPEN | **Identification**: Condition on $Z$

---

## Structure 2: The Chain (Mediator)

**Configuration**: $X \rightarrow M \rightarrow Y$ (e.g., Education → Skills → Wages)

**Mechanism**: $X$ affects $Y$ through pathway $M$ (OPEN)

**Critical error**: DO NOT condition on $M$ for total effect
- Blocks causal channel (over-controlling bias)
- Only control for mediation analysis

---

## Structure 3: The Collider (Common Effect)

**Configuration**: $X \rightarrow C \leftarrow Y$

```
X (Talent) → C (Fame) ← Y (Beauty)
```

**Path status**: BLOCKED by default ($X \perp Y$ if no other paths)

**The collider bias trap**: Conditioning on $C$ OPENS the path
- Creates spurious association between independent causes $X$ and $Y$

---

## Collider: The Battery-Lightbulb Intuition

**Setup**: Battery ($X$) and Lightbulb ($Y$) both needed for Light ($C$)

**General population**: $X \perp Y$ (battery and bulb independent)

**Conditioning on "Light is OFF"** ($C = 0$):
- If battery works ($X = 1$) → bulb must be broken ($Y = 0$)
- Creates negative correlation between independent components!

**Lesson**: Collider conditioning is the structural foundation of selection bias

---

## Structure 4: The Descendant

**Configuration**: $X \rightarrow C \leftarrow Y$ and $C \rightarrow D$

**Key insight**: $D$ is descendant of collider $C$ (noisy proxy)

**Implication**: Conditioning on $D$ partially opens collider path
- Often inadvertent (e.g., controlling for income descending from ability)

---

## Summary: The Four Structures

| Structure | Configuration | Default Status | Conditioning Effect | Action |
|-----------|--------------|----------------|-------------------|--------|
| **Fork** | $Z \to X, Z \to Y$ | OPEN | CLOSES path | Include as control ✓ |
| **Chain** | $X \to M \to Y$ | OPEN | CLOSES path | Exclude (over-control) ✗ |
| **Collider** | $X \to C \gets Y$ | CLOSED | OPENS path | Exclude (induces bias) ✗ |
| **Descendant** | Collider child | CLOSED | Partially opens | Exclude (noisy collider) ✗ |

---

<!-- _class: lead -->

# Part 3
## The Taxonomy of "Bad Controls"

---

## Beyond "Kitchen Sink" Regression

**Traditional econometric intuition**:
- Control for everything available to minimize omitted variable bias
- More controls = better identification
- "Kitchen sink" approach

**DAG theory reveals**: This is formally incorrect!

**Critical distinction needed**:
- **Good controls**: Confounders that close backdoor paths (forks)
- **Bad controls**: Variables that introduce or amplify bias
  - Mediators (block causal mechanisms)
  - Colliders (open spurious paths)
  - Descendants of colliders (noisy collider bias)

---

## Bad Control 1: M-Bias

<!-- Suggested image: M-bias diagram showing U1 -> X, U1 -> Z <- U2, U2 -> Y structure -->

**M-Structure**: $U_1 \to X$, $U_1 \to Z \gets U_2$, $U_2 \to Y$ (where $Z$ is observed)

$X$ = treatment, $Y$ = outcome, $Z$ = pre-treatment, $U_1, U_2$ = unobserved

**Backdoor path**: $X \gets U_1 \to Z \gets U_2 \to Y$ BLOCKED at $Z$

---

## M-Bias: The Error

**Without controlling for $Z$**: Path blocked at collider (no bias)

**Controlling for $Z$ (because "it's pre-treatment")**:
- Opens collider, creates bridge between $U_1$ and $U_2$
- Manufactures bias where none existed!

**Key lesson**: Pre-treatment status ≠ sufficient for inclusion
- Must understand structural origin of variables

---

## Bad Control 2: The Class Size Paradox

**Research question**: Effect of Class Size ($X$) on Math Score ($Y$)

**True data generating process**:
- Good School ($G$) → Class Size, Math Score, History Score
- Ability ($A$) → Math Score, History Score

**Key observation**: History Score is a collider
- $G \to H \gets A$
- Observed pre-treatment, but should NOT control for it!

---

## Class Size Paradox (cont.)

**DAG analysis**:
- Path 1: $X \gets G \to Y$ (confounding, need to block)
- Path 2: $X \gets G \to H \gets A \to Y$ (blocked at collider $H$)

**Naive strategy**: Control for History Score
- Opens path 2, conflates class size with ability effect

**Correct strategy**: Control for $G$ (school quality) directly, NOT History

---

## Bad Control 3: Sample Selection

**"Famous Actress" Paradox**: Structure $T \to F \gets B$ (Fame is collider)

**Analyzing only famous actresses** (conditioning on $F$):
- If not very talented → must be very beautiful (to be in sample)
- Induces negative correlation between $T$ and $B$
- Tradeoff doesn't exist in general population

**Result**: "Worst" AND "most attractive" paradox explained by collider bias!

---

## Employment Selection Example

**DAG**: Education ($E$) → Employment ($S$) ← Ability ($A$) → Wages ($W$)

$E \to S$, $A \to S$, $A \to W$ where $A$ is unobserved

**Result**: $S$ is collider between $E$ and $A$ (selection bias)

---

## Employment Selection (cont.)

**Conditioning on Employment** ($S = 1$):
- Opens collider path: $E \to S \gets A \to W$
- High education + barely employed → lower ability
- Negative correlation transmitted to wages

**Direction of bias**: Downward biases return to education estimate

**Solution**: Heckman correction or structural selection model

---

<!-- _class: lead -->

# Part 4
## Double Machine Learning Framework

---

## The High-Dimensional Problem

**After using DAGs**: Identified correct adjustment set $W$

**New challenge**: $W$ often high-dimensional (text, geographic, demographic data)

**Traditional OLS failure**: Curse of dimensionality
- As $p \to n$: variance explodes, overfitting, invalid inference

**Solution needed**: ML's flexible modeling + valid causal inference

---

## The Partially Linear Model Setup

**Framework**: Partially Linear Regression (PLR)
$$Y = D\theta_0 + g_0(X) + U$$
$$D = m_0(X) + V$$

$Y$ = outcome, $D$ = treatment, $X$ = high-dim controls, $\theta_0$ = causal effect
$g_0(X)$, $m_0(X)$ = unknown nuisance functions

**Goal**: Estimate $\theta_0$ with valid confidence intervals

---

## Why Naive ML Fails

**Naive approach**: Use ML to estimate $g_0(X)$, then plug in

**Problem 1**: Regularization bias in $\hat{g}$ transmits to $\hat{\theta}$

**Problem 2**: Slow convergence rate $n^{-1/4}$ (vs parametric $n^{-1/2}$)
- Not $\sqrt{n}$-consistent, invalid confidence intervals

**Bottom line**: Cannot plug in ML predictions for causal inference

---

## The DML Solution: Neyman Orthogonality

**Core innovation**: Neyman Orthogonal score immunizes $\hat{\theta}$ against nuisance errors

**Orthogonal score for PLR**:
$$\psi(W; \theta, \eta) = (Y - D\theta - g(X)) \cdot (D - m(X))$$

$(Y - D\theta - g(X))$ = residualized outcome, $(D - m(X))$ = residualized treatment

**Key property**: Frisch-Waugh-Lovell theorem generalized to nonparametric $g, m$

---

## Why Orthogonality Works: The Math

**Neyman Orthogonality Condition**:
$$\frac{\partial}{\partial \eta} E[\psi(W; \theta_0, \eta)] \bigg|_{\eta = \eta_0} = 0$$

Score is "flat" w.r.t. nuisance parameters (second-order effect)

**The magic of products**: $(\hat{g} - g_0) \cdot (\hat{m} - m_0)$
- Each term: $n^{-1/4}$ convergence
- Product: $n^{-1/4} \times n^{-1/4} = n^{-1/2}$ (restores parametric rate!)

---

## Cross-Fitting: Breaking the Overfitting Correlation

**Remaining problem**: Overfitting bias
- Estimating nuisances on same data used for $\theta$
- Correlation between residuals and estimation error persists

**Solution**: Cross-Fitting (sample splitting)
- Estimate nuisances on independent sample, predict on held-out data
- Breaks correlation between estimation errors
- Like cross-validation, but for inference not prediction

---

## Cross-Fitting Algorithm

1. **Split**: Partition sample into $K$ folds ($I_k$, $I_k^c$)
2. **Train**: For each fold $k$, use $I_k^c$ to train $\hat{g}_k(X)$, $\hat{m}_k(X)$
3. **Predict**: Apply models to held-out fold $I_k$
4. **Residualize**: For $i \in I_k$: $\tilde{Y}_i = Y_i - \hat{g}_k(X_i)$, $\tilde{D}_i = D_i - \hat{m}_k(X_i)$
5. **Estimate**: Pool all residuals (DML2):
   $$\hat{\theta} = \frac{\sum_{k=1}^K \sum_{i \in I_k} \tilde{D}_i \tilde{Y}_i}{\sum_{k=1}^K \sum_{i \in I_k} \tilde{D}_i^2}$$

---

## DML: Complete Solution

**Innovations**: (1) Neyman Orthogonal score, (2) Cross-Fitting

**Properties**:
- $\sqrt{n}$-consistent, asymptotically normal, valid CIs
- Works with any ML method (RF, Lasso, Neural Nets)

**Software**: `DoubleML` package (Python/R)

---

## Case Study: Merger Retrospectives

**Context**: Estimating counterfactual price effects of mergers
- Example: T-Mobile/Sprint, MillerCoors joint venture

**The Problem**: Mergers are non-random
- Lack clean counterfactual
- Control vector $X$ is high-dimensional (demographics, trends, competition)
- Traditional methods fail (miss heterogeneity, parallel trends violated)

---

## Merger Retrospectives (cont.)

**The DML Solution** (Chernozhukov et al. 2018):

1. **Predict price trends**: Use ML to model $E[P | X]$
2. **Predict merger probability**: Model propensity $P(D=1 | X)$ with ML
3. **Orthogonalize**: Estimate treatment effect on residuals
4. **Result**: Valid inference on heterogeneous effects

---

## Merger Study: Key Findings

**Miller & Weinberg (2017)** - MillerCoors joint venture:

**Without DML**:
- Naive OLS: Small average price increase (~2%)
- Masks substantial heterogeneity

**With DML**:
- Average effect: 6-8% price increase
- **Heterogeneity revealed**:
  - Urban markets: 3-4% (competitive constraints)
  - Rural markets: 12-15% (reduced competition)
  - Areas with craft beer entry: 2-3% (substitutes available)

**Policy insight**: Average effects hide distributional impacts
- Rural consumers bear disproportionate burden

---

<!-- _class: lead -->

# Part 5
## Structural Identification with Endogenous Entry

---

## Case Study: Endogenous Entry in Airlines

**Context**: Estimating demand in oligopoly markets
- Airline route entry/exit is strategic
- Firm entry decisions driven by expected profitability
- Profitability depends on rivals' actions

**The Problem**: Standard selection corrections fail

**Key challenge**: Monotonicity assumption breaks down
- Heckman/Heckit assume Single Crossing Property
- In strategic games: high demand → more rivals → might deter entry
- **Strategic Substitutes**: Your entry less attractive when rivals enter

---

## Beyond Selection on Observables

**DML addresses**: Selection on high-dimensional observables ($Y \perp D | X$)

**Many problems involve**: Selection on unobservables
- Driven by strategic behavior in games
- Firms observe demand shocks $\xi$ that econometrician doesn't

**Challenge**: Standard corrections require monotonicity
- Fails when strategic interactions create multiple equilibria
- Need structural approach for game-theoretic selection

---

## The Breakdown of Monotonicity

**In competitive entry game**, firm's entry depends on:
1. **Market profitability**: Higher $\xi$ → higher profit → more entry
2. **Expected competition**: Higher $\xi$ → more rivals → lower profit → less entry

**Net effect**: Non-monotonic relationship
- Entry probability not monotonic in $\xi$
- Multiple equilibria for same $\xi$
- Invalidates standard Heckman-style control functions

---

## The Structural Model: Two-Stage Game

**Stage 1**: Firms $j \in \{1, ..., J\}$ enter if $\pi_{jm} = f(X_m, \xi_m, \text{competitors}) > 0$ (Bayesian Nash)

**Stage 2**: Active firms set prices via Nash-Bertrand (BLP logit demand with $\xi_{jm}$)

**Endogeneity**: $\xi_m$ observed by firms but not econometrician
- Drives both entry and pricing → biased demand estimates

---

## The Solution: Finite Mixture Model

**Step 1: Model Entry as Mixture**
- Economy has $T^*$ "Latent Market Types" (equilibrium configurations)
- Examples: "Hub-dominated", "Competitive routes", "Thin markets"

**Estimation**: EM algorithm
- Estimate $P(T_m = t | X_m)$ for each type
- Uses observed entry patterns, recovers equilibrium selection

---

## Step 2: Demand Estimation with Type Controls

**Control function approach**:
$$\text{Control} = \sum_{t=1}^{T^*} w_t \cdot P(T_m = t | X_m, \text{realized structure})$$

**Intuition**: Within equilibrium class, residual entry variation is quasi-random w.r.t. $\xi_m$
- Blocks backdoor path through unobserved $\xi_m$

**Result**: Consistent estimation of price elasticities ($\alpha$)
- Despite endogenous product availability, handles multiple equilibria

---

## Empirical Application: US Airlines

**Data**: Quarterly route-level observations
- Entry/exit decisions by carrier
- Prices and market shares
- Route characteristics (distance, airports, demographics)

**Key findings (Aguirregabiria et al. 2023)**:

1. **Bias from ignoring endogeneity**: Substantial
   - Naive OLS: Price elasticity = -1.2
   - Corrected estimate: Price elasticity = -2.5
   - Naive estimates underestimate sensitivity by ~50%

2. **Latent market types identified**: 3-4 distinct equilibrium classes

---

## Policy Implications

**Merger analysis without correction**:
- Underestimates consumer price sensitivity
- Overestimates disciplining power of potential entrants
- Predicts smaller welfare losses from consolidation

**With structural correction**:
- Reveals true elasticity is much higher
- Entry barriers are strategic, not just cost-based
- Merger simulations show 2-3x larger welfare losses

**Broader lesson**:
- Strategic selection requires game-theoretic modeling
- Cannot be addressed by standard selection corrections
- Mixture approach generalizes to other IO settings

---

<!-- _class: lead -->

# Part 6
## The Applied Identification Workflow

---

## Integrated Three-Phase Workflow

**Phase 1: Structural Definition** (The DAG)
- Map causal structure before estimation
- Identify confounders, mediators, colliders
- Audit for M-bias and bad controls
- DAG serves as "constitution" of research design

**Phase 2: Estimation Strategy**
- Match method to selection mechanism
- Observables → DML
- Strategic unobservables → Structural control functions

**Phase 3: Robustness and Validation**
- Placebo tests
- Sensitivity analysis
- Cross-validation across specifications

---

## Phase 1: Building the DAG

**Systematic steps**:

1. **Variable mapping**: Define $Y$ (outcome), $D$ (treatment), potential $W$ (controls)
2. **Structure identification**: Find forks (include), colliders (exclude)
3. **Bad control audit**: Check for M-bias, mediators, collider descendants
4. **State assumptions**: Document independence assumptions and open paths

**Output**: Clear causal graph documenting identification strategy

---

## Phase 2: Choosing the Estimator

**Scenario A**: Selection on observables ($Y \perp D | W$)
- **Tool**: DML | **Output**: Unbiased, $\sqrt{n}$-consistent $\theta_0$

**Scenario B**: Selection on strategic unobservables ($\xi \to D, Y$)
- **Tool**: Structural Control Functions | **Output**: Deep parameters

---

## Phase 3: Robustness Checks

**Placebo tests**:
- Test for effects that should theoretically be zero
- Example: Effect of future policy on past outcomes
- Use DML machinery to check null hypotheses

**Sensitivity analysis**:
- DML relies on unconfoundedness assumption
- Quantify: How strong must unobserved confounder be?
- Use omitted variable bias bounds (Oster, Cinelli methods)

**Cross-validation of estimators**:
- Verify estimates stable across ML algorithms
- Check sensitivity to hyperparameter tuning
- Compare DML variants (DML1 vs. DML2)

---

## Comparison of Identification Approaches

| Feature | OLS + Controls | Double ML (DML) | Structural CF |
|---------|----------------|-----------------|---------------|
| **Assumption** | Linearity, $E[U\|X,D]=0$ | Unconfoundedness | Economic primitives |
| **Controls** | Linear projection | Nonlinear, high-dim ML | Equilibrium model |
| **Output** | Correlations | Causal effect $\theta_0$ | Deep parameters |
| **Key Risk** | Omitted variable | Overlap violation | Misspecification |

---

<!-- _class: lead -->

# Conclusion

---

## Key Takeaways

**1. DAGs clarify causal structure**
   - Distinguish confounders from colliders; "control for everything" is dangerous

**2. DML enables valid inference with ML**
   - Orthogonality + Cross-fitting restore $\sqrt{n}$-consistency

**3. Structural methods handle strategic selection**
   - Mixture models identify equilibrium types without monotonicity

---

## Practical Guidelines for Researchers

**Your empirical workflow should be**:

1. **Start with the DAG** (before looking at data!)
   - Map out all potential causal paths
   - Identify which variables are confounders vs. colliders
   - Document assumptions explicitly

2. **Identify correct control set**
   - Include: Confounders (forks)
   - Exclude: Mediators, colliders, descendants

3. **Match estimator to selection mechanism**
   - Observables + high-dim → DML
   - Strategic unobservables → Structural models

4. **Validate extensively**
   - Placebo tests, sensitivity analysis
   - Cross-validation across specifications

---

## The Path Forward

**The credibility revolution has matured**:
- Moved beyond "RCT vs. Structural" binary
- Sophisticated synthesis of tools

**Key principle**: ML is not a silver bullet
- Without DAG: Precisely estimate the wrong parameter
- Without causal structure: Results are not interpretable

**The modern synthesis offers**:
- Topological rigor of Pearl's DAGs
- Algorithmic power of Chernozhukov's DML
- Game-theoretic insights of structural IO

**Result**: Tackle immense complexity with unprecedented rigor

---

## Further Reading

**DAGs and Causal Inference**:
- **Pearl & Mackenzie (2018)**: *The Book of Why* - DAGs and causality
- **Cinelli, Forney, & Pearl (2022)**: "A crash course in good and bad controls"
- **Hernán & Robins (2020)**: *Causal Inference: What If*

**Double Machine Learning**:
- **Chernozhukov et al. (2018)**: "Double/debiased ML for treatment and structural parameters"
- **Miller & Weinberg (2017)**: "Understanding the price effects of the MillerCoors joint venture"

**Structural Identification**:
- **Aguirregabiria, Iaria, & Sokullu (2023)**: "Identification and estimation of demand models with endogenous product entry and exit"
- **Ciliberto & Tamer (2009)**: "Market structure and multiple equilibria in airline markets"

**Software**: Python/R `DoubleML`, `EconML` packages

---

<!-- _class: lead -->

# Questions?

**Office Hours**: [To be announced]
**Email**: [Your email]
**Course Website**: [Course link]

---

<!-- _class: lead -->

# Thank You!

**Next Lecture**: Instrumental Variables & DML-IV

See you next time!

