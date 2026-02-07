---
marp: true
theme: academic
paginate: true
math: mathjax
footer: 'ECON6083 Lecture 8 | Structural Causal Models & Applied Identification'
---

<!-- _class: lead -->

# Lecture 8
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

**Modern paradigm integrates three pillars:**

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

**Think of DAGs as pipe networks:**
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

```
      Z (Age)
     / \
    ↓   ↓
    X    Y
(Income) (Health)
```

**Mechanism**:
- $Z$ exerts causal influence on both $X$ and $Y$
- Variations in $Z$ induce simultaneous variations in $X$ and $Y$
- Creates spurious correlation between $X$ and $Y$

**Path status**: OPEN (association flows from $X$ to $Z$ to $Y$)

**Identification**: Condition on $Z$ to block the backdoor path
- Within strata of $Z$, variations in $X$ no longer driven by $Z$

---

## Structure 2: The Chain (Mediator)

**Configuration**: $X \rightarrow M \rightarrow Y$

```
X (Education) → M (Skills) → Y (Wages)
```

**Mechanism**:
- Causal effect of $X$ on $Y$ transmitted through $M$
- $M$ is the mechanism or pathway

**Path status**: OPEN (causality flows from $X$ to $Y$ through $M$)

**Critical identification issue**:
- **DO NOT** condition on $M$ if estimating total effect
- Controlling for mediator blocks the causal channel
- Common error: "over-controlling" biases toward zero

**When to control**: Only for mediation analysis (direct vs. indirect effects)

---

## Structure 3: The Collider (Common Effect)

**Configuration**: $X \rightarrow C \leftarrow Y$

```
X (Talent) → C (Fame) ← Y (Beauty)
```

**Mechanism**:
- Two independent variables both cause $C$
- $X$ and $Y$ have separate causal pathways to $C$

**Path status**: BLOCKED by default
- $X$ and $Y$ are independent (if no other paths)
- Knowing $X$ tells us nothing about $Y$

**The collider bias trap**:
- Conditioning on $C$ OPENS the path
- Creates spurious association between independent causes
- Most counterintuitive structure in DAG theory

---

## Collider: The Battery-Lightbulb Intuition

**Setup**: Battery ($X$) and Lightbulb ($Y$) both needed for Light ($C$)

**In general population**:
- Battery status and bulb status are independent
- Knowing battery works tells us nothing about bulb

**Conditioning on "Light is OFF"** ($C = 0$):
- Observe battery works ($X = 1$)
- Can infer bulb must be broken ($Y = 0$)
- Created negative correlation between independent components!

**Lesson**: Conditioning on colliders induces spurious associations
- This is the structural foundation of selection bias

---

## Structure 4: The Descendant

**Configuration**: $X \rightarrow C \leftarrow Y$ and $C \rightarrow D$

```
X → C ← Y
    ↓
    D
```

**Key insight**: $D$ is a descendant of collider $C$
- Contains information about $C$ (noisy proxy)

**Implication**: Conditioning on $D$ partially opens collider path
- Same bias mechanism as conditioning on $C$ itself
- Attenuated magnitude but same direction

**Practical importance**: Often inadvertent in applied work
- Example: Controlling for "income" when income descends from unobserved "ability" collider

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

**Standard intuition**: Control for all pre-treatment variables

**M-Structure counterexample**:
```
U₁ → X              Y ← U₂
  ↘   ↑            ↑   ↗
       Z (observed)
```

- $X$ = treatment, $Y$ = outcome
- $Z$ = pre-treatment covariate (observed)
- $U_1, U_2$ = unobserved latent variables

**Analysis of backdoor path**: $X \gets U_1 \to Z \gets U_2 \to Y$
- Path is BLOCKED at collider $Z$
- No spurious correlation in raw data

---

## M-Bias: The Error

**Without controlling for $Z$**:
- Path blocked at collider
- No bias through this pathway

**Controlling for $Z$ (because "it's pre-treatment")**:
- Opens the collider
- Creates bridge between $U_1$ and $U_2$
- Manufactures bias where none existed!

**Key lesson**:
- Pre-treatment status ≠ sufficient condition for inclusion
- Must understand structural origin of variables
- Challenge to standard "control for baseline covariates" advice

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

**Naive strategy**: Control for History Score
- Thinking it's a "good proxy for ability" or "school quality"

**DAG analysis reveals the problem**:
- Backdoor path 1: $X \gets G \to Y$ (confounding, need to block)
- Backdoor path 2: $X \gets G \to H \gets A \to Y$ (blocked at collider $H$)

**Controlling for History**:
- Opens path 2 between Good School and Ability
- Conflates class size effect with student ability effect
- Biased estimate (direction depends on correlations)

**Correct strategy**: Control for school quality directly, NOT History

---

## Bad Control 3: Sample Selection

**Sample selection as collider bias**: Heckman selection problem

**"Famous Actress" Paradox**:
- To become Famous ($F$): need high Talent ($T$) OR high Beauty ($B$)
- Structure: $T \to F \gets B$ (Fame is collider)

**Analyzing only famous actresses** (conditioning on $F$):
- If actress is not very talented → must be very beautiful (to be in sample)
- Induces negative correlation between Talent and Beauty
- This tradeoff doesn't exist in general population

**Voted "worst" AND "most attractive"**: Makes sense with collider bias!

---

## Employment Selection Example

**Estimating returns to education** ($E \to W$)

**Problem**: Only observe wages for employed workers ($S$)

**DAG structure**:
```
Education → Employment ← Ability
                ↓
              Wages
```

- Education ($E$) increases Employment probability ($S$)
- Unobserved Ability ($A$) increases Employment ($S$)
- Unobserved Ability ($A$) increases Wages ($W$)

**Result**: Employment is collider between Education and Ability

---

## Employment Selection (cont.)

**Mechanism of bias**:

Conditioning on Employment (only $S = 1$):
- Opens collider path: $E \to S \gets A \to W$
- Education negatively correlated with ability in sample
- High education + barely employed → lower ability
- Low education + employed → higher ability

**Direction of bias**:
- Negative correlation transmitted to wages
- Downward biases return to education estimate

**Solution**: Heckman correction or structural selection model

---

<!-- _class: lead -->

# Part 4
## Double Machine Learning Framework

---

## The High-Dimensional Problem

**After using DAGs**: Identified correct adjustment set $W$

**New challenge**: $W$ is often high-dimensional in modern applications
- Text data (thousands of word features)
- Geographic indicators (census tracts)
- Complex demographic interactions
- Image data, sensor data

**Traditional OLS failure**: "Curse of dimensionality"
- As $p \to n$: variance of OLS explodes
- Model overfits, predictions unreliable
- Cannot do valid statistical inference

**Need solution**: Use ML's flexible modeling while preserving causal inference

---

## The Partially Linear Model Setup

**Framework**: Partially Linear Regression (PLR)
$$Y = D\theta_0 + g_0(X) + U$$
$$D = m_0(X) + V$$

**Components**:
- $Y$ = outcome variable
- $D$ = treatment/policy variable of interest
- $X$ = high-dimensional vector of controls
- $\theta_0$ = causal effect (parameter of interest)
- $g_0(X)$ = unknown confounding function (nuisance)
- $m_0(X)$ = unknown propensity function (nuisance)

**Goal**: Estimate $\theta_0$ with valid confidence intervals

---

## Why Naive ML Fails

**Naive approach**: Use ML to estimate $g_0(X)$, then plug in

**Problem 1: Regularization bias**
- ML deliberately biased (shrinks coefficients)
- Bias in $\hat{g}$ transmits to $\hat{\theta}$

**Problem 2: Slow convergence**
- ML rate: $n^{-1/4}$ (vs parametric $n^{-1/2}$)
- Not $\sqrt{n}$-consistent
- Invalid confidence intervals

**Bottom line**: Cannot plug in ML predictions for causal inference

---

## The DML Solution: Neyman Orthogonality

**Core innovation**: Construct "Neyman Orthogonal" score function
- Immunizes $\hat{\theta}$ against small errors in nuisance estimation

**Orthogonal score for PLR**:
$$\psi(W; \theta, \eta) = (Y - D\theta - g(X)) \cdot (D - m(X))$$

**Interpretation**:
- $(Y - D\theta - g(X))$ = residualized outcome
- $(D - m(X))$ = residualized treatment
- Regression of residualized $Y$ on residualized $D$

**Key property**: "Frisch-Waugh-Lovell" theorem generalized to nonparametric $g, m$

---

## Why Orthogonality Works: The Math

**Neyman Orthogonality Condition**:
$$\frac{\partial}{\partial \eta} E[\psi(W; \theta_0, \eta)] \bigg|_{\eta = \eta_0} = 0$$

**Meaning**: Score is "flat" w.r.t. nuisance parameters
- Small errors in $\hat{g}, \hat{m}$ have second-order effect

**The magic of products**:
$$(\hat{g} - g_0) \cdot (\hat{m} - m_0)$$

- Each term: $n^{-1/4}$ convergence
- Product: $n^{-1/4} \times n^{-1/4} = n^{-1/2}$ ✓
- Restores parametric rate!

---

## Cross-Fitting: Breaking the Overfitting Correlation

**Remaining problem**: Overfitting bias
- If estimate nuisances on same data used for $\theta$
- ML captures stochastic sample noise
- Correlation between residuals and estimation error persists

**Solution**: Cross-Fitting (sample splitting)
- Estimate nuisances on independent sample
- Always predict on held-out data
- Breaks correlation between estimation errors

**Analogy**: Like cross-validation, but for inference not prediction

---

## Cross-Fitting Algorithm (Part 1)

**Steps 1-3**: Data Splitting and Training

1. **Split**: Randomly partition sample into $K$ folds (e.g., $K=5$)
   - Let $I_k$ = fold $k$, $I_k^c$ = training complement

2. **Train**: For each fold $k$, use $I_k^c$ to train ML models:
   - $\hat{g}_k(X)$ predicting $Y$ from $X$
   - $\hat{m}_k(X)$ predicting $D$ from $X$

3. **Predict**: Apply models to held-out fold $I_k$

---

## Cross-Fitting Algorithm (Part 2)

**Steps 4-5**: Residualization and Estimation

4. **Residualize**: For each $i \in I_k$:
   - $\tilde{Y}_i = Y_i - \hat{g}_k(X_i)$
   - $\tilde{D}_i = D_i - \hat{m}_k(X_i)$

5. **Estimate**: Pool all residuals (DML2 variant):
   $$\hat{\theta} = \frac{\sum_{k=1}^K \sum_{i \in I_k} \tilde{D}_i \tilde{Y}_i}{\sum_{k=1}^K \sum_{i \in I_k} \tilde{D}_i^2}$$

---

## DML: Complete Solution

**Two key innovations**:

1. **Neyman Orthogonal score**
   - Immunizes against regularization bias
   - Product-of-errors converges faster

2. **Cross-Fitting**
   - Breaks overfitting correlation
   - Ensures independence between nuisance estimates and residuals

**Resulting properties**:
- $\sqrt{n}$-consistent estimation of $\theta_0$
- Asymptotically normal distribution
- Valid confidence intervals from influence function
- Works with any ML method (Random Forest, Lasso, Neural Nets, etc.)

**Software**: `DoubleML` package (Python/R)

---

## Case Study: Merger Retrospectives (Setup)

**Context**: Estimating counterfactual price effects of mergers
- Example: T-Mobile/Sprint, MillerCoors joint venture
- Need to estimate "what would have happened" without merger

**The Problem**: Mergers are non-random
- Occur in markets with specific trends and characteristics
- Lack clean counterfactual (no randomization)
- Control vector $X$ is high-dimensional
  - Thousands of local demographic features
  - Market-specific time trends
  - Competitive structure variables

---

## Merger Retrospectives (Solution)

**Traditional approach fails**:
- OLS with market fixed effects: misses heterogeneity
- Difference-in-differences: requires parallel trends (often violated)
- Manual control selection: arbitrary, omitted variable bias

**The DML Solution** (Chernozhukov et al. 2018):

1. **Predict price trends**: Use ML (Lasso/Random Forest) to model $E[P | X]$
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
- Antitrust analysis should account for local market structure

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

**Key References**:
- Aguirregabiria, Iaria, & Sokullu (2023): Latent market types solution
- Ciliberto & Tamer (2009): Multiple equilibria in airline markets

---

## Beyond Selection on Observables

**DML addresses**: Selection on high-dimensional observables
- Assumes $Y \perp D | X$ after controlling for $X$
- Flexible estimation of complex $g_0(X), m_0(X)$

**Many problems involve**: Selection on unobservables
- Driven by strategic behavior in games
- Firms observe demand shocks $\xi$ that econometrician doesn't
- Example: Airlines know local demand conditions we don't observe

**Challenge**: Standard corrections require monotonicity
- Heckman: Assumes single crossing property
- Fails when strategic interactions create multiple equilibria
- Need structural approach to handle game-theoretic selection

---

## The Breakdown of Monotonicity

**In competitive entry game**, firm's entry depends on:

1. **Market profitability** (direct effect)
   - Higher demand shock $\xi$ → higher profit → more likely to enter ✓

2. **Expected competition** (strategic effect)
   - Higher $\xi$ → rivals more likely to enter
   - More rivals → lower profit → less likely to enter ✗

**Net effect**: Non-monotonic relationship
- Entry probability not monotonic in $\xi$
- Can have multiple equilibria for same $\xi$
- Invalidates standard Heckman-style control functions

---

## The Structural Model: Stage 1

**Entry Game** (Discrete game of incomplete information)

- Firms $j \in \{1, ..., J\}$ simultaneously decide entry
- Profit function: $\pi_{jm} = f(X_m, \xi_m, \text{number of competitors})$
- Enter if $\pi_{jm} > 0$
- Equilibrium: Bayesian Nash

**Key feature**: Strategic interaction
- Firm's entry depends on expected rivals' actions
- Multiple equilibria possible

---

## The Structural Model: Stage 2

**Pricing Game** (Conditional on entry)

- Active firms observe realized market structure
- Set prices via Nash-Bertrand equilibrium
- Consumer demand: BLP-style logit with unobserved quality $\xi_{jm}$

**Endogeneity problem**:
- $\xi_m$ observed by firms in Stage 1
- Unobserved by econometrician
- Drives both entry decision and pricing → biased demand estimates

---

## The Solution: Finite Mixture Model (Step 1)

**Innovation**: Two-Step Mixture Model approach

**Step 1: Model Entry as Mixture**
- Economy has $T^*$ "Latent Market Types"
- Each type = specific equilibrium configuration
  - Type 1: "Hub-dominated"
  - Type 2: "Competitive routes"
  - Type 3: "Thin markets"

**Estimation**: EM algorithm
- Estimate $P(T_m = t | X_m)$ for each type
- Uses observed entry patterns
- Recovers equilibrium selection mechanism

---

## The Solution: Finite Mixture Model (Step 2)

**Step 2: Demand Estimation with Type Controls**

**Control function approach**:
- Instead of single inverse Mill's ratio
- Use weighted sum of type probabilities

$$\text{Control} = \sum_{t=1}^{T^*} w_t \cdot P(T_m = t | X_m, \text{realized structure})$$

**Intuition**:
- Within specific market type (equilibrium class)
- Residual variation in entry is quasi-random w.r.t. demand error
- Blocks backdoor path through unobserved $\xi_m$

**Result**: Consistent estimation of price elasticities ($\alpha$)
- Despite endogenous product availability
- Handles multiple equilibria explicitly

---

## Multiple Equilibria in Airline Markets

**Ciliberto & Tamer (2009)** document the problem:

**Empirical observation**: Same market characteristics → different entry patterns
- Route A: American + United enter
- Route B (similar): Only American enters
- Route C (similar): Both stay out

**Explanation**: Multiple Nash equilibria
- Equilibrium 1: "Both enter" (optimistic beliefs)
- Equilibrium 2: "One enters" (focal coordination)
- Equilibrium 3: "Neither enters" (pessimistic beliefs)

**Standard approach fails**: Cannot predict which equilibrium
- Need to model equilibrium selection mechanism

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

2. **Economic mechanism**:
   - High-price carriers disproportionately in markets with market power
   - Consumers appear "loyal" but actually respond to competitive pressure

3. **Latent market types identified**: 3-4 distinct equilibrium classes
   - Hub-dominated routes (Type 1)
   - Competitive leisure routes (Type 2)
   - Thin business routes (Type 3)

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

## Case Studies: Summary Comparison

| Feature | Merger Retrospectives | Airline Entry |
|---------|----------------------|---------------|
| **Main Challenge** | High-dimensional observables | Unobserved strategic shocks |
| **Selection Type** | Non-random treatment timing | Endogenous market structure |
| **Key Assumption** | Unconfoundedness $Y \perp D \| X$ | Multiple equilibria |
| **Method** | Double Machine Learning | Finite Mixture Models |
| **Key Innovation** | Orthogonal score + Cross-fit | Latent market types |
| **Output** | Heterogeneous treatment effects | Structural demand parameters |
| **Policy Use** | Ex-post merger evaluation | Ex-ante merger simulation |

**Both require**: Explicit modeling of selection mechanism
**Both deliver**: Valid inference despite complex confounding

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

2. **Structure identification**:
   - Find forks (confounders to include)
   - Find colliders (variables to exclude)

3. **Bad control audit**:
   - Scrutinize pre-treatment variables for M-bias
   - Exclude mediators to avoid over-controlling
   - Check for descendants of colliders

4. **State assumptions explicitly**:
   - Document independence assumptions
   - Note which paths remain open

**Output**: Clear causal graph documenting identification strategy

---

## Phase 2: Choosing the Estimator

**Scenario A: Selection on High-Dimensional Observables**

- **Condition**: $Y \perp D | W$ (unconfoundedness given large $W$)
- **Tool**: Double/Debiased Machine Learning (DML)
- **Implementation**:
  - Cross-Fitting with $K$ folds
  - Neyman Orthogonal scores (PLR, IRM, LATE)
  - Any ML method for nuisances (RF, Lasso, GBM)
- **Output**: Unbiased, $\sqrt{n}$-consistent estimate of $\theta_0$

**Scenario B: Selection on Strategic Unobservables**

- **Condition**: $\xi \to D$ and $\xi \to Y$, driven by game theory
- **Tool**: Structural Control Functions (Aguirregabiria et al.)
- **Implementation**: Finite Mixture Model → propensity controls
- **Output**: Deep structural parameters (elasticities, marginal costs)

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
| **Primary Assumption** | Linearity, $E[U\|X,D]=0$ | Unconfoundedness | Economic primitives |
| **Control Variables** | Linear projection | Nonlinear, high-dim ML | Equilibrium model |
| **Selection Bias** | Vulnerable if unobserved | Vulnerable if unobserved | Explicitly modeled |
| **Monotonicity** | N/A | N/A | Not required ✓ |
| **Output** | Correlations | Causal effect $\theta_0$ | Deep parameters |
| **Key Risk** | Omitted variable | Overlap violation | Game misspecification |
| **External Validity** | Limited | Limited | Structural (transportable) |

---

<!-- _class: lead -->

# Conclusion

---

## Key Takeaways

**1. DAGs clarify causal structure**
   - Distinguish confounders from colliders
   - "Control for everything" is dangerous
   - Selection bias has structural foundation

**2. DML enables valid inference with ML**
   - Orthogonality + Cross-fitting = valid inference
   - Works with any ML method
   - Restores $\sqrt{n}$-consistency

**3. Structural methods handle strategic selection**
   - Mixture models identify equilibrium types
   - No monotonicity assumption required

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

## Applications and Extensions

**Labor Economics**
- Wage equations with high-dimensional controls
- Employer-employee matched data selection

**Industrial Organization**
- Demand with endogenous market structure
- Merger simulations with strategic entry

**Policy Evaluation**
- Heterogeneous treatment effects
- Optimal policy learning

**Development Economics**
- High-dimensional program evaluation
- Network spillover effects

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
- **Pearl, Glymour, & Jewell (2016)**: *Causal Inference in Statistics: A Primer*
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

**Office Hours**: [Your information here]
**Next Lecture**: [Topic here]

---

## Appendix: Practical Checklist

### Control Variable Decision Table

| Variable Type | Definition | Regression Action | Reason |
|--------------|------------|-------------------|--------|
| **Confounder** | Causes both $D$ and $Y$ | **INCLUDE** | Blocks backdoor path |
| **Mediator** | $D \to M \to Y$ | **EXCLUDE** | Blocks causal mechanism |
| **Collider** | $D \to C \gets Y$ | **EXCLUDE** | Opens spurious path |
| **Descendant** | Child of collider | **EXCLUDE** | Partial collider bias |
| **Proxy** | Correlated with confounder | **INCLUDE** (cautiously) | Reduces unobserved bias |
| **M-bias variable** | Pre-treatment collider | **EXCLUDE** | Opens latent path |

---

<!-- _class: lead -->

# Thank You!

**Contact**: haoyu@hku.hk
**Course materials**: [Website](https://jasminehao.com/econ6083-slides/)
