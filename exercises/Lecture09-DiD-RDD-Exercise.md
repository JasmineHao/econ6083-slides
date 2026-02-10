# In-Class Exercise: Difference-in-Differences & RDD

**Course:** ECON6083 - Machine Learning in Economics
**Topic:** DiD with Parallel Trends, Staggered Treatment, and RDD Basics
**Time:** 25 minutes

---

## Part I: Parallel Trends Assessment (10 minutes)

### Scenario: Minimum Wage and Employment

**Research Question:** What is the effect of minimum wage increases on teenage employment?

**Classic Study:** Card & Krueger (1994) - NJ vs PA minimum wage

**Setup:**
- **Treatment group:** New Jersey (raised minimum wage from $4.25 to $5.05 in April 1992)
- **Control group:** Pennsylvania (no change in minimum wage)
- **Outcome:** Teenage employment rate (% of teens employed)
- **Data:** Monthly employment from Jan 1990 to Dec 1993

---

### Question 1: Visualizing Trends

You are shown **four graphs** of treatment vs control group trends over time.

For each graph, decide: **Does it support the parallel trends assumption?**

---

#### **Graph A:**

```
Employment Rate (%)
      |
  20  |     Treatment ----
      |    /
  18  |   /    Control ----
      |  /____________________
  16  | /
      |/
  14  +------------------------
      1990  1991  1992  1993
           (↑ policy change April 1992)
```

**Parallel trends before April 1992?**
- [ ] Yes - lines are parallel before treatment
- [ ] No - lines have different slopes

**Answer:** **Yes** - Both groups have flat trends before the policy. After April 1992, treatment group increases (positive effect of min wage on employment? Controversial!)

---

#### **Graph B:**

```
Employment Rate (%)
      |
  20  |            Treatment ----
      |          ↗
  18  |        ↗    Control ----
      |      ↗____↗_______________
  16  |    ↗    ↗
      |  ↗    ↗
  14  +------------------------
      1990  1991  1992  1993
           (↑ policy change)
```

**Parallel trends before April 1992?**
- [ ] Yes
- [ ] No

**Answer:** **No** - Treatment group has a steeper upward trend BEFORE the policy change. This violates parallel trends. Any difference after April 1992 could be due to pre-existing trends, not the policy.

**What to do:** Use a different control group, or add time trends to the regression.

---

#### **Graph C:**

```
Employment Rate (%)
      |  Treatment ----
  20  | ____________
      |     Control ----
  18  |    ___________________
      |
  16  |
      |
  14  +------------------------
      1990  1991  1992  1993
           (↑ policy change)
```

**Parallel trends before April 1992?**
- [ ] Yes
- [ ] No

**Answer:** **Yes** - Both groups are flat and parallel before treatment. After treatment, BOTH remain flat (no effect).

**Interpretation:** The minimum wage increase had **no effect** on employment (consistent with Card & Krueger's finding).

---

#### **Graph D:**

```
Employment Rate (%)
      |
  20  |     Treatment ----
      |    ↗        ↓
  18  |   ↗         ↓
      |  ↗__________↓__________
  16  | ↗    Control ----
      |↗
  14  +------------------------
      1990  1991  1992  1993
           (↑ policy change)
```

**Parallel trends before April 1992?**
- [ ] Yes
- [ ] No

**Answer:** **No** - The lines converge before treatment (different slopes). This is a **convergence** pattern, which violates parallel trends.

**Problem:** Even if we see a drop in treatment group after April 1992, it could be due to pre-existing convergence, not the policy.

---

### Question 2: Testing Parallel Trends

**Formal test:** Regress outcome on treatment × pre-period time trends.

**Regression:**
$$Y_{it} = \alpha + \beta_1 Treat_i + \beta_2 Post_t + \gamma (Treat_i \times t) + \varepsilon_{it}$$

where:
- $Treat_i = 1$ if New Jersey
- $Post_t = 1$ if after April 1992
- $t$ = time trend

**What are we testing?**

We want $\gamma = 0$ (treatment and control have same pre-treatment trends).

**If $\gamma \neq 0$:** Parallel trends violated—DiD estimates may be biased.

---

**Question:** In which graph above would we expect $\gamma \neq 0$?

**Answer:** **Graph B and Graph D** (different pre-treatment slopes).

---

## Part II: Difference-in-Differences Calculation (8 minutes)

### Standard 2×2 DiD

**Data:**
| Group | Before (1991) | After (1993) | Difference |
|-------|---------------|--------------|------------|
| Treatment (NJ) | 18.5% | 19.2% | +0.7% |
| Control (PA) | 18.0% | 17.8% | -0.2% |
| **Difference** | +0.5% | +1.4% | **+0.9%** |

**DiD estimator:**
$$\hat{\tau}^{DiD} = (Y_{Treat,After} - Y_{Treat,Before}) - (Y_{Control,After} - Y_{Control,Before})$$
$$= (19.2 - 18.5) - (17.8 - 18.0) = 0.7 - (-0.2) = \mathbf{+0.9 \text{ percentage points}}$$

**Interpretation:** The minimum wage increase **increased** teenage employment by 0.9 percentage points in NJ relative to PA.

---

**Question:** Is this effect causal?

**Answer:** Only if:
1. **Parallel trends** holds (we checked graphically)
2. **No spillovers** (PA firms don't respond to NJ policy)
3. **No other confounders** (no other policy changes in NJ vs PA at the same time)

**If these assumptions hold:** Yes, this is the causal effect.

---

## Part III: Staggered DiD (7 minutes)

### Modern DiD Challenge: Staggered Treatment

**Problem:** In many real-world settings, treatment happens at different times for different units.

**Example:** Minimum wage increases across US states (2000-2020)

| State | MW increase year |
|-------|------------------|
| California | 2007 |
| New York | 2009 |
| Washington | 2010 |
| (20 more states) | 2011-2020 |
| Control states | Never |

**Classic 2×2 DiD doesn't work!** We have multiple treatment groups and multiple time periods.

---

### Task: Calculate DiD for Staggered Treatment

**Simple Data (for illustration):**

| State | Year | Employment | Treated? |
|-------|------|------------|----------|
| A | 2005 | 15.0 | No |
| A | 2007 | 15.2 | **Yes** (treated in 2007) |
| A | 2009 | 15.5 | Yes |
| B | 2005 | 14.5 | No |
| B | 2007 | 14.6 | No |
| B | 2009 | 14.4 | **Yes** (treated in 2009) |
| C | 2005 | 16.0 | No |
| C | 2007 | 16.1 | No |
| C | 2009 | 16.2 | No (never treated) |

---

**Naive approach:** Run a regression with state and year fixed effects:
$$Y_{st} = \alpha_s + \gamma_t + \tau D_{st} + \varepsilon_{st}$$

where $D_{st} = 1$ if state $s$ is treated in year $t$.

**Problem with this approach (Goodman-Bacon 2021, etc.):**
- Earlier-treated units serve as controls for later-treated units
- This can lead to **negative weights** on some treatment effects
- Estimates can be biased if treatment effects vary over time (heterogeneity)

---

### Correct Approach: Separate 2×2 Comparisons

**DiD #1: State A (treated 2007) vs State C (never treated)**

| Group | 2005 | 2007 | Δ |
|-------|------|------|---|
| A (treated 2007) | 15.0 | 15.2 | +0.2 |
| C (never treated) | 16.0 | 16.1 | +0.1 |
| **DiD** | | | **+0.1** |

**DiD #2: State B (treated 2009) vs State C (never treated)**

| Group | 2007 | 2009 | Δ |
|-------|------|------|---|
| B (treated 2009) | 14.6 | 14.4 | -0.2 |
| C (never treated) | 16.1 | 16.2 | +0.1 |
| **DiD** | | | **-0.3** |

---

**Question:** What is the overall treatment effect?

**Naive answer:** Average of DiD estimates = $(+0.1 + (-0.3))/2 = -0.1$

**Problem:** Are these comparable? State A and B might have different treatment effects (heterogeneity)!

**Modern solutions (Callaway & Sant'Anna 2021, etc.):**
- Compute separate DiD for each treatment cohort
- Weight appropriately
- Use never-treated or not-yet-treated as controls
- Report **event study** showing effects over time

---

### Event Study Plot (Example)

```
Effect on Employment
      |
  0.4 |              ●
      |            ↗
  0.2 |          ●     ●
      |        ↗
  0.0 |●---●--+---------------
      |       |
 -0.2 |       ↓
      +---|---|---|---|---|---
         -2  -1  0  +1  +2  +3
      Years relative to treatment
         (0 = year of MW increase)
```

**Interpretation:**
- No effect before treatment (parallel trends check!)
- Effect appears after treatment
- Effect grows over time (dynamic effect)

---

## Part IV: RDD Basics (Bonus - if time permits)

### Regression Discontinuity Design (RDD)

**Idea:** Exploit sharp cutoffs for treatment assignment.

**Example:** Effect of class size on student achievement
- **Running variable:** Enrollment (# of students)
- **Cutoff:** 40 students (maximum class size)
- **Treatment:** Small class (if enrollment = 41, split into 2 classes of ~20 each)

---

**DAG:**
```
Enrollment (X) → Treatment (D) → Achievement (Y)
```

**Key assumption:** Units just below vs just above cutoff are comparable (local randomization).

---

**RDD Estimator (Sharp RDD):**
$$\tau_{RDD} = \lim_{x \downarrow c} E[Y|X=x] - \lim_{x \uparrow c} E[Y|X=x]$$

**In words:** Compare outcomes just above vs just below the cutoff.

**Estimation:** Local linear regression on each side of cutoff.

```python
from sklearn.linear_model import LinearRegression

# Observations just below cutoff (c - h, c)
below = df[(df['enrollment'] >= c - h) & (df['enrollment'] < c)]
model_below = LinearRegression().fit(below[['enrollment']], below['achievement'])
y_below = model_below.predict([[c]])

# Observations just above cutoff (c, c + h)
above = df[(df['enrollment'] >= c) & (df['enrollment'] < c + h)]
model_above = LinearRegression().fit(above[['enrollment']], above['achievement'])
y_above = model_above.predict([[c]])

# RDD estimate
tau_rdd = y_above - y_below
```

---

**Question:** What is the bandwidth $h$?

**Answer:** The range around the cutoff to include in the regression. Too small → high variance. Too large → bias (includes units far from cutoff).

**Optimal bandwidth:** Use cross-validation or formal methods (Imbens & Kalyanaraman 2012).

---

## Summary: Key Takeaways

1. ✅ **Parallel trends:** Visual and statistical tests are crucial
2. ✅ **Classic DiD:** Simple 2×2 estimator for single treatment time
3. ✅ **Staggered DiD:** Use modern methods (Callaway & Sant'Anna, etc.)
4. ✅ **Event studies:** Plot effects over time to check dynamics
5. ✅ **RDD:** Exploit discontinuities for causal inference
6. ✅ **ML in DiD/RDD:** Can use flexible controls, but be careful about overfitting

---

## Connection to DML

**How does DML help in DiD and RDD?**

**DiD + DML:**
- Use ML to flexibly control for time-varying covariates
- Example: County-level DiD with many county characteristics
- DML allows Lasso/RF for nuisance models while estimating treatment effect

**RDD + DML:**
- Use ML for flexible functional form on each side of cutoff
- Avoid misspecification bias from linear/polynomial assumptions
- Still need to be careful about overfitting near discontinuity!

---

## Preview: HW3 Track B (DiD-DML)

In **HW3 Track B**, you will:
- Estimate the effect of minimum wage on employment using county-level panel data
- Implement standard DiD, DiD with controls, and DML-DiD
- Check parallel trends graphically
- Compare results across methods

**Dataset:** County employment data with staggered MW increases (300 counties × 20 time periods)

---

## Additional Resources

**Required Reading:**
- Angrist & Pischke (2009), *Mostly Harmless Econometrics*, Chapter 5 (DiD) and Chapter 6 (RDD)
- Callaway & Sant'Anna (2021), "Difference-in-Differences with Multiple Time Periods"

**Classic Papers:**
- Card & Krueger (1994), "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in NJ and PA"
- Lee & Lemieux (2010), "Regression Discontinuity Designs in Economics" (survey)

**Modern DiD Methods:**
- Goodman-Bacon (2021), "Difference-in-Differences with Variation in Treatment Timing"
- Borusyak, Jaravel, Spiess (2021), "Revisiting Event Study Designs: Robust and Efficient Estimation"

**Software:**
- Python: `differences` package for modern DiD
- Python: `rdd` or `rdrobust` for RDD
- R: `did` package (Callaway & Sant'Anna), `rdrobust` (Calonico et al.)

---

**For Discussion:**
- When does parallel trends seem plausible vs implausible?
- How do we choose control groups in practice?
- What if there's spillover between treated and control units?
- Can ML "fix" violations of parallel trends? (Spoiler: No! Assumptions still matter!)
