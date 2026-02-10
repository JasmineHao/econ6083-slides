# In-Class Exercise: DAGs and Structural Causal Models

**Course:** ECON6083 - Machine Learning in Economics
**Topic:** Causal Graphs, Identification, and the Backdoor Criterion
**Time:** 25 minutes

---

## Part I: Draw the DAG (10 minutes)

### Scenario: Effect of Education on Wages

You want to estimate the causal effect of education on wages.

**Variables:**
- **D** = Education (years of schooling) - **Treatment**
- **Y** = Wages (hourly wage) - **Outcome**
- **U** = Ability (unobserved - cognitive skills, motivation, etc.)
- **F** = Family Income (parents' income during childhood)
- **T** = Test Scores (SAT/ACT scores)

**Relationships (based on economic theory):**
1. Family income affects education (richer families can afford more schooling)
2. Family income affects test scores (better schools, tutoring)
3. Ability affects education (smarter students go further)
4. Ability affects test scores
5. Ability affects wages (directly)
6. Education affects wages (the causal effect we want!)
7. Test scores affect college admission, hence education

---

### Task 1: Draw the DAG

Draw a directed acyclic graph (DAG) showing all relationships.

**Instructions:**
- Use arrows to show causal relationships (A → B means "A causes B")
- Include all 5 variables: D, Y, U, F, T
- Mark U as unobserved (use a circle or dashed box)

**Your DAG here:**

```
                    U (Ability - Unobserved)
                   / | \
                  /  |  \
                 /   |   \
                ↓    ↓    ↓
               T     D    Y
              ↗     ↑
             /      |
            /       |
           F -------+
    (Family Income)
```

**Explanation:**
- **F → D**: Family income increases education (afford college, no need to work)
- **F → T**: Family income improves test scores (better schools)
- **U → D**: Ability affects educational attainment
- **U → T**: Ability determines test performance
- **U → Y**: Ability directly affects wages (even without education)
- **T → D**: Test scores determine college admission
- **D → Y**: Education causes higher wages (the effect we want!)

---

### Question 1: Confounders, Mediators, or Colliders?

For the causal effect of **D** (education) on **Y** (wages), classify each variable:

**(a) Family Income (F)**

Is F a:
- [ ] Confounder (causes both D and Y)
- [ ] Mediator (on the causal path D → ? → Y)
- [ ] Collider (caused by both D and Y)

**Answer:** Neither a confounder nor a collider nor a mediator!

**Explanation:**
- F causes D (✓)
- But F does NOT directly cause Y
- F only affects Y through D (via the path F → D → Y)
- F is an **ancestor of treatment** but not a confounder of the D→Y relationship

**Key insight:** A variable must be a common cause of BOTH treatment and outcome to be a confounder.

---

**(b) Ability (U)**

Is U a:
- [ ] Confounder
- [ ] Mediator
- [ ] Collider

**Answer:** **Confounder**

**Explanation:**
- U causes D (smarter people get more education)
- U causes Y (smarter people earn more, even without education)
- This opens a backdoor path: D ← U → Y
- If we don't control for U, we'll overestimate the effect of education (confounding bias)

**Problem:** U is unobserved! This is the classic ability bias problem in returns-to-education literature.

---

**(c) Test Scores (T)**

Is T a:
- [ ] Confounder
- [ ] Mediator
- [ ] Collider

**Answer:** **Collider**

**Explanation:**
- T is caused by both F and U
- T also causes D
- T is on the path F → T ← U, where T is a collider
- **Key question:** Should we control for T?

---

### Question 2: Should We Control for Test Scores?

**Two students debate:**

**Student A:** "We should control for test scores because they affect education. Controlling for T will reduce bias."

**Student B:** "We should NOT control for test scores because T is a collider. Controlling for T will CREATE bias."

**Who is correct?**

**Answer:** Both are partially right, but **Student B's concern is more serious**.

**Why controlling for T is problematic:**

1. **Collider bias:** T is caused by both F and U. Conditioning on T induces a spurious association between F and U (collider bias / Berkson's paradox).

2. **Example:** Among students with the same test score (say, SAT = 1400):
   - Low ability students likely had high family income (could afford tutoring)
   - High ability students likely had low family income (raw talent)
   - This creates a negative correlation between F and U within each test score level!

3. **Result:** Controlling for T can actually INCREASE bias rather than reduce it.

**When would controlling for T be okay?**
- If T were a pure mediator (D → T → Y), we could control for it (though we'd shut down one causal pathway)
- If there were no U (ability), T would just be a post-treatment variable

**Conclusion:** DON'T control for T. It's a collider and introduces bias.

---

## Part II: Backdoor Criterion (10 minutes)

The **Backdoor Criterion** (Pearl, 1995) tells us which variables to control for to identify causal effects.

**Definition:** A set of variables Z satisfies the backdoor criterion for the effect of D on Y if:
1. **No descendants:** Z does not contain any descendants of D
2. **Blocks all backdoor paths:** Z blocks all paths from D to Y that have an arrow INTO D

---

### Task: Apply the Backdoor Criterion

For each DAG below, identify:
- All backdoor paths from D to Y
- A valid adjustment set Z (variables to control for)

---

#### **DAG A:**

```
    U
   / \
  ↓   ↓
  D → Y
```

**Backdoor paths:**
- D ← U → Y

**Valid adjustment set:**
- Z = {U}

**Explanation:** The path D ← U → Y is a backdoor path (arrow into D). Controlling for U blocks this path. No other paths exist.

---

#### **DAG B:**

```
  D → M → Y
```

**Backdoor paths:**
- None! (No arrows into D)

**Valid adjustment set:**
- Z = {} (empty set - no controls needed)

**Explanation:** There are no backdoor paths, so no confounding. We don't need to control for anything. D is as-if randomized.

**Note:** M is a mediator. We should NOT control for M if we want the total effect of D on Y.

---

#### **DAG C:**

```
    X → D → Y
    ↓       ↑
    +-- U --+
```

(U is unobserved)

**Backdoor paths:**
- D ← X → U → Y

**Valid adjustment set:**
- Z = {X} is NOT sufficient (doesn't block the full path)
- Need to control for X AND U
- But U is unobserved!

**Conclusion:** The causal effect is NOT identified from observed data alone. We have unobserved confounding (U).

**Solution strategies:**
- Instrumental variables (if we have a valid instrument)
- Panel data with fixed effects (if U is time-invariant)
- Difference-in-differences (if there's a natural experiment)

---

#### **DAG D (Trickier):**

```
    C
   / \
  ↓   ↓
  D → Y
  ↑
  X
```

**Backdoor paths:**
- D ← X (stops here, no path to Y)
- D ← C → Y

**Valid adjustment set:**
- Z = {C} (blocks the D ← C → Y path)
- OR Z = {X, C} (also valid but includes more than necessary)

**Question:** Should we control for X?

**Answer:** No need to control for X. It only affects D (causes treatment) but not Y (no path to outcome). Including X doesn't hurt, but it's not necessary for identification.

**Key insight:** The backdoor criterion tells us the **minimal** set needed. Extra controls are harmless but reduce efficiency.

---

## Part III: Economic Application (5 minutes)

### Case Study: Returns to Education with Ability Bias

**Standard Mincer equation:**
$$\log(wage) = \beta_0 + \beta_1 \cdot education + \beta_2' X + \varepsilon$$

**DAG:**
```
          Ability (U)
           /   \
          ↓     ↓
    Education → Wage
       ↑
   Family Income (F)
```

---

**Question 1:** If we run OLS without controlling for ability (U), what is the bias?

**Answer:**

$$\hat{\beta}_1^{OLS} = \beta_1 + \underbrace{\frac{Cov(D, U)}{Var(D)} \cdot \beta_U}_{\text{Ability Bias}}$$

- $\beta_1$ = true causal effect of education
- $\beta_U$ = effect of ability on wages
- $Cov(D, U) > 0$ (smarter people get more education)
- $\beta_U > 0$ (ability increases wages)

**Result:** $\hat{\beta}_1^{OLS} > \beta_1$ (overestimate returns to education)

**Magnitude:** Estimates suggest OLS overstates returns by 10-30% (Card 1999, 2001).

---

**Question 2:** Common "solutions" in the literature

| Method | What it does | Does it solve ability bias? |
|--------|--------------|----------------------------|
| Control for observables (X) | Include age, race, family background | ⚠️ Partial (only if X captures U) |
| Instrumental Variables (IV) | Use distance to college as instrument for D | ✅ Yes (if valid IV) |
| Twin studies | Compare twins with different education | ✅ Yes (U is same for twins) |
| Fixed effects | Within-person variation over time | ⚠️ Only if education changes |
| RDD | Exploit cutoffs (e.g., birthdate → compulsory schooling) | ✅ Yes (local to cutoff) |

**Best solution:** IV or natural experiments (RDD, DiD) are most convincing.

---

## Summary: Key Takeaways

1. ✅ **DAGs clarify assumptions**: Make causal relationships explicit
2. ✅ **Confounders vs Colliders**: Different roles, different implications
3. ✅ **Backdoor criterion**: Systematic way to find valid adjustment sets
4. ✅ **Collider bias**: Controlling for colliders creates bias!
5. ✅ **Unobserved confounding**: Some DAGs imply non-identification
6. ✅ **Economic applications**: Ability bias, omitted variable bias, etc.

---

## Connection to DML

**How does this relate to DML (Lecture 5)?**

DML assumes **unconfoundedness**:
$$(Y(1), Y(0)) \perp D | X$$

**In DAG terms:** This means:
- All backdoor paths from D to Y are blocked by X
- X satisfies the backdoor criterion
- No unobserved confounders (like U in ability bias example)

**If the DAG shows unobserved confounding:**
- DML will NOT recover the causal effect
- Need stronger methods: IV, panel data, DiD, RDD

**DAG is a tool for assessing identification BEFORE running DML!**

---

## Preview: Lecture 8 (IV-DML)

Next lecture, we'll see how to combine IV (which solves unobserved confounding) with DML (which handles high-dimensional controls).

**Example DAG with IV:**
```
       U (unobserved)
       ↓           ↓
   Z → D → Y

Z = Instrument (e.g., draft lottery for military service)
D = Treatment (military service)
Y = Outcome (earnings)
U = Unobserved ability/health
```

**IV solves the U problem by using variation in D caused only by Z.**

---

## Additional Resources

**Required Reading:**
- Pearl (2009), *Causality: Models, Reasoning, and Inference* (Chapters 1-3)
- Pearl, Glymour, Jewell (2016), *Causal Inference in Statistics: A Primer*

**Software:**
- Python: `causalgraphicalmodels` package
- R: `dagitty` package
- Web: DAGitty (browser-based tool): http://dagitty.net/

**Optional:**
- Cinelli, Forney, Pearl (2022), "A Crash Course in Good and Bad Controls"
- Huntington-Klein (2021), *The Effect* (free online book with great DAG examples)

---

## Practice Exercise (Homework - Optional)

**Draw the DAG for your Midterm Project:**
1. Identify treatment, outcome, and all relevant variables
2. Draw arrows based on economic theory
3. Apply backdoor criterion—what should you control for?
4. Check if unconfoundedness is plausible

This will strengthen your identification argument in the Midterm report!

---

**For Discussion:**
- Can we ever be 100% sure we've included all confounders in X?
- What if the "true" DAG is unknown or debated?
- How do DAGs relate to randomized experiments? (Hint: Randomization cuts all arrows INTO treatment!)
