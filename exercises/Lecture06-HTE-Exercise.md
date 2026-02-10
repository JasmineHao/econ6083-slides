# In-Class Exercise: Heterogeneous Treatment Effects (HTE)

**Course:** ECON6083 - Machine Learning in Economics
**Topic:** Conditional Average Treatment Effects (CATE) and Policy Targeting
**Time:** 25 minutes

---

## Part I: Conceptual Understanding (10 minutes)

### Question 1: When is CATE Identification Possible?

The **Conditional Average Treatment Effect (CATE)** is defined as:
$$\tau(x) = E[Y_i(1) - Y_i(0) | X_i = x]$$

For CATE to be identified at $X = x$, we need:

**Assumption 1 (Unconfoundedness):** $(Y_i(1), Y_i(0)) \perp D_i | X_i$

**Assumption 2 (Overlap):** $0 < P(D_i = 1 | X_i = x) < 1$

---

**Scenario:** Job training program evaluation

You want to estimate the CATE of training on earnings for:
- **Group A:** College graduates with 10 years experience
- **Group B:** High school dropouts with 0 years experience

**Given data:**
- College graduates: 80% treated, 20% control
- High school dropouts: 5% treated, 95% control

**Questions:**

**(a)** For which group is CATE identification more reliable? Why?

- [ ] A. Group A (college graduates)
- [ ] B. Group B (high school dropouts)
- [ ] C. Both equally reliable
- [ ] D. Neither can be identified

**Answer:** ___________

**Explanation:**

Group A satisfies overlap better (80-20 split is closer to balance than 5-95). For Group B, we have very few treated observations (5%), which means:
- High variance in CATE estimates
- Extrapolation risk (comparing very different types of people)
- Overlap assumption is technically satisfied but practically weak

**Key insight:** CATE estimation requires sufficient data in BOTH treatment groups for each subgroup $X = x$.

---

**(b)** What happens if we try to estimate CATE for a region with NO overlap?

**Example:** Suppose all college graduates are treated (100%) and no controls.

What does this mean?

**Answer:**

We **cannot identify** $\tau(college) = E[Y(1) - Y(0) | college]$ because we never observe $Y(0)$ for college graduates. We would need to extrapolate from other groups (high school grads), which relies on untestable assumptions.

**Practical implication:** When using Causal Forests or other HTE methods, be cautious about interpreting CATE for subgroups with poor overlap.

---

### Question 2: Honest Splitting in Causal Trees

Causal Trees use **honest splitting** to avoid overfitting in HTE estimation.

**Standard Tree (NOT honest):**
1. Use all data to choose split points (maximize treatment effect heterogeneity)
2. Use the same data to estimate leaf-level treatment effects

**Problem:** Overfits! Splits are chosen to maximize differences in the training sample, leading to exaggerated HTE.

**Honest Tree:**
1. Split sample into two parts: **sample 1** and **sample 2**
2. Use **sample 1** to build the tree structure (choose splits)
3. Use **sample 2** to estimate treatment effects within each leaf

---

**Question:** Why does honest splitting reduce overfitting?

Fill in the blanks:

Honest splitting separates the sample used for ___________ (structure learning / effect estimation) from the sample used for ___________ (structure learning / effect estimation). This prevents the tree from selecting splits that are ___________ (optimal / overfitted) to noise in the training data.

**Answers:**
- Structure learning
- Effect estimation
- Overfitted

**Analogy:** Similar to cross-validation—we don't evaluate a model on the same data used to train it.

---

## Part II: CATE Interpretation Exercise (15 minutes)

### Scenario: Job Training Program

You have trained a **Causal Forest** on data from a job training RCT:
- **Treatment:** Job training program (D = 1) vs no training (D = 0)
- **Outcome:** Annual earnings 2 years after training ($)
- **Features (X):**
  - `age`: 18-65
  - `education`: years of schooling (8-20)
  - `prior_earnings`: earnings before training ($0-$80k)
  - `industry`: manufacturing, service, retail, etc.

**Given:** CATE predictions for 1,000 individuals (already computed and provided)

---

### Task 1: Visualize CATE Heterogeneity (5 min)

You are provided with a dataset:

```python
# Already computed (you don't need to train the model)
df['cate'] = causal_forest.predict(X_test)  # CATE for each individual

# Columns: age, education, prior_earnings, industry, cate
```

**Task:** Create two scatter plots:

**Plot 1:** CATE vs Education
```python
import matplotlib.pyplot as plt

# TODO: Scatter plot of CATE vs education
plt.scatter(df['education'], df['cate'], alpha=0.3)
plt.xlabel('Years of Education')
plt.ylabel('CATE ($ earnings gain from training)')
plt.axhline(y=0, color='red', linestyle='--', label='No effect')
plt.title('Treatment Effect Heterogeneity by Education')
plt.legend()
plt.show()
```

**Plot 2:** CATE vs Prior Earnings
```python
# TODO: Same for prior_earnings
plt.scatter(df['prior_earnings'], df['cate'], alpha=0.3)
plt.xlabel('Prior Earnings ($)')
plt.ylabel('CATE ($ earnings gain)')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Treatment Effect Heterogeneity by Prior Earnings')
plt.show()
```

---

**Expected pattern (for discussion):**

Imagine you see:
- **Plot 1:** CATE decreases with education (high school dropouts benefit more)
- **Plot 2:** CATE is highest for middle-income workers (not very poor or very rich)

**Question:** What is the economic interpretation?

**Answer:**

- **Education:** Training provides basic skills most valuable to those who lack them (dropouts). College grads already have skills, so training has less marginal value.
- **Prior Earnings:** Very low earners might face other barriers (health, discrimination) that training alone can't fix. Very high earners have less to gain. Middle earners benefit most.

---

### Task 2: Identify High Responders (5 min)

**Question:** Who benefits most from the training program?

```python
# TODO: Find top 10% of individuals with highest CATE
threshold_90 = df['cate'].quantile(0.90)
high_responders = df[df['cate'] >= threshold_90]

print(f"Top 10% have CATE >= ${threshold_90:.0f}")
print("\nCharacteristics of high responders:")
print(high_responders[['age', 'education', 'prior_earnings']].describe())
```

**Example output:**
```
Top 10% have CATE >= $8,500

Characteristics of high responders:
         age  education  prior_earnings
mean    35.2       11.5          $25,000
std      8.1        2.3           $8,000
min     22.0        8.0          $12,000
max     52.0       16.0          $42,000
```

**Interpretation:** High responders are:
- Mid-career (age ~35)
- Moderate education (11-12 years = some college/high school diploma)
- Lower-middle income (prior earnings ~$25k)

**Policy implication:** Target training programs at this demographic for maximum impact.

---

### Task 3: Design Targeting Policy (5 min)

**Scenario:** You have a budget to train only 30% of applicants.

**Question:** How much better is targeting (based on CATE) vs uniform random assignment?

**Approach A: Uniform Random (current policy)**
- Randomly select 30% of applicants
- Average treatment effect on the treated (ATT) = average CATE among treated

**Approach B: Targeting (proposed policy)**
- Select top 30% by predicted CATE
- ATT = average CATE among top 30%

```python
# Approach A: Random selection
random_sample = df.sample(frac=0.3, random_state=42)
att_random = random_sample['cate'].mean()

# Approach B: Targeting
top_30 = df.nlargest(int(0.3 * len(df)), 'cate')
att_targeted = top_30['cate'].mean()

print(f"Uniform random policy: ATT = ${att_random:.0f}")
print(f"Targeted policy: ATT = ${att_targeted:.0f}")
print(f"Gain from targeting: ${att_targeted - att_random:.0f} per person")
print(f"Total gain (for 300 people): ${(att_targeted - att_random) * 300:.0f}")
```

**Expected output:**
```
Uniform random policy: ATT = $4,200
Targeted policy: ATT = $7,800
Gain from targeting: $3,600 per person
Total gain (for 300 people): $1,080,000
```

**Interpretation:**

By targeting the program at those who benefit most (based on CATE predictions), we increase the average effect from $4,200 to $7,800—an 86% improvement!

With 300 trainees, this is an extra $1.08 million in total earnings gains.

**Economic value of HTE:** This is why economists care about heterogeneous treatment effects—it enables efficient resource allocation.

---

## Part III: Discussion Questions (remaining time)

### Q1: Ethical Considerations

If we use CATE to target training programs, we would:
- Give training to those predicted to benefit most
- Deny training to those predicted to benefit less

**Ethical concerns:**
1. What if predictions are wrong for some individuals?
2. Should we deny training to someone just because the average person like them doesn't benefit?
3. Potential for discrimination if CATE varies by race/gender?

**Question:** How would you balance efficiency (targeting) vs fairness (equal access)?

**Possible answers:**
- Use CATE for prioritization when demand exceeds supply, but don't exclude anyone
- Set minimum eligibility (everyone above threshold gets training)
- Audit CATE predictions for bias across demographic groups
- Randomize within CATE bands (e.g., among "high responders")

---

### Q2: When NOT to Trust CATE Predictions

**Red flags:**
1. **Poor overlap:** If a subgroup has very few treated or controls, CATE is noisy
2. **Small sample:** Causal Forests need N > 1000 for stable results
3. **Confounding:** If unconfoundedness fails for some subgroups, CATE is biased
4. **Overfitting:** If honest splitting wasn't used, CATE is exaggerated

**Question:** How can you check if CATE predictions are reliable?

**Answer:**
- Check overlap: Plot propensity scores by subgroup
- Cross-validation: Train-test split, check if CATE patterns replicate
- Placebo tests: Check for heterogeneity in pre-treatment outcomes (should be zero)
- Sensitivity analysis: Re-run with different ML methods, check consistency

---

## Summary: Key Takeaways

1. ✅ **CATE requires overlap** for each subgroup $X = x$
2. ✅ **Honest splitting** prevents overfitting in Causal Trees
3. ✅ **CATE visualization** reveals who benefits most/least
4. ✅ **Targeting policies** can substantially improve welfare
5. ✅ **Ethical trade-offs** between efficiency and fairness
6. ✅ **Validation is crucial** to avoid over-interpreting noise

---

## Preview: Midterm Project & HW4

**Midterm Project (next week):**
- Apply DML to estimate **average** treatment effect (ATE)
- Simpler than HTE, good for learning the basics

**HW4 (later):**
- Apply Causal Forests to estimate **heterogeneous** treatment effects
- Design optimal targeting policy
- You'll do exactly what we practiced today, but on a full dataset!

---

## Additional Resources

**Required Reading:**
- Wager & Athey (2018), "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests," *JASA*
- Athey & Imbens (2016), "Recursive Partitioning for Heterogeneous Causal Effects," *PNAS*

**Software:**
- Python: `econml.grf.CausalForest` or `econml.dml.CausalForestDML`
- R: `grf` package (original implementation)

**Optional:**
- Athey & Wager (2019), "Estimating Treatment Effects with Causal Forests: An Application," *Observational Studies*

---

**For Discussion:**
- What economic applications would benefit from HTE analysis?
- How do we communicate CATE uncertainty to policymakers?
- What if CATE varies by race or gender—how do we handle that?
