# Assignment 4: Heterogeneous Treatment Effects with Causal Forests

**Course:** ECON6083 - Machine Learning in Economics
**Weight:** 10% of Total Grade
**Due:** May 1 at 23:59 (1 week after Lecture 10)

---

## 📋 Assignment Overview

In this assignment, you will use **Causal Forests** to estimate **heterogeneous treatment effects** of a job training program. You'll identify which subgroups benefit most from the program, design an optimal targeting policy, and evaluate the policy gains from targeting versus uniform treatment.

**Dataset:** Simulated data from a randomized controlled trial (RCT) of a job training program with 2,000 participants.

**Goal:** Understand treatment effect heterogeneity and design optimal policy interventions.

---

## 🎯 Learning Objectives

By completing this assignment, you will:

1. Estimate **Conditional Average Treatment Effects (CATE)** using Causal Forests
2. Understand how treatment effects vary across individual characteristics
3. Design **optimal targeting policies** based on predicted treatment effects
4. Evaluate policy gains from targeted interventions vs uniform treatment
5. Interpret heterogeneous treatment effects from an economic perspective

---

## 📁 Files Provided

```
student-template/
├── README.md              # This file (instructions)
├── hw4_code.py            # Code template with TODOs
├── hw4_report.md          # Report template
└── requirements.txt       # Python dependencies

data/
├── job_training.csv       # Main dataset (2,000 observations)
└── data_documentation.md  # Data dictionary
```

---

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- `econml` - EconML library for Causal Forests
- `scikit-learn` - Machine learning utilities
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization

### Step 2: Load the Data

```python
import pandas as pd

df = pd.read_csv('../data/job_training.csv')
print(df.head())
print(df.describe())
```

**Dataset structure:**
- **Treatment:** `treated` (1 = received training, 0 = control)
- **Outcome:** `earnings_post` (earnings 18 months after program, in dollars)
- **Covariates:** `age`, `education`, `prior_earnings`, `female`, `married`, `children`

### Step 3: Complete the Code Template

Open `hw4_code.py` and complete all sections marked with `# TODO`.

### Step 4: Fill in the Report

Answer all questions in `hw4_report.md` based on your analysis.

### Step 5: Submit

Submit **both files** to Moodle:
```
hw4_code_<StudentID>_<LastName>.py
hw4_report_<StudentID>_<LastName>.md
```

---

## 📊 Assignment Structure

### Part I: Exploratory Analysis (10 points)

**Tasks:**

1. **Check covariate balance** between treated and control groups
   - Compare means of all covariates (age, education, etc.)
   - Is the randomization balanced? (t-tests or visual inspection)

2. **Visualize outcome distributions**
   - Histogram of `earnings_post` for treated vs control
   - Box plots by treatment status

3. **Calculate naive Average Treatment Effect (ATE)**
   - Simple difference in means: `mean(earnings_post | treated=1) - mean(earnings_post | treated=0)`
   - What is the overall impact of the program?

**Deliverable:** Summary statistics table, visualizations, naive ATE estimate

---

### Part II: Causal Forest Implementation (40 points)

#### Task A: Train Causal Forest (20 points)

Use `CausalForestDML` from the EconML package to estimate heterogeneous treatment effects.

**Implementation steps:**

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# Define features
X = df[['age', 'education', 'prior_earnings', 'female', 'married', 'children']]
Y = df['earnings_post']  # Outcome
T = df['treated']        # Treatment

# Nuisance models (for E[Y|X] and E[T|X])
model_y = RandomForestRegressor(n_estimators=100, random_state=42)
model_t = RandomForestRegressor(n_estimators=100, random_state=42)

# Causal Forest
cf = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    n_estimators=1000,      # Number of trees
    min_samples_leaf=10,    # Minimum samples per leaf
    honest=True,            # Honest splitting (prevents overfitting)
    random_state=42
)

# Fit the model
cf.fit(Y, T, X=X)

# Predict CATE for each individual
cate = cf.effect(X)
df['cate'] = cate
```

**Key points:**
- **Honest splitting:** Use `honest=True` to ensure unbiased CATE estimates
- **Sufficient trees:** Use at least 1,000 trees for stable estimates
- **Nuisance models:** RF models estimate E[Y|X] and E[T|X] to control for confounding

**Questions to answer:**
- What is the range of CATE estimates?
- What is the mean CATE? (Should be close to naive ATE)
- How much heterogeneity exists? (standard deviation of CATE)

---

#### Task B: Analyze Heterogeneity (20 points)

**Visualizations required:**

1. **CATE distribution**
   ```python
   import matplotlib.pyplot as plt

   plt.hist(cate, bins=30, edgecolor='black', alpha=0.7)
   plt.xlabel('CATE (Treatment Effect)')
   plt.ylabel('Frequency')
   plt.title('Distribution of Conditional Average Treatment Effects')
   plt.axvline(cate.mean(), color='red', linestyle='--', label=f'Mean CATE: ${cate.mean():.0f}')
   plt.legend()
   plt.show()
   ```

2. **CATE vs Education**
   - Scatter plot: x-axis = education, y-axis = CATE
   - Add horizontal line for mean CATE
   - Interpret: Do high school dropouts (education < 12) benefit more?

3. **CATE vs Prior Earnings**
   - Scatter plot: x-axis = prior_earnings, y-axis = CATE
   - Interpret: Do low-income or middle-income workers benefit more?

4. **CATE by Subgroups**
   - Box plots comparing CATE for:
     - Education < 12 vs Education ≥ 12
     - Female vs Male
     - Married vs Single

**Analysis questions:**
- Which characteristics are associated with higher treatment effects?
- Who benefits MOST from the program?
- Who benefits LEAST from the program?
- Provide economic interpretation (why might these patterns exist?)

---

### Part III: Policy Design and Evaluation (50 points)

#### Task A: Identify High Responders (15 points)

**Questions:**

1. What percentage of the population has CATE > $3,000?
   ```python
   pct_high = (cate > 3000).mean() * 100
   print(f"{pct_high:.1f}% have CATE > $3,000")
   ```

2. What are the **characteristics** of high responders?
   - Define high responders as CATE > 75th percentile
   - Compare their average age, education, prior_earnings to low responders (CATE < 25th percentile)

3. Create a summary table:

| Characteristic | High Responders (CATE > p75) | Low Responders (CATE < p25) | Difference |
|----------------|------------------------------|----------------------------|------------|
| Age            | ?                            | ?                          | ?          |
| Education      | ?                            | ?                          | ?          |
| Prior Earnings | ?                            | ?                          | ?          |
| % Female       | ?                            | ?                          | ?          |

**Deliverable:** Summary table + interpretation (2-3 sentences)

---

#### Task B: Design Targeting Policy (20 points)

**Scenario:** The job training program costs $5,000 per person. Due to budget constraints, you can only train **30% of applicants**.

**Two policy approaches:**

1. **Baseline: Uniform Random Selection**
   - Randomly select 30% of applicants
   - Average treatment effect among selected individuals

2. **Proposed: Targeting by Predicted CATE**
   - Select top 30% with highest predicted CATE
   - Average treatment effect among selected individuals

**Implementation:**

```python
import numpy as np

N = len(df)

# Baseline: Random selection
np.random.seed(42)
random_treated = np.random.choice(N, size=int(0.3*N), replace=False)
ate_random = cate[random_treated].mean()

# Targeting: Top 30% by CATE
top_30_idx = np.argsort(cate)[-int(0.3*N):]
ate_targeted = cate[top_30_idx].mean()

# Policy gain
gain_dollars = ate_targeted - ate_random
gain_pct = (gain_dollars / ate_random) * 100

print(f"Random selection: ${ate_random:.0f} per person")
print(f"Targeted selection: ${ate_targeted:.0f} per person")
print(f"Absolute gain: ${gain_dollars:.0f}")
print(f"Percentage gain: {gain_pct:.1f}%")
```

**Questions:**
- What is the average effect under random selection?
- What is the average effect under targeted selection?
- What is the percentage gain from targeting?
- Is targeting worth the additional implementation costs?

---

#### Task C: Policy Evaluation (15 points)

**1. Efficiency Gain**
- How much more effective is targeting compared to random selection?
- If the program budget is $1 million, how many additional dollars of earnings are generated by targeting?

**2. Budget Implications**
- Training costs $5,000/person
- Average earnings gain from targeting = ?
- What is the **Return on Investment (ROI)** under each policy?
  - ROI = (Earnings gain - Training cost) / Training cost
  - Example: If CATE = $3,800 and cost = $5,000, ROI = ($3,800 - $5,000) / $5,000 = -24% (net loss!)
- Is the program cost-effective under **any** policy?

**3. Equity Concerns**
- Who gets left behind with targeting?
- Compare characteristics of **selected** vs **not selected** under targeting policy
- Discuss potential equity/fairness concerns (2-3 sentences)

**4. Sensitivity Analysis**
- How do results change with different targeting thresholds?
- Create a table comparing top 20%, 30%, 40%, 50%

| Targeting Level | Avg CATE | Policy Gain vs Random |
|-----------------|----------|----------------------|
| Top 20%         | ?        | ?                    |
| Top 30%         | ?        | ?                    |
| Top 40%         | ?        | ?                    |
| Top 50%         | ?        | ?                    |

**Deliverable:** Policy comparison table + discussion (200-300 words)

---

## 📝 Report Structure

Fill in `hw4_report.md` with the following sections:

### Part I: Exploratory Analysis (10 points)
- Covariate balance table
- Outcome visualizations
- Naive ATE calculation

### Part II: Causal Forest Results (40 points)
- CATE distribution summary statistics
- 4 visualizations (CATE distribution, vs education, vs prior_earnings, by subgroups)
- Interpretation of heterogeneity patterns (150-200 words)

### Part III: Policy Analysis (50 points)
- High responders characterization table
- Targeting policy comparison (random vs targeted)
- Efficiency, equity, and sensitivity analysis (250-300 words)

**Total word count:** ~500-700 words (excluding tables)

---

## ⚠️ Common Mistakes to Avoid

### 1. Not Using Honest Splitting
**Mistake:** Setting `honest=False` in CausalForestDML
**Impact:** Overfit CATE estimates with exaggerated heterogeneity
**Solution:** Always use `honest=True`

### 2. Too Few Trees
**Mistake:** Using `n_estimators < 500`
**Impact:** Unstable CATE estimates that vary across runs
**Solution:** Use at least 1,000 trees

### 3. Confusing ATE with CATE
**Mistake:** Reporting mean CATE as "the treatment effect for everyone"
**Correct:** ATE = average effect, CATE_i = treatment effect for individual i

### 4. Wrong Targeting Direction
**Mistake:** Selecting individuals with LOWEST CATE (helping those who benefit least)
**Correct:** Use `np.argsort(cate)[-N:]` to get top N individuals

### 5. Ignoring Covariate Balance
**Mistake:** Not checking if treatment is randomized
**Impact:** If unbalanced, may need additional controls
**Check:** Compare covariate means between treated and control

### 6. No Economic Interpretation
**Mistake:** Only reporting numbers without explaining what they mean
**Solution:** Always interpret results in economic terms (e.g., "Low-educated workers benefit $1,500 more on average because...")

---

## 💡 Tips for Success

### Code Quality
- Use clear variable names (`cate`, `ate_targeted`, not `x`, `y`)
- Add comments explaining each step
- Use `random_state=42` for reproducibility
- Verify your code runs without errors before submission

### Visualization
- Always label axes clearly with units (e.g., "CATE (dollars)")
- Use titles that describe what the plot shows
- Include legends when comparing groups
- Use color strategically (e.g., red for treated, blue for control)

### Economic Analysis
- Provide concrete examples ("A 10% increase in X leads to...")
- Discuss policy implications (Who should receive treatment? Why?)
- Consider trade-offs (efficiency vs equity)
- Support claims with evidence from your results

### Report Writing
- Be concise but thorough
- Use tables to organize information clearly
- Cite specific numbers from your analysis
- Proofread for clarity and grammar

---

## 📚 Background Reading

### Required
- Lecture 6 slides on Heterogeneous Treatment Effects
- In-Class Exercise 6 (HTE & CATE)
- Wager & Athey (2018), "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" (provided on Moodle)

### Optional
- Athey & Wager (2019), "Estimating Treatment Effects with Causal Forests: An Application"
- EconML documentation: https://econml.azurewebsites.net/
- CausalForestDML guide: https://econml.azurewebsites.net/spec/estimation/dml.html

---

## 🔍 Frequently Asked Questions

**Q1: All my CATE estimates are the same. What's wrong?**
A: Check if you passed X (features) to `cf.fit()`. Make sure X has variation.

**Q2: Some people have negative CATE. Is that possible?**
A: Yes! Some individuals may not benefit from treatment (or even be harmed). This is what heterogeneity means.

**Q3: Targeting gives worse results than random. What did I do wrong?**
A: Check the direction of `np.argsort()`. Use `argsort(cate)[-N:]` for top N, not `argsort(cate)[:N]`.

**Q4: How many trees should I use?**
A: Start with 1,000. More trees = more stable estimates but slower computation. 1,000-2,000 is a good range.

**Q5: Should I control for all covariates?**
A: Yes, include all available covariates in X. Causal Forests automatically select relevant ones.

**Q6: My CATE range is very wide ($1,000 to $8,000). Is this normal?**
A: Check your results. Expected range is $1,000 to $5,000. If wider, you may have an error.

---

## 🎯 Grading Rubric (100 points)

| Component | Points | Description |
|-----------|--------|-------------|
| **Part I: Exploration** | 10 | Balance checks, outcome viz, naive ATE |
| **Part II: Causal Forest** | 40 | |
| - Training CF | 20 | Correct implementation, hyperparameters |
| - Heterogeneity analysis | 20 | Visualizations, patterns, interpretation |
| **Part III: Policy Design** | 50 | |
| - High responders | 15 | Identification, characterization |
| - Targeting policy | 20 | Implementation, gain calculation |
| - Policy evaluation | 15 | Efficiency, equity, sensitivity |
| **Code Quality** | -5 max | Deductions for errors, poor organization |
| **Bonus (optional)** | +10 max | Advanced analyses (see below) |

**Total:** 100 points (can earn up to 110 with bonus, capped at 100)

---

## 🌟 Bonus Opportunities (Optional, +10 max)

1. **Confidence Intervals for CATE** (+3 points)
   - Use bootstrap to construct 95% CI for individual CATE
   - Visualize CIs for a random sample of 50 individuals

2. **Multiple Policies** (+3 points)
   - Compare top 10%, 20%, 30%, ..., 90%
   - Plot policy curve showing average CATE vs fraction treated

3. **Subgroup Analysis** (+2 points)
   - Estimate separate Causal Forests for men vs women
   - Compare heterogeneity patterns across subgroups

4. **Cost-Benefit Analysis** (+2 points)
   - Incorporate program costs into targeting rule
   - Select only individuals with CATE > cost ($5,000)
   - Calculate net benefit under this policy

**Note:** Bonus work should be clearly labeled in your report and code.

---

## 📅 Submission Checklist

Before submitting, ensure:

- [ ] Code runs without errors
- [ ] All TODOs in `hw4_code.py` are completed
- [ ] All questions in `hw4_report.md` are answered
- [ ] Visualizations are clear and properly labeled
- [ ] Tables are complete and formatted correctly
- [ ] Economic interpretations are provided (not just numbers)
- [ ] Word counts are within guidelines
- [ ] Files are named correctly: `hw4_code_<ID>_<Name>.py` and `hw4_report_<ID>_<Name>.md`
- [ ] Both files are uploaded to Moodle before deadline

---

## 🆘 Getting Help

- **Course forum:** Post questions on Moodle discussion board
- **Office hours:** [Check course website for schedule]
- **Email TA:** For clarification on assignment requirements (not debugging help)
- **Review materials:** Lecture 6 slides, In-Class Exercise 6

---

## ⚖️ Academic Integrity

- You may discuss **concepts** with classmates
- You may NOT share **code** or **reports**
- All submitted work must be your own
- Violations will be reported to the university

---

**Good luck! 🚀**

**Assignment created by:** ECON6083 Teaching Team
**Last updated:** 2026-02-10
