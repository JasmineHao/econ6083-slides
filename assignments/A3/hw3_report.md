# Homework 3 Report: Demand Estimation with IV-DML

**Student Name:** ___________________________
**Student ID:** ___________________________
**Date:** ___________________________

---

## Part I: Data Exploration (10 points)

### Summary Statistics

Copy the summary statistics from your code output:

```
[Paste summary statistics here from df.describe()]
```

### Visualization Analysis

**Figure 1: Price vs Quantity**

![Price vs Quantity](price_quantity_scatter.png)

**Observation:**

_[Describe the pattern you observe. Does it show a downward-sloping relationship as expected for a demand curve?]_

---

**Figure 2: First Stage (Cost Shock vs Price)**

![First Stage](first_stage_plot.png)

**Observation:**

_[Does cost_shock have a positive relationship with price? This would support the relevance assumption for IV.]_

---

### Discussion: Why is Price Endogenous?

_[Write 100-150 words explaining why price is endogenous in demand estimation. Hint: Discuss simultaneity bias - prices and quantities are determined jointly by supply and demand equilibrium.]_

**Your answer:**

Price is endogenous in demand estimation because...

---

## Part II: Estimation Results (50 points)

### Comparison Table

| Method | Elasticity | Std Error | 95% CI | First Stage F |
|--------|-----------|-----------|--------|---------------|
| OLS | [fill in] | [fill in] | [[fill in], [fill in]] | - |
| 2SLS | [fill in] | [fill in] | [[fill in], [fill in]] | [fill in] |
| DML-IV | [fill in] | [fill in] | [[fill in], [fill in]] | - |

**Note:** Copy values from your code output (`comparison_table.csv`).

---

## Part III: Economic Analysis (40 points)

### 1. Comparison of Methods (15 points)

**(200-250 words)**

#### Why is OLS Biased?

_[Explain the simultaneity problem in demand estimation. Price and quantity are determined jointly by supply and demand. The OLS estimate is biased because price is correlated with the unobserved demand shock (ε_demand).]_

**Your answer:**

OLS is biased because...

**Direction of bias:**

_[Is OLS biased upward (toward zero) or downward (away from zero)? Explain using the formula for omitted variable bias.]_

---

#### Do 2SLS and DML-IV Give Similar Results?

_[Compare your 2SLS and DML-IV estimates. Are they close? If yes, this suggests the relationship is approximately linear. If no, discuss what might explain the difference.]_

**Your answer:**

My 2SLS estimate is _____ and DML-IV estimate is _____. They are [similar/different] because...

---

#### When Would DML-IV Outperform 2SLS?

_[Discuss scenarios where DML-IV's flexibility would be advantageous:]_

**Three scenarios:**

1. **High-dimensional controls:** When there are many control variables (e.g., hundreds of product characteristics), ML can handle them better than 2SLS.

2. **Nonlinear relationships:** If the relationship between price and quantity is nonlinear (e.g., kinked demand curves), ML can capture this while 2SLS assumes linearity.

3. **Complex interactions:** [Your example here]

---

### 2. Economic Interpretation (15 points)

**(200-250 words)**

Assume your 2SLS estimate of price elasticity is **[fill in your actual estimate]**.

#### What Does This Number Mean?

A price elasticity of _____ means that...

**Example calculation:**
- If price increases by 10%, quantity demanded will [increase/decrease] by ___%.
- If price decreases by 5%, quantity demanded will [increase/decrease] by ___%.

---

#### Is Demand Elastic or Inelastic?

**Classification:**
- Elastic demand: |elasticity| > 1
- Unit elastic: |elasticity| = 1
- Inelastic demand: |elasticity| < 1

**Your answer:**

Based on my estimate of _____, demand is [elastic/inelastic/unit elastic] because...

**Economic implication:**

When demand is elastic, consumers are [very/not very] responsive to price changes. This typically occurs for products that have [many/few] substitutes available.

---

#### Revenue Implications

**Total Revenue = Price × Quantity**

**If demand is elastic (|ε| > 1):**
- Increasing price → Quantity falls by more than price rises → Revenue decreases
- Decreasing price → Quantity rises by more than price falls → Revenue increases

**If demand is inelastic (|ε| < 1):**
- Increasing price → Quantity falls by less than price rises → Revenue increases
- Decreasing price → Quantity rises by less than price falls → Revenue decreases

**Your recommendation:**

Based on my elasticity estimate of _____, the firm should [raise/lower/keep constant] prices because...

---

#### Revenue-Maximizing Price

**Formula:** At the revenue-maximizing price, elasticity = -1 (unit elastic).

If your estimate is |ε| = 1.2 (elastic), the current price is [below/above] the revenue-maximizing level.

**Your analysis:**

_[Should the firm adjust prices? By how much? What are the trade-offs?]_

---

### 3. IV Validity Discussion (10 points)

**(200-250 words)**

Discuss the validity of **cost_shock** as an instrument for **price**.

---

#### Assumption 1: Relevance

**Question:** Does cost_shock predict price?

**Evidence:**
- First stage F-statistic from my 2SLS results: F = _____
- Rule of thumb: F > 10 indicates a strong instrument

**Your assessment:**

My F-statistic is _____, which [does/does not] exceed 10. This suggests the instrument is [strong/weak].

[If F < 10: Discuss implications - estimates may be biased, SEs may be unreliable]

---

#### Assumption 2: Exclusion Restriction

**Question:** Does cost_shock affect quantity ONLY through its effect on price?

**Economic argument:**

Cost shocks (e.g., changes in shipping costs, oil prices, raw material costs) directly affect firms' production costs. This shifts the supply curve, changing the equilibrium price. However, cost shocks should NOT directly affect consumer demand - they don't change consumer preferences or income.

**Potential threats to validity:**

1. **Quality correlation:** If cost shocks are correlated with product quality changes (e.g., expensive inputs = better quality), they might directly affect demand.

2. **Seasonal patterns:** If cost shocks follow seasonal patterns that also affect demand (e.g., holiday shipping costs), the exclusion restriction is violated.

3. [Your additional threat]

**Your assessment:**

I believe the exclusion restriction is [plausible/questionable] because...

---

#### Assumption 3: Monotonicity

**Question:** Does higher cost always lead to higher price?

**Your answer:**

In competitive markets, firms pass on cost increases to consumers (higher cost → higher price). This seems plausible because...

[Any scenarios where this might fail?]

---

#### Overall IV Credibility

**Conclusion:**

Based on the three assumptions:
- Relevance: [✓ / ⚠️ / ✗]
- Exclusion restriction: [✓ / ⚠️ / ✗]
- Monotonicity: [✓ / ⚠️ / ✗]

I conclude that the IV is [credible / somewhat credible / not credible] for estimating price elasticity.

The main threat to validity is ____________.

---

## Reflection (Optional - No Points)

### What was the most challenging part of this assignment?

_[Your answer]_

---

### What did you learn about IV methods and DML?

_[Your answer]_

---

### How might IV-DML be useful for your own research?

_[Your answer]_

---

## Declaration

I hereby declare that this assignment is my own work and that I have not copied from any other student's work or from any other source except where due acknowledgment is made explicitly in the text.

**Signature:** ___________________________

**Date:** ___________________________

---

## Submission Checklist

Before submitting, make sure you have:

- [ ] Completed all estimation tasks (OLS, 2SLS, DML-IV)
- [ ] Filled in comparison table with actual values
- [ ] Written 200+ word analysis for each section in Part III
- [ ] Discussed all three IV assumptions
- [ ] Included plots (price_quantity_scatter.png, first_stage_plot.png)
- [ ] Code file (`hw3_code_*.py`) runs without errors
- [ ] Both files renamed with Student ID and Last Name
- [ ] Signed the declaration above

---

**Submission Instructions:**

1. Save this file as: `hw3_report_<StudentID>_<LastName>.md`
2. Submit both `hw3_code_*.py` and `hw3_report_*.md` to Moodle
3. Due: April 24 at 23:59

**Good luck!** 🍀
