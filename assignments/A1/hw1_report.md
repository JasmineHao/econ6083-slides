# ECON 491: Homework 1 Report
## Hedonic Pricing with OLS and Penalized Regression

**Student Name:** [Your Name Here]
**Student ID:** [Your ID Here]
**Date:** [Submission Date]

---

## Part I: Theoretical Concepts

**Instructions:** Fill in the blanks using the terms from the word bank.

**Word Bank:** *Variance, Bias, Zero, Infinity, Overfitting, Underfitting, Regularization, Sparsity, L1, L2, Increased, Decreased, Unbiased.*

### Question 1
The mean squared error (MSE) of a model can be decomposed into three parts: Irreducible Error, **[__________]** squared, and **[__________]**.

**Your Answer:**
The mean squared error (MSE) of a model can be decomposed into three parts: Irreducible Error, **[FILL IN]** squared, and **[FILL IN]**.

---

### Question 2
In a high-dimensional setting where the number of features is large relative to the sample size, standard OLS estimates tend to have low bias but extremely high **[__________]**, leading to poor out-of-sample performance. This phenomenon is often referred to as **[__________]**.

**Your Answer:**
In a high-dimensional setting where the number of features is large relative to the sample size, standard OLS estimates tend to have low bias but extremely high **[FILL IN]**, leading to poor out-of-sample performance. This phenomenon is often referred to as **[FILL IN]**.

---

### Question 3
Ridge Regression introduces a penalty term to the objective function. This intentionally adds a small amount of **[__________]** to the model in exchange for a significant reduction in **[__________]**, thereby improving overall prediction accuracy.

**Your Answer:**
Ridge Regression introduces a penalty term to the objective function. This intentionally adds a small amount of **[FILL IN]** to the model in exchange for a significant reduction in **[FILL IN]**, thereby improving overall prediction accuracy.

---

### Question 4
Lasso Regression uses **[__________]** regularization. A unique economic property of Lasso is its ability to force certain coefficients to become exactly **[__________]**, effectively performing automatic variable selection. This property is known as **[__________]**.

**Your Answer:**
Lasso Regression uses **[FILL IN]** regularization. A unique economic property of Lasso is its ability to force certain coefficients to become exactly **[FILL IN]**, effectively performing automatic variable selection. This property is known as **[FILL IN]**.

---

## Part II: Model Performance Comparison

### 2.1 Data Split
- **Training Set Size:** [FILL IN] observations
- **Test Set Size:** [FILL IN] observations
- **Random Seed:** 42

### 2.2 RMSE Results

| Model | Test RMSE (Log scale) | Test RMSE (Original $) |
|-------|----------------------|------------------------|
| OLS   | [FILL IN]            | [FILL IN]              |
| Ridge (Best α) | [FILL IN]   | [FILL IN]              |
| Lasso (Best α) | [FILL IN]   | [FILL IN]              |

**Best Hyperparameters:**
- Ridge Best α: [FILL IN]
- Lasso Best α: [FILL IN]

---

## Part III: Economic Analysis

### 3.1 The OLS Failure

**Question:** Did OLS perform worse than the regularized models? Explain why, referring to the "Bias-Variance Trade-off" and the dimensionality of the dataset.

**Your Answer:**

[Write your analysis here. Consider:
- How many features vs. observations?
- What happens to OLS variance in high dimensions?
- How does regularization help?
- Reference the bias-variance tradeoff]

---

### 3.2 Sparsity Analysis

**Question:** For your best Lasso model, how many coefficients were reduced to exactly zero? What does this imply about the redundancy of variables in the original dataset?

**Your Answer:**

- **Number of non-zero coefficients:** [FILL IN]
- **Number of zero coefficients:** [FILL IN]
- **Percentage of features eliminated:** [FILL IN]%

**Economic Interpretation:**

[Write your analysis here. Consider:
- What does sparsity tell us about feature redundancy?
- Which types of features were eliminated?
- Does this make economic sense?]

---

### 3.3 Hedonic Interpretation

**Question:** List the top 5 features with the largest absolute coefficients from your Lasso model. Do these make economic sense?

**Top 5 Most Important Features:**

| Rank | Feature Name | Coefficient | Economic Interpretation |
|------|-------------|-------------|-------------------------|
| 1    | [FILL IN]   | [FILL IN]   | [Explain why this matters] |
| 2    | [FILL IN]   | [FILL IN]   | [Explain why this matters] |
| 3    | [FILL IN]   | [FILL IN]   | [Explain why this matters] |
| 4    | [FILL IN]   | [FILL IN]   | [Explain why this matters] |
| 5    | [FILL IN]   | [FILL IN]   | [Explain why this matters] |

**Overall Hedonic Interpretation:**

[Write your analysis here. Consider:
- Do the most important features align with economic theory?
- Are quality metrics (OverallQual, etc.) important?
- Do location features matter?
- Any surprising findings?]

---

## Part IV: Reflection (Optional Bonus)

### What I Learned

[Optional: Share 2-3 key insights you gained from this assignment about:
- The practical differences between OLS and regularized regression
- How to handle real-world data issues (missing values, feature engineering)
- The importance of the bias-variance tradeoff in economic prediction problems]

---

## Code Appendix

**Note:** Your `hw1_code.py` file should be submitted separately. Do not paste code into this report.

### Key Design Decisions

[Optional: Briefly describe any important implementation choices you made:
- Which features did you engineer and why?
- How did you handle missing values for ambiguous cases?
- Any other notable decisions?]

---

**End of Report**
