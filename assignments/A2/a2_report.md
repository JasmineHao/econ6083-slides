# Homework 2 Report: Income Prediction with Tree-Based Models

**Student Name:** ___________________________
**Student ID:** ___________________________
**Date:** ___________________________

---

## Part I: Theoretical Understanding (15 points)

### Question 1: Decision Tree Splitting (3 points)

A decision tree is deciding between two splitting rules for a classification problem:

**Split A**: Gini impurity decreases from 0.5 to 0.3
**Split B**: Gini impurity decreases from 0.5 to 0.2

Which split will the algorithm choose, and why?

**Your answer:**

_[Write your answer here]_

---

### Question 2: Ensemble Methods - Bagging vs Boosting (4 points)

Explain the key difference between Random Forests (bagging) and Gradient Boosting (boosting) in terms of:
1. How they combine multiple trees
2. Whether trees are trained sequentially or in parallel
3. Which method is more prone to overfitting

**Your answer:**

_[Write your answer here]_

---

### Question 3: Bias-Variance Tradeoff in Trees (4 points)

As you increase `max_depth` in a decision tree:
- What happens to bias? (increases / decreases / stays the same)
- What happens to variance? (increases / decreases / stays the same)
- What happens to training error? (increases / decreases / stays the same)
- What is the likely effect on test error? (increases / decreases / first decreases then increases)

**Your answers:**

- **Bias:** ___________
- **Variance:** ___________
- **Training error:** ___________
- **Test error:** ___________

**Explanation:**

_[Explain why these changes occur]_

---

### Question 4: Handling Imbalanced Data (4 points)

The UCI Adult Income dataset is imbalanced (75% earn ≤50K, 25% earn >50K).

Suppose Model A achieves 78% accuracy, while Model B achieves 76% accuracy.

Can you conclude that Model A is better? Why or why not? What additional metrics should you examine?

**Your answer:**

_[Write your answer here]_

---

## Part II: Implementation Results (55 points)

*Note: This section is graded automatically based on your code. Include the output here for reference.*

### Model Performance Summary

Copy the comparison table from your code output:

```
Model                Accuracy     Precision    Recall       F1           AUC
--------------------------------------------------------------------------------
decision_tree        [value]      [value]      [value]      [value]      [value]
random_forest        [value]      [value]      [value]      [value]      [value]
gradient_boosting    [value]      [value]      [value]      [value]      [value]
```

### Best Hyperparameters

**Decision Tree:**
- max_depth: _______
- min_samples_split: _______
- min_samples_leaf: _______

**Random Forest:**
- n_estimators: _______
- max_depth: _______
- min_samples_split: _______

**Gradient Boosting:**
- n_estimators: _______
- learning_rate: _______
- max_depth: _______

---

## Part III: Economic Analysis (30 points)

### 1. Interpretability vs Performance Trade-off (10 points)

Compare the single decision tree (interpretable) with ensemble methods (more accurate but less interpretable).

**Discussion prompts:**
- Which model would you deploy for a bank making loan decisions? Why?
- What are the regulatory implications? (Consider Fair Lending laws, GDPR's "right to explanation")
- Can you achieve both interpretability AND high performance? How?
- What is the economic cost of choosing a simpler but less accurate model?

**Your analysis (250-350 words):**

_[Write your analysis here]_

---

### 2. Feature Importance Analysis (10 points)

Analyze the top 5 most important features from your Random Forest or Gradient Boosting model.

**Top 5 Features Table:**

| Rank | Feature Name | Importance Score | Economic Interpretation |
|------|-------------|------------------|------------------------|
| 1    | [name]      | [score]          | [why is it predictive?] |
| 2    | [name]      | [score]          | [why is it predictive?] |
| 3    | [name]      | [score]          | [why is it predictive?] |
| 4    | [name]      | [score]          | [why is it predictive?] |
| 5    | [name]      | [score]          | [why is it predictive?] |

**Discussion prompts:**
- Do these results align with labor economics theory?
- Are there any surprising features? Why might they be important?
- Do feature importances differ between Random Forest and Gradient Boosting?

**Your analysis (200-300 words):**

_[Write your analysis here]_

---

### 3. Fairness and Bias Analysis (10 points)

Examine potential discrimination in your model's predictions across demographic groups.

**Fairness Metrics by Group:**

Copy your fairness analysis output, or create a table like:

| Group  | FPR (False Positive Rate) | FNR (False Negative Rate) | F1-Score |
|--------|---------------------------|---------------------------|----------|
| Male   | [value]                   | [value]                   | [value]  |
| Female | [value]                   | [value]                   | [value]  |

**Discussion prompts:**
- Are error rates balanced across groups? If not, which group is disadvantaged?
- What are the real-world implications of these disparities?
- What might cause these differences? (Historical discrimination, feature selection, etc.)
- How would you mitigate bias? (Re-weighting, fairness constraints, separate models, etc.)

**Your analysis (250-350 words):**

_[Write your analysis here]_

---

## Part IV: Reflection Questions (Optional - No points, but helpful!)

### What was the most challenging part of this assignment?

_[Your answer]_

---

### What did you learn about the trade-offs between different tree-based models?

_[Your answer]_

---

### How might you apply these methods to an economics research question you're interested in?

_[Your answer]_

---

## Declaration

I hereby declare that this assignment is my own work and that I have not copied from any other student's work or from any other source except where due acknowledgment is made explicitly in the text.

**Signature:** ___________________________

**Date:** ___________________________

---

## Submission Checklist

Before submitting, make sure you have:

- [ ] Completed all theoretical questions in Part I
- [ ] Included model performance table in Part II
- [ ] Written 250+ word analysis for interpretability (Part III.1)
- [ ] Written 200+ word analysis for feature importance (Part III.2)
- [ ] Written 250+ word analysis for fairness (Part III.3)
- [ ] Included fairness metrics table
- [ ] Code file (`a2_code_<ID>_<Name>.py`) runs without errors
- [ ] Both files renamed with Student ID and Last Name
- [ ] Signed the declaration above

---

**Submission Instructions:**

1. Save this file as: `a2_report_<StudentID>_<LastName>.md`
2. Submit both `a2_code_*.py` and `a2_report_*.md` to Moodle
3. Due: Week 6 at 23:59

**Good luck!** 🍀
