# In-Class Exercise: Classification with Decision Trees

**Course**: ECON6083 - Machine Learning in Economics
**Topic**: Decision Trees and Overfitting
**Time**: 25 minutes

---

## Part I: Conceptual Check

### Question 1: Tree Splitting Criteria

A decision tree is choosing between two splitting rules for a binary classification problem:

**Split A**: Results in two nodes with class distributions:
- Left node: 90% Class 0, 10% Class 1
- Right node: 10% Class 0, 90% Class 1

**Split B**: Results in two nodes with class distributions:
- Left node: 60% Class 0, 40% Class 1
- Right node: 40% Class 0, 60% Class 1

Which split will be preferred by the Gini impurity criterion?

(A) Split A, because it creates purer nodes
(B) Split B, because it's more balanced
(C) Both are equally good
(D) Cannot determine without knowing the sample sizes

**Answer**: ___________

**Explanation**: The Gini impurity measures node purity. For a node with proportion $p$ of class 1:
$$\text{Gini} = p(1-p)$$

---

### Question 2: Overfitting in Trees

You train a decision tree on 1,000 samples. As you increase `max_depth` from 1 to 20, you observe:

- Training accuracy: 65% → 75% → 88% → 99% → 100%
- Test accuracy: 64% → 74% → 82% → 78% → 65%

At what approximate depth does overfitting begin?

(A) Depth = 1
(B) Depth = 5
(C) Depth = 10
(D) Overfitting never occurs in this example

**Answer**: ___________

**Economic Intuition**: This is analogous to adding too many control variables in a regression. The model starts "memorizing" noise instead of learning true patterns.

---

### Question 3: Ensemble Intuition

Suppose you train 100 decision trees on different bootstrap samples of your data (this is called a Random Forest). Each individual tree has 70% accuracy. Will the ensemble of 100 trees have:

(A) Approximately 70% accuracy (same as individual trees)
(B) Lower than 70% accuracy (worse than individual trees)
(C) Higher than 70% accuracy (better than individual trees)
(D) Cannot determine without more information

**Answer**: ___________

**Key Concept**: This relates to the "wisdom of crowds" - averaging predictions from diverse models reduces variance.

---

## Part II: Analytical Problem

### The Bias-Variance Decomposition in Trees

Consider a simple 1D classification problem: predict $Y \in \{0, 1\}$ from $X \in \mathbb{R}$.

**True Model**:
$$P(Y=1 | X) = \begin{cases} 0.9 & \text{if } X > 0 \\ 0.1 & \text{if } X \leq 0 \end{cases}$$

You have a training set of 10 samples:
- 5 samples with $X < 0$, labels: [0, 0, 0, 0, 1]
- 5 samples with $X > 0$, labels: [1, 1, 1, 1, 0]

**Task 1**: What is the optimal decision tree (with max_depth=1) for this training data?

Split point: $X = \_\_\_\_\_\_\_$
Left node prediction: $\hat{Y} = \_\_\_\_\_\_\_$
Right node prediction: $\hat{Y} = \_\_\_\_\_\_\_$

**Task 2**: Calculate the training error rate.

Training error rate = \_\_\_\_\_\_\_ / 10 = \_\_\_\_\_\_%

**Task 3**: What is the true error rate (using the true model $P(Y|X)$)?

Expected error rate = \_\_\_\_\_\_\_%

*(Hint: The tree predicts 0 for $X \leq 0$ and 1 for $X > 0$. Error occurs when we're wrong.)*

---

## Part III: Practical Scenario

### Case Study: Credit Default Prediction

You are a data scientist at a bank. Your task is to predict which loan applicants will default (not repay their loan).

**Dataset**: 5,000 past loan applications with features:
- `income` (annual income in thousands)
- `credit_score` (300-850)
- `loan_amount` (requested loan in thousands)
- `employment_years` (years at current job)
- `previous_defaults` (0 or 1)

**Target**: `default` (1 = defaulted, 0 = repaid)

**Class Distribution**:
- 4,500 repaid (90%)
- 500 defaulted (10%)

### Discussion Questions

**1. Accuracy Trap**

You train a simple model that always predicts "no default" (class 0). What is its accuracy?

Accuracy = \_\_\_\_\_\_\_%

Why is this misleading? What metric would be more appropriate?

**Suggested Metric**: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

**2. Interpretability vs Performance**

You compare two models:

**Model A: Single Decision Tree** (max_depth=3)
- Test Accuracy: 87%
- Easy to visualize and explain to loan officers
- Rules like: "IF credit_score < 600 AND income < 40K THEN predict default"

**Model B: Random Forest** (100 trees, max_depth=10)
- Test Accuracy: 91%
- Black box - cannot easily explain individual predictions
- Better performance

Which model would you deploy? Consider:
- Regulatory requirements (banks must explain loan decisions)
- Cost of false positives (deny good customers) vs false negatives (approve bad customers)
- Fairness concerns (avoiding discrimination)

**Your recommendation**:

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

**3. Cost-Sensitive Classification**

Not all errors are equal in this problem:

- **False Negative** (predict no default, but they default): Bank loses \$10,000 on average
- **False Positive** (predict default, but they repay): Bank loses potential interest income of \$500

How would you modify your decision tree to account for this asymmetric cost?

(A) Lower the classification threshold (predict default more often)
(B) Increase the classification threshold (predict default less often)
(C) Use class weights in training
(D) Both A and C

**Answer**: \_\_\_\_\_\_\_\_\_

---

## Part IV: Quick Coding Exercise (Optional)

If you have Python available, try this 5-minute coding challenge:

```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate synthetic data
X, y = make_classification(n_samples=500, n_features=5,
                          n_informative=3, n_redundant=2,
                          random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TODO: Train trees with different depths and compare
depths = [1, 3, 5, 10, 20]
train_accs = []
test_accs = []

for depth in depths:
    # YOUR CODE HERE:
    # 1. Create DecisionTreeClassifier with max_depth=depth
    # 2. Fit on training data
    # 3. Calculate training accuracy
    # 4. Calculate test accuracy
    # 5. Append to lists
    pass

# Print results
print(f"{'Depth':<10} {'Train Acc':<12} {'Test Acc':<12}")
print("-" * 35)
for i, depth in enumerate(depths):
    print(f"{depth:<10} {train_accs[i]:<12.3f} {test_accs[i]:<12.3f}")
```

**Expected Output Pattern**:
- Training accuracy should increase monotonically
- Test accuracy should increase then decrease (overfitting)

---

## Summary: Key Takeaways

After this exercise, you should understand:

1. ✅ **Tree Splitting**: How Gini/Entropy measure node purity
2. ✅ **Overfitting**: Deeper trees memorize training data
3. ✅ **Ensembles**: Combining models reduces variance
4. ✅ **Metrics Matter**: Accuracy can be misleading with imbalanced classes
5. ✅ **Trade-offs**: Interpretability vs performance is a real concern
6. ✅ **Cost-Sensitivity**: Different errors have different costs

---

## Preview: Assignment 2

In the upcoming homework, you will:
- Work with real census income data (48,000 samples)
- Build and compare Decision Tree, Random Forest, and Gradient Boosting
- Handle imbalanced classes properly
- Analyze feature importance from an economic perspective
- Discuss fairness and bias in algorithmic decision-making

The skills you practiced today will be essential!

---

**For Discussion**:
- When might you prefer a simple tree over a random forest?
- How do you balance model performance with interpretability in high-stakes decisions?
- What are the ethical implications of using ML for credit scoring?
