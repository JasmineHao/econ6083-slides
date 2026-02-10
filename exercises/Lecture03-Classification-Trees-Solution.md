# Solution: Classification with Decision Trees

**Course**: ECON6083 - Machine Learning in Economics
**Topic**: Decision Trees and Overfitting

---

## Part I: Conceptual Check - Solutions

### Question 1: Tree Splitting Criteria

**Answer**: **(A) Split A, because it creates purer nodes**

**Explanation**:

The Gini impurity for each node:

**Split A**:
- Left node: Gini = 0.9 × 0.1 = 0.09
- Right node: Gini = 0.1 × 0.9 = 0.09
- Average: 0.09 (very pure nodes)

**Split B**:
- Left node: Gini = 0.6 × 0.4 = 0.24
- Right node: Gini = 0.4 × 0.6 = 0.24
- Average: 0.24 (less pure nodes)

Split A is preferred because it creates purer nodes (lower Gini impurity). The goal is to separate classes as cleanly as possible.

---

### Question 2: Overfitting in Trees

**Answer**: **(C) Depth = 10**

**Explanation**:

Overfitting begins when test accuracy starts decreasing while training accuracy continues to increase. Looking at the pattern:

- Depth 1-5: Both training and test accuracy increase (learning real patterns)
- Depth ~10: Test accuracy peaks at 82%
- Depth >10: Test accuracy drops to 78% → 65%, while training accuracy reaches 100%

Around depth 10, the tree starts memorizing noise rather than learning generalizable patterns. The optimal depth would be around 5-10.

**Economic Intuition**: This is like running a regression with too many controls. Adding more splits initially captures important patterns (like key demographic factors), but eventually you're just fitting noise (like obscure interaction terms that don't generalize).

---

### Question 3: Ensemble Intuition

**Answer**: **(C) Higher than 70% accuracy (better than individual trees)**

**Explanation**:

Random forests leverage the "wisdom of crowds" effect. If individual trees make independent errors, averaging their predictions reduces variance.

**Why it works**:
- Each tree is trained on different bootstrap samples → sees different data
- Each tree makes different mistakes → errors partially cancel out
- Majority voting or averaging smooths out individual errors

**Key requirement**: Trees must be somewhat diverse (uncorrelated errors). This is why random forests also use feature subsampling.

**Caveat**: If all trees make the *same* systematic mistakes (high bias), ensemble won't help much.

---

## Part II: Analytical Problem - Solutions

### The Bias-Variance Decomposition in Trees

**Task 1**: Optimal decision tree with max_depth=1

- **Split point**: X = 0
- **Left node prediction** (X ≤ 0): Ŷ = 0 (since 4 out of 5 samples are class 0)
- **Right node prediction** (X > 0): Ŷ = 1 (since 4 out of 5 samples are class 1)

**Explanation**: The tree splits at X = 0 (the natural boundary) and predicts the majority class in each region.

---

**Task 2**: Training error rate

On the training data:
- Left node (X ≤ 0): Predicts 0, but one sample has Y=1 → 1 error
- Right node (X > 0): Predicts 1, but one sample has Y=0 → 1 error

**Training error rate** = 2/10 = **20%**

---

**Task 3**: True error rate

Using the true model P(Y=1|X):

For X ≤ 0:
- Tree predicts Ŷ = 0
- True probability: P(Y=1|X) = 0.1
- Error rate: 0.1 (we make mistakes 10% of the time)

For X > 0:
- Tree predicts Ŷ = 1
- True probability: P(Y=1|X) = 0.9
- Error rate: 0.1 (we make mistakes 10% of the time when Y=0)

**Expected error rate** = 0.5 × 0.1 + 0.5 × 0.1 = **10%**

(Assuming equal probability of X > 0 and X ≤ 0)

**Key Insight**: Training error (20%) > True error (10%). This is because the training sample happened to have unusual noise (one mislabeled sample in each region). With infinite data, training error would converge to true error.

---

## Part III: Practical Scenario - Solutions

### Case Study: Credit Default Prediction

**1. Accuracy Trap**

**Accuracy** = 4,500/5,000 = **90%**

**Why this is misleading**:

A model that predicts "no default" for everyone achieves 90% accuracy without learning anything useful! It completely misses all 500 actual defaults (Recall = 0%).

**More appropriate metrics**:
- **Precision**: Of those predicted to default, how many actually did?
- **Recall**: Of all actual defaults, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (measures discriminative ability)
- **Precision-Recall curve**: Better for imbalanced data

For banking applications, **Recall** is often critical (can't afford to miss defaults).

---

**2. Interpretability vs Performance**

**Recommended approach**: **Model A (Decision Tree) for deployment, with Model B for validation**

**Reasoning**:

**Regulatory compliance**:
- US Fair Lending laws require explainability of credit decisions
- GDPR "right to explanation" in EU
- Simple trees can be audited for discriminatory patterns

**Business considerations**:
- Loan officers need to explain rejections to customers
- "Your credit score is below 600 and income is below $40K" is actionable
- Random forest predictions are hard to justify

**Hybrid approach**:
- Deploy Model A for operational decisions
- Use Model B to identify where Model A makes systematic errors
- If performance gap is very large (e.g., >10%), consider:
  - Using Model B for initial screening
  - Using Model A for final explainable decisions
  - Investing in interpretable ML (e.g., GAMs, rule lists)

**Cost-benefit analysis**:
- 4% accuracy gain = ~200 fewer errors on 5,000 loans
- If each error costs $5,000 → $1M potential savings
- But: Risk of regulatory fines, lawsuits, reputation damage
- Explainability may be worth the 4% performance trade-off

---

**3. Cost-Sensitive Classification**

**Answer**: **(D) Both A and C**

**Explanation**:

**Option A: Lower the classification threshold**
- Default sklearn threshold: 0.5 (predict default if P(default) > 0.5)
- Lower to 0.3: More aggressive in flagging defaults
- Increases Recall (catch more defaults) at cost of Precision (more false alarms)

**Option C: Use class weights**
- In sklearn: `class_weight={0: 1, 1: 20}` (weight the minority class more)
- During training, penalize missing a default 20× more than a false alarm
- Naturally shifts the decision boundary

**Implementation**:

```python
from sklearn.tree import DecisionTreeClassifier

# Approach 1: Class weights
tree = DecisionTreeClassifier(
    class_weight={0: 1, 1: 20},  # Cost ratio = 10000/500 = 20
    random_state=42
)
tree.fit(X_train, y_train)

# Approach 2: Custom threshold
probs = tree.predict_proba(X_test)[:, 1]
y_pred = (probs > 0.3).astype(int)  # Lower threshold
```

**Economic justification**:
- False Negative cost: $10,000 (loan default)
- False Positive cost: $500 (lost interest)
- Cost ratio: 20:1
- Model should be 20× more willing to reject a loan than to approve a risky one

---

## Part IV: Quick Coding Exercise - Solution

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

# Train trees with different depths and compare
depths = [1, 3, 5, 10, 20]
train_accs = []
test_accs = []

for depth in depths:
    # Create and train the model
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)

    # Calculate accuracies
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_accs.append(train_acc)
    test_accs.append(test_acc)

# Print results
print(f"{'Depth':<10} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<12}")
print("-" * 50)
for i, depth in enumerate(depths):
    gap = train_accs[i] - test_accs[i]
    print(f"{depth:<10} {train_accs[i]:<12.3f} {test_accs[i]:<12.3f} {gap:<12.3f}")

# Optional: Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(depths, train_accs, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(depths, test_accs, 's-', label='Test Accuracy', linewidth=2)
plt.xlabel('Max Depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Overfitting in Decision Trees', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()
```

**Expected Output**:

```
Depth      Train Acc    Test Acc     Gap
--------------------------------------------------
1          0.868        0.860        0.008
3          0.958        0.900        0.058
5          0.990        0.890        0.100
10         1.000        0.870        0.130
20         1.000        0.850        0.150
```

**Observations**:
1. Training accuracy increases monotonically (deeper trees fit training data better)
2. Test accuracy peaks around depth=3-5
3. The gap between training and test accuracy grows with depth (overfitting)
4. At depth=20, the tree perfectly memorizes the training data (100% accuracy) but generalizes poorly (85% test accuracy)

---

## Summary: Key Takeaways

1. ✅ **Tree Splitting**: Gini/Entropy minimize impurity to create pure nodes
2. ✅ **Overfitting**: Occurs when test performance degrades while training performance improves
3. ✅ **Ensembles**: Averaging diverse models reduces variance and improves generalization
4. ✅ **Metrics Matter**: Accuracy is misleading for imbalanced data; use Precision, Recall, F1, AUC
5. ✅ **Trade-offs**: Interpretability vs performance is a real concern in high-stakes decisions
6. ✅ **Cost-Sensitivity**: Use class weights and custom thresholds when errors have asymmetric costs

---

## Connection to Economics

**Empirical Economics Applications**:
- **Credit scoring**: Predict loan defaults (as in this exercise)
- **Labor economics**: Predict employment outcomes, job mobility
- **Public policy**: Target interventions (e.g., tax audits, health screenings)
- **Development economics**: Predict poverty status for targeted transfers

**Key Economic Insight**:
Machine learning excels at prediction, but economists must carefully consider:
- Cost-benefit analysis of different error types
- Fairness and distributional impacts
- Interpretability requirements for policy
- Causal vs predictive goals (trees predict, they don't identify causal effects)

---

## Preview: What's Next?

In **Assignment 2**, you will apply these concepts to real census income data:
- Handle messy real-world data (missing values, categorical features)
- Compare Decision Trees, Random Forests, and Gradient Boosting
- Analyze fairness and bias in algorithmic decision-making
- Interpret feature importance from an economic perspective

The concepts practiced today (overfitting, evaluation metrics, interpretability trade-offs) will be essential!
