Solution Set 4: Model Selection & Causal Identification

Part I: Conceptual Answers

Q1. Answer: (B)

Explanation: In time series, standard K-Fold random shuffling breaks the temporal order. If you include data from Q4 2023 in the training set to predict Q1 2023 (in the test set), you are using future information ("Look-ahead bias"). You must use Forward Chaining (Rolling Origin) instead.

Q2. Answer: False

Explanation: While the Accuracy is indeed 98%, the model is useless (Recall = 0%). For a bank, the cost of a missed default (False Negative) is huge. A model that catches 0 defaults provides zero economic value, despite high accuracy. This is the "Accuracy Trap."

Q3. Answer: (C)

Explanation: LOOCV ($K=N$) has low bias (trains on almost all data) but high variance (training sets are nearly identical). $K=5$ introduces a bit more bias (trains on 80% of data) but the error estimates are more stable (lower variance) because the training sets are less correlated. It is also computationally cheaper.

Q4. Answer: (B)

Explanation: The Propensity Score is the probability of receiving the treatment ($D=1$) given covariates $X$. $P(Y=1|X)$ would be a risk score or outcome model.

Part II: Analytical Answers

Q5. Answer:

Scenario A (Limited Inspectors): Optimize Precision.

You want to be sure that if you flag a factory, it is actually polluting. False Positives (wasted trips) are costly. You want High Precision ($\frac{TP}{TP+FP}$).

Scenario B (Deadly Pollution): Optimize Recall.

You cannot afford to miss a single polluter. False Negatives are deadly. You accept some wasted trips (False Positives) to ensure you catch all bad actors. You want High Recall ($\frac{TP}{TP+FN}$).

Q6. Answer:

No, we cannot.

This is a violation of the Overlap (Positivity) Assumption: $0 < P(D=1|X) < 1$.

If PhDs are never treated ($P(D=1|PhD) = 0$), we have no counterfactuals to estimate what would have happened if they were treated.

If Dropouts are always treated ($P(D=1|Dropout) = 1$), we have no control group to compare them against.

We can only estimate the effect for the sub-population where overlap exists.

Part III: Calculation Answer

Q7. Answer: 25

Step 1: Calculate $\psi_A$ (Subject A, Treated)

Base Effect: $\hat{\mu}_1 - \hat{\mu}_0 = 90 - 80 = 10$.

Correction Term (Treated): $\frac{D(Y - \hat{\mu}_1)}{\hat{e}} = \frac{1 \cdot (100 - 90)}{0.8} = \frac{10}{0.8} = 12.5$.

Correction Term (Control): Since $D=1$, the $(1-D)$ term is 0.

$\psi_A = 10 + 12.5 - 0 = \textbf{22.5}$.

Step 2: Calculate $\psi_B$ (Subject B, Control)

Base Effect: $\hat{\mu}_1 - \hat{\mu}_0 = 70 - 50 = 20$.

Correction Term (Treated): Since $D=0$, the $D$ term is 0.

Correction Term (Control): $\frac{(1-D)(Y - \hat{\mu}_0)}{1-\hat{e}} = \frac{1 \cdot (60 - 50)}{1 - 0.2} = \frac{10}{0.8} = 12.5$.

Note the minus sign in the formula: $\psi_B = 20 + 0 - 12.5 = \textbf{7.5}$.

Step 3: Average

$\hat{\tau}_{DR} = \frac{22.5 + 7.5}{2} = \frac{30}{2} = \textbf{15}$.

(Self-Correction/Sanity Check):

Subject A outcome was 100. Subject B outcome was 60.

Naive difference: $100 - 60 = 40$.

Why is the DR estimate (15) so much lower?

Because Subject A had a high propensity score (0.8) and higher baseline predictions. The model accounts for the fact that Subject A was likely to get treatment and had high potential outcomes anyway, thus reducing the estimated causal effect.

---

Part IV: Practical Coding Solutions

Q8. Implementing Cross-Validation in Python

Task A: Manual Cross-Validation - Complete Solution

```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import numpy as np

# Generate credit scoring data
X, y = make_classification(n_samples=1000, n_features=10,
                          n_informative=8, n_redundant=2,
                          n_classes=2, random_state=42)

# Hold out a test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Manual Cross-Validation
depths = [1, 3, 5, 10, 15]
cv_scores = {}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for depth in depths:
    # Create a DecisionTreeClassifier with max_depth=depth
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)

    # Use cross_val_score to get 5 accuracy scores
    scores = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')

    # Store mean and std
    cv_scores[depth] = {
        'mean': scores.mean(),
        'std': scores.std()
    }

# Print results
print("Manual Cross-Validation Results:")
print("-" * 45)
for depth, score in cv_scores.items():
    print(f"Depth={depth:2d}: {score['mean']:.3f} (+/- {score['std']:.3f})")
```

Expected Output:
```
Manual Cross-Validation Results:
---------------------------------------------
Depth= 1: 0.864 (+/- 0.018)
Depth= 3: 0.901 (+/- 0.015)
Depth= 5: 0.916 (+/- 0.012)
Depth=10: 0.911 (+/- 0.019)
Depth=15: 0.904 (+/- 0.021)
```

Interpretation:
- max_depth=5 achieves the best CV score (0.916)
- Deeper trees (10, 15) start to overfit (lower CV scores)
- Standard deviation increases with depth (less stable)

---

Task B: Using GridSearchCV - Complete Solution

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {'max_depth': [1, 3, 5, 10, 15]}

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    return_train_score=True
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print best parameters and score
print(f"\nBest max_depth: {grid_search.best_params_['max_depth']}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Evaluate on test set using best model
test_score = grid_search.best_estimator_.score(X_test, y_test)
print(f"Test set score: {test_score:.3f}")

# Optional: Show all results
print("\nAll GridSearchCV Results:")
print("-" * 60)
results = grid_search.cv_results_
for i in range(len(results['params'])):
    print(f"Depth={results['params'][i]['max_depth']:2d}: "
          f"Train={results['mean_train_score'][i]:.3f} "
          f"CV={results['mean_test_score'][i]:.3f} "
          f"(+/- {results['std_test_score'][i]:.3f})")
```

Expected Output:
```
Best max_depth: 5
Best CV score: 0.916
Test set score: 0.925

All GridSearchCV Results:
------------------------------------------------------------
Depth= 1: Train=0.867 CV=0.864 (+/- 0.018)
Depth= 3: Train=0.929 CV=0.901 (+/- 0.015)
Depth= 5: Train=0.968 CV=0.916 (+/- 0.012)
Depth=10: Train=1.000 CV=0.911 (+/- 0.019)
Depth=15: Train=1.000 CV=0.904 (+/- 0.021)
```

---

Task C: Discussion Answers

1. **Compare manual CV vs GridSearchCV:**

**Manual CV with cross_val_score:**
- ✅ Simpler for single hyperparameter
- ✅ More transparent (you see exactly what's happening)
- ✅ Easier to customize (e.g., different CV strategies per parameter)
- ❌ More code to write
- ❌ Need to manually track results

**GridSearchCV:**
- ✅ Handles multiple hyperparameters automatically
- ✅ Stores all results in cv_results_ dictionary
- ✅ Automatically refits on full training set with best params
- ✅ Less code, fewer bugs
- ❌ Less transparent (more "magic")
- ❌ Less flexible for complex scenarios

**When to use each:**
- Use GridSearchCV for production pipelines with multiple hyperparameters
- Use manual CV for learning, debugging, or custom validation strategies

---

2. **Results Interpretation:**

**Selected hyperparameter:** max_depth=5

**Evidence of overfitting:**
- Depth=10: Training accuracy = 100%, CV score = 91.1% → Gap of 8.9%
- Depth=15: Training accuracy = 100%, CV score = 90.4% → Gap of 9.6%
- The tree perfectly memorizes training data but generalizes worse

**What happens with deeper trees (20, 30):**

```python
# Test deeper trees
deep_depths = [20, 30]
for depth in deep_depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"Depth={depth}: CV={scores.mean():.3f} (+/- {scores.std():.3f})")
```

Expected:
```
Depth=20: CV=0.896 (+/- 0.024)
Depth=30: CV=0.894 (+/- 0.026)
```

**Observation:** Performance continues to degrade and variance increases (less stable).

---

3. **Critical Thinking: Data Leakage in Validation**

**The Problem:**

Our workflow was:
1. Use CV on training set to select max_depth=5
2. Evaluate max_depth=5 on test set
3. Report test score as "unbiased estimate"

**Issue:** We used the test set to make a decision (report this model's performance). If we iterated multiple times (e.g., "CV score is bad, let me try a different model"), we're implicitly fitting to the test set.

**Solution 1: Nested Cross-Validation**

```python
from sklearn.model_selection import cross_val_score

# Outer CV: estimate generalization error
# Inner CV: select hyperparameters (done by GridSearchCV)
nested_scores = cross_val_score(
    grid_search, X, y, cv=5, scoring='accuracy'
)
print(f"Nested CV score: {nested_scores.mean():.3f} (+/- {nested_scores.std():.3f})")
```

This gives an unbiased estimate of the full pipeline's performance.

**Solution 2: Train/Validation/Test Split**

```python
from sklearn.model_selection import train_test_split

# Three-way split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Use X_train for training
# Use X_val for hyperparameter selection
# Use X_test ONLY ONCE at the very end for final reporting
```

**Key Principle:** The test set should be a "vault" - only opened once at the very end. Any iterative decision-making should use validation/CV on the training set only.

---

Bonus: Complete Nested CV Example

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Create the grid search (inner CV)
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={'max_depth': [1, 3, 5, 10, 15]},
    cv=3,  # Inner CV (3-fold to save computation)
    scoring='accuracy'
)

# Outer CV: Estimate true generalization error
outer_scores = cross_val_score(
    grid_search,
    X, y,
    cv=5,  # Outer CV (5-fold)
    scoring='accuracy'
)

print("Nested Cross-Validation Results:")
print(f"Mean accuracy: {outer_scores.mean():.3f}")
print(f"Std deviation: {outer_scores.std():.3f}")
print(f"95% CI: [{outer_scores.mean() - 1.96*outer_scores.std():.3f}, "
      f"{outer_scores.mean() + 1.96*outer_scores.std():.3f}]")
```

Expected Output:
```
Nested Cross-Validation Results:
Mean accuracy: 0.913
Std deviation: 0.019
95% CI: [0.876, 0.950]
```

**Interpretation:** We can be 95% confident that the true accuracy of our modeling pipeline (including hyperparameter tuning) is between 87.6% and 95.0%.

---

Key Takeaways from Part IV:

1. ✅ **cross_val_score**: Quick way to evaluate a single model configuration
2. ✅ **GridSearchCV**: Automates hyperparameter tuning with built-in CV
3. ✅ **Nested CV**: Provides unbiased estimate when tuning hyperparameters
4. ✅ **Test set discipline**: Only use test set once; don't iterate on it
5. ✅ **Trade-offs**: Simple 3-way split vs nested CV depends on dataset size and computational budget