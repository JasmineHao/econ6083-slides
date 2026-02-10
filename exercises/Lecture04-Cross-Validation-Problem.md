Problem Set 4: Model Selection & Causal Identification

Course: Machine Learning for Economists
Topic: Validation, Metrics, and Doubly Robust Estimation

Part I: Conceptual Questions (Multiple Choice & True/False)

Q1. Cross-Validation in Time Series
You are building a model to forecast China's quarterly GDP growth. Your research assistant suggests using standard 10-Fold Cross-Validation with random shuffling to evaluate the model's performance.

(A) This is a good idea; it maximizes sample utilization.

(B) This is a bad idea; it introduces look-ahead bias (future leakage).

(C) This is a good idea; it reduces the variance of the error estimate.

(D) It does not matter; time series data is the same as cross-sectional data.

Q2. The Accuracy Trap
In a credit default dataset, only 2% of borrowers default. A "Lazy Model" predicts "No Default" for everyone.

True or False: This model will have an Accuracy of 98%, making it an excellent model for the bank's risk management department.

Q3. Bias-Variance Trade-off in K-Fold
Compared to Leave-One-Out Cross-Validation (LOOCV, where $K=N$), setting $K=5$:

(A) Increases the computational cost.

(B) Results in higher variance of the error estimate.

(C) Introduces slightly more bias but reduces the variance of the estimate.

(D) Uses more data for training in each fold.

Q4. The Propensity Score
The Propensity Score $e(X)$ is defined as:

(A) $P(Y=1 | X)$

(B) $P(D=1 | X)$

(C) $E[Y | D=1, X] - E[Y | D=0, X]$

(D) The probability of the outcome occurring given the treatment.

Part II: Analytical Scenarios

Q5. Precision vs. Recall in Policy
Imagine you are designing a Machine Learning algorithm to flag factories for environmental inspections ($Y=1$ means the factory is illegally polluting).

Scenario A: You have very limited budget and only 10 inspectors. You cannot afford to waste a trip on a compliant factory.

Scenario B: The pollution is toxic and deadly. Missing a polluting factory could cause a health crisis.

Question: In which scenario should you optimize for Precision, and in which should you optimize for Recall? Explain why.

Q6. The Overlap Assumption
In the LaLonde job training dataset, suppose we find that individuals with a PhD degree ($X_{edu} > 20$) are always in the control group (never treated), and individuals with no high school degree are always in the treatment group.

Can we estimate the Average Treatment Effect (ATE) for the entire population using Propensity Score Matching? Why or why not?

Part III: Calculation (Doubly Robust Estimation)

Q7. Manual Calculation of AIPW
Consider a sample of 2 individuals. We want to estimate the treatment effect $\tau$.

Subject A: Treated ($D=1$), Outcome $Y=100$.

Propensity Score: $\hat{e}(X_A) = 0.8$

Outcome Model Predictions: $\hat{\mu}_1(X_A) = 90$, $\hat{\mu}_0(X_A) = 80$.

Subject B: Control ($D=0$), Outcome $Y=60$.

Propensity Score: $\hat{e}(X_B) = 0.2$

Outcome Model Predictions: $\hat{\mu}_1(X_B) = 70$, $\hat{\mu}_0(X_B) = 50$.

Calculate the Doubly Robust estimate ($\hat{\tau}_{DR}$) for this sample of $N=2$.

Hint: The formula for the individual score $\psi_i$ is:


$$\psi_i = (\hat{\mu}_1 - \hat{\mu}_0) + \frac{D(Y - \hat{\mu}_1)}{\hat{e}} - \frac{(1-D)(Y - \hat{\mu}_0)}{1-\hat{e}}$$

$$\hat{\tau}_{DR} = \frac{1}{N} \sum \psi_i$$

---

Part IV: Practical Coding Exercise (Optional, 10 minutes)

Q8. Implementing Cross-Validation in Python

You have been given the following credit scoring dataset and need to tune the max_depth hyperparameter for a Decision Tree.

```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np

# Generate credit scoring data
X, y = make_classification(n_samples=1000, n_features=10,
                          n_informative=8, n_redundant=2,
                          n_classes=2, random_state=42)

# Hold out a test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Task A: Manual Cross-Validation

Complete the code to manually implement 5-fold cross-validation for different max_depth values:

```python
from sklearn.model_selection import KFold

depths = [1, 3, 5, 10, 15]
cv_scores = {}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for depth in depths:
    # TODO: Create a DecisionTreeClassifier with max_depth=depth
    clf = _______________

    # TODO: Use cross_val_score to get 5 accuracy scores
    scores = _______________

    # Store mean and std
    cv_scores[depth] = {
        'mean': scores.mean(),
        'std': scores.std()
    }

# Print results
for depth, score in cv_scores.items():
    print(f"Depth={depth}: {score['mean']:.3f} (+/- {score['std']:.3f})")
```

Task B: Using GridSearchCV

Now do the same using GridSearchCV (the automatic way):

```python
# TODO: Define parameter grid
param_grid = {'max_depth': [1, 3, 5, 10, 15]}

# TODO: Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=_______________,
    param_grid=_______________,
    cv=5,
    scoring='accuracy',
    return_train_score=True
)

# TODO: Fit the grid search
grid_search.fit(_______________, _______________)

# Print best parameters and score
print(f"\nBest max_depth: {grid_search.best_params_['max_depth']}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# TODO: Evaluate on test set using best model
test_score = _______________
print(f"Test set score: {test_score:.3f}")
```

Task C: Discussion Questions

1. Compare the manual CV and GridSearchCV approaches:
   - Which is easier to implement?
   - Which gives you more control?
   - When would you use each?

2. Look at your results:
   - Which max_depth was selected?
   - Is there evidence of overfitting? (Compare CV score to test score)
   - What happens if you increase max_depth to 20 or 30?

3. Critical Thinking:
   - You used the SAME data to select max_depth (via CV) and then evaluate on the test set. Is this truly unbiased?
   - How could you implement a more rigorous validation strategy? (Hint: nested CV or train/val/test split)