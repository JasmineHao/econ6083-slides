In-Class Practice: Regularization Methods

Course: Machine Learning for Economists
Topic: Ridge, Lasso, and Elastic Net

Part I: Conceptual Check (Circle the best answer)

1. The Bias-Variance Trade-off
In Ridge Regression, as we increase the penalty parameter $\lambda$ (lambda) from 0 to $\infty$:

(A) The Bias decreases, and Variance increases.

(B) The Bias increases, and Variance decreases.

(C) Both Bias and Variance decrease.

(D) Both Bias and Variance increase.

2. The Geometry of Sparsity
Why does Lasso ($\ell_1$ penalty) tend to produce sparse solutions (coefficients exactly zero) while Ridge ($\ell_2$ penalty) does not?

(A) Because the $\ell_1$ cost function is differentiable everywhere.

(B) Because the $\ell_1$ constraint region (diamond) has corners on the axes, making it probable for the RSS contours to touch there.

(C) Because Lasso uses a squared penalty which punishes large coefficients more.

(D) Because Lasso is computationally faster to solve than Ridge.

3. The Standardization Trap
Suppose you are running a Lasso regression to predict House Prices.

$X_1$: Distance to city center (measured in km). Range: [0, 50].

$X_2$: Size of the house (measured in sq meters). Range: [50, 500].
If you do not standardize these variables before running Lasso, which variable is more likely to be penalized heavily (shrunk to zero) by the model, assuming their true effects are roughly equal in economic magnitude?

(A) $X_1$ (Distance)

(B) $X_2$ (Size)

(C) Neither, Lasso is scale-invariant.

Part II: Analytical Derivation (The "Toy" Example)

Consider a simple regression with one variable and no intercept:


$$y = \beta x + \epsilon$$


Assume we have a single observation $(x, y) = (1, 3)$.
We want to estimate $\beta$.

1. OLS Estimate:
Minimize $(y - \beta x)^2$. Since $x=1$ and $y=3$, we minimize $(3 - \beta)^2$.

$\hat{\beta}_{OLS} =$ ___________

2. Ridge Estimate:
Minimize $(y - \beta x)^2 + \lambda \beta^2$. Let $\lambda = 1$.
Objective: $(3 - \beta)^2 + 1 \cdot \beta^2$.
Find the $\beta$ that minimizes this.

$\hat{\beta}_{Ridge} =$ ___________

Interpretation: Compared to OLS, did the coefficient move towards zero?

3. Lasso Estimate:
Minimize $\frac{1}{2}(y - \beta x)^2 + \lambda |\beta|$. Let $\lambda = 2$.
Objective: $\frac{1}{2}(3 - \beta)^2 + 2|\beta|$.
(Hint: Think about the Soft-Thresholding intuition. Does the marginal benefit of reducing the residual justify the cost of increasing $|\beta|$?)

$\hat{\beta}_{Lasso} =$ ___________

Interpretation: Did the coefficient become zero? Why?

Part III: Applied Scenario (Group Discussion)

Scenario: You are a researcher trying to estimate the causal effect of Class Size on Test Scores. You have a dataset with 500 observations but 200 potential control variables (student demographics, teacher characteristics, school facilities, neighborhood data, interactions, etc.).

Discussion:

Why is running a simple OLS with all 200 controls a bad idea?

Your colleague suggests: "Just run Lasso of Test Score on Class Size and all 200 controls. The coefficient on Class Size is your answer." Why is this approach (Naive Lasso) potentially biased? (Think about Omitted Variable Bias).