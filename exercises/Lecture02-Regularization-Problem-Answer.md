Instructor Guide: Solutions & Talking Points

Part I: Solutions

(B) Bias increases, Variance decreases.

Explanation: $\lambda=0$ is OLS (Unbiased, High Variance). $\lambda=\infty$ sets $\beta=0$ (Highly Biased, Zero Variance). The goal is to find the sweet spot.

(B) Corners on the axes.

Explanation: The geometric intersection of the RSS ellipses and the $\ell_1$ diamond usually happens at a vertex. The vertex lies on an axis where at least one coordinate is 0.

(B) $X_2$ (Size).

Explanation: This is tricky but crucial.

$X_2$ (Size) has a larger range (magnitude) than $X_1$ (Distance).

To have the same impact on $Y$ (Price), the coefficient $\beta_2$ for Size needs to be smaller than $\beta_1$. (e.g., Price increases $1000 per sq meter vs $50,000 per km).

Since $\beta_2$ is already numerically smaller, the penalty term $\lambda |\beta_2|$ is smaller, but more importantly, standardization puts them on equal footing.

Wait, let's re-evaluate: If we don't standardize, variables with large variance ($X_2$) tend to have small coefficients. Variables with small variance ($X_1$) tend to have large coefficients.

Lasso penalizes the magnitude of $\beta$. It will try to shrink the large coefficient ($\beta_1$) more aggressively?

Correction: Actually, the standard result is that Lasso unfairly penalizes variables with small variance (which require large coefficients to be effective). Wait, if $X$ is large, $\beta$ is small. Lasso penalizes $\beta$. So a small $\beta$ is "cheaper" for Lasso.

Therefore, Lasso prefers variables with large scale (because they allow small $\beta$). It will penalize $X_1$ (Small scale $\rightarrow$ Large $\beta$) more heavily.

Correct Answer Revision: (A) $X_1$ (Distance). Lasso penalizes the variable that requires a large coefficient to do its job. $X_1$ is small range, so it needs a huge $\beta$. Lasso hates huge $\beta$. So $X_1$ gets killed.

Part II: Analytical Solutions

1. OLS:

Minimize $(3-\beta)^2$. Derivative: $-2(3-\beta)=0 \rightarrow \beta = 3$.

$\hat{\beta}_{OLS} = 3$.

2. Ridge ($\lambda=1$):

Minimize $(3-\beta)^2 + \beta^2$.

Derivative: $-2(3-\beta) + 2\beta = 0 \rightarrow -6 + 2\beta + 2\beta = 0 \rightarrow 4\beta = 6 \rightarrow \beta = 1.5$.

Interpretation: The coefficient shrank from 3 to 1.5. (Shrinkage Factor = $1/(1+\lambda) = 1/2$).

3. Lasso ($\lambda=2$):

Minimize $\frac{1}{2}(3-\beta)^2 + 2|\beta|$.

Subgradient at $\beta > 0$: $-(3-\beta) + 2 = 0 \rightarrow \beta - 1 = 0 \rightarrow \beta = 1$.

Wait, let's check the condition. Soft Thresholding formula: $S(z, \lambda) = \text{sign}(z) \max(|z| - \lambda, 0)$.

Here OLS $\hat{\beta} = 3$. Threshold is $\lambda = 2$ (Note: factor $1/2$ in objective usually implies threshold $\lambda$).

If objective is $\frac{1}{2}RSS + \lambda|\beta|$, threshold is $\lambda$.

$\hat{\beta}_{Lasso} = \max(3 - 2, 0) = 1$.

Alternative setup: If objective was $RSS + \lambda|\beta|$ (no 1/2), threshold is $\lambda/2$.

Let's stick to the calculation: $-(3-\beta) + 2 = 0 \Rightarrow \beta = 1$.

Interpretation: The coefficient shrank from 3 to 1. It is not zero yet.

Follow up: What if $\lambda = 4$? Then $\max(3-4, 0) = 0$. The coefficient would be zero.

Part III: Discussion Guide

OLS Issues: Overfitting. With 500 obs and 200 vars, $p/n = 0.4$. The variance will be huge. Standard errors will be blown up. Likely to find spurious correlations.

Naive Lasso Bias: Lasso's job is prediction, not causal inference. It might drop a variable $X_j$ that is highly correlated with Class Size (Treatment) but only weakly correlated with Test Score (Outcome). Dropping this confounder creates Omitted Variable Bias in the coefficient of Class Size. We need "Double Selection" (Lecture 4) to fix this.