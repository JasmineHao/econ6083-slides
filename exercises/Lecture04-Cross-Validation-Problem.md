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