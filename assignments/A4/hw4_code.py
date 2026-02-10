"""
Assignment 4: Heterogeneous Treatment Effects with Causal Forests
ECON6083 - Machine Learning in Economics

Student Name: [Your Name]
Student ID: [Your ID]
Date: [Date]

This template provides the structure for analyzing heterogeneous treatment effects
using Causal Forests. Complete all sections marked with TODO.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# For Causal Forest
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# Set random seed for reproducibility
np.random.seed(42)

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class HTEAnalyzer:
    """
    Class for analyzing heterogeneous treatment effects using Causal Forests.
    """

    def __init__(self, data_path):
        """
        Initialize the analyzer with data.

        Parameters:
        -----------
        data_path : str
            Path to the job_training.csv file
        """
        self.data_path = data_path
        self.df = None
        self.cf_model = None
        self.cate = None

    def load_data(self):
        """
        Load and display basic information about the dataset.
        """
        # TODO: Load the data from self.data_path
        self.df = pd.read_csv(self.data_path)

        # TODO: Print dataset shape
        print(f"Dataset shape: {self.df.shape}")
        print("\n" + "="*50)

        # TODO: Display first few rows
        print("First 5 rows:")
        print(self.df.head())
        print("\n" + "="*50)

        # TODO: Display summary statistics
        print("Summary statistics:")
        print(self.df.describe())
        print("\n" + "="*50)

        # TODO: Check for missing values
        print("Missing values:")
        print(self.df.isnull().sum())
        print("\n" + "="*50)

        return self.df

    def check_covariate_balance(self):
        """
        Check covariate balance between treated and control groups.
        This verifies the randomization was successful.
        """
        print("\n" + "="*50)
        print("PART I: COVARIATE BALANCE")
        print("="*50 + "\n")

        # TODO: Define covariates to check
        covariates = ['age', 'education', 'prior_earnings', 'female', 'married', 'children']

        # Create balance table
        balance_results = []

        for covar in covariates:
            # TODO: Calculate means for treated and control
            treated_mean = self.df[self.df['treated'] == 1][covar].mean()
            control_mean = self.df[self.df['treated'] == 0][covar].mean()

            # TODO: Calculate difference
            diff = treated_mean - control_mean

            # TODO: Perform t-test
            treated_vals = self.df[self.df['treated'] == 1][covar]
            control_vals = self.df[self.df['treated'] == 0][covar]
            t_stat, p_value = stats.ttest_ind(treated_vals, control_vals)

            balance_results.append({
                'Covariate': covar,
                'Treated Mean': treated_mean,
                'Control Mean': control_mean,
                'Difference': diff,
                'p-value': p_value
            })

        # Create and display balance table
        balance_df = pd.DataFrame(balance_results)
        print("Covariate Balance Table:")
        print(balance_df.to_string(index=False))
        print("\n")

        # Interpretation
        print("Interpretation:")
        print("- If p-values > 0.05, treatment is balanced (randomization successful)")
        print("- Large differences suggest potential balance issues")
        print("\n" + "="*50)

        return balance_df

    def visualize_outcomes(self):
        """
        Visualize outcome distributions by treatment status.
        """
        # TODO: Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Histograms
        treated_earnings = self.df[self.df['treated'] == 1]['earnings_post']
        control_earnings = self.df[self.df['treated'] == 0]['earnings_post']

        axes[0].hist(control_earnings, bins=30, alpha=0.6, label='Control', color='blue', edgecolor='black')
        axes[0].hist(treated_earnings, bins=30, alpha=0.6, label='Treated', color='red', edgecolor='black')
        axes[0].set_xlabel('Post-Program Earnings ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Outcome Distribution by Treatment Status')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Subplot 2: Box plots
        self.df.boxplot(column='earnings_post', by='treated', ax=axes[1])
        axes[1].set_xlabel('Treatment Status')
        axes[1].set_ylabel('Post-Program Earnings ($)')
        axes[1].set_title('Earnings by Treatment Status')
        axes[1].set_xticklabels(['Control', 'Treated'])
        plt.suptitle('')  # Remove default title

        plt.tight_layout()
        plt.savefig('outcome_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nOutcome visualizations saved as 'outcome_distributions.png'")

    def calculate_naive_ate(self):
        """
        Calculate naive Average Treatment Effect (simple difference in means).
        """
        print("\n" + "="*50)
        print("NAIVE AVERAGE TREATMENT EFFECT (ATE)")
        print("="*50 + "\n")

        # TODO: Calculate mean outcomes for treated and control
        treated_mean = self.df[self.df['treated'] == 1]['earnings_post'].mean()
        control_mean = self.df[self.df['treated'] == 0]['earnings_post'].mean()

        # TODO: Calculate ATE
        ate = treated_mean - control_mean

        # TODO: Calculate standard error (using pooled variance)
        treated_vals = self.df[self.df['treated'] == 1]['earnings_post']
        control_vals = self.df[self.df['treated'] == 0]['earnings_post']

        n1 = len(treated_vals)
        n0 = len(control_vals)
        var1 = treated_vals.var()
        var0 = control_vals.var()

        se = np.sqrt(var1/n1 + var0/n0)

        # TODO: Calculate 95% confidence interval
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        print(f"Treated mean: ${treated_mean:,.2f}")
        print(f"Control mean: ${control_mean:,.2f}")
        print(f"\nNaive ATE: ${ate:,.2f}")
        print(f"Standard Error: ${se:,.2f}")
        print(f"95% CI: [${ci_lower:,.2f}, ${ci_upper:,.2f}]")
        print("\n" + "="*50)

        return ate, se, ci_lower, ci_upper

    def train_causal_forest(self):
        """
        Train Causal Forest using CausalForestDML from EconML.
        """
        print("\n" + "="*50)
        print("PART II: CAUSAL FOREST TRAINING")
        print("="*50 + "\n")

        # TODO: Define features (X), treatment (T), and outcome (Y)
        feature_cols = ['age', 'education', 'prior_earnings', 'female', 'married', 'children']
        X = self.df[feature_cols].values
        T = self.df['treated'].values
        Y = self.df['earnings_post'].values

        print(f"Features shape: {X.shape}")
        print(f"Treatment shape: {T.shape}")
        print(f"Outcome shape: {Y.shape}\n")

        # TODO: Define nuisance models for E[Y|X] and E[T|X]
        model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
        model_t = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)

        # TODO: Initialize Causal Forest
        # Key parameters:
        # - n_estimators: Number of trees (use 1000 for stable estimates)
        # - min_samples_leaf: Minimum samples per leaf (prevents overfitting)
        # - honest: Use honest splitting (True)
        # - random_state: For reproducibility (42)

        self.cf_model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=1000,
            min_samples_leaf=10,
            honest=True,
            random_state=42
        )

        print("Training Causal Forest...")
        print("(This may take 1-2 minutes)")

        # TODO: Fit the model
        self.cf_model.fit(Y, T, X=X)

        print("Training complete!\n")

        # TODO: Predict CATE for all individuals
        self.cate = self.cf_model.effect(X)

        # Add CATE to dataframe
        self.df['cate'] = self.cate

        # Display CATE summary statistics
        print("CATE Summary Statistics:")
        print(f"  Mean CATE: ${self.cate.mean():,.2f}")
        print(f"  Std Dev: ${self.cate.std():,.2f}")
        print(f"  Min CATE: ${self.cate.min():,.2f}")
        print(f"  Max CATE: ${self.cate.max():,.2f}")
        print(f"  Median CATE: ${np.median(self.cate):,.2f}")
        print(f"\n  25th percentile: ${np.percentile(self.cate, 25):,.2f}")
        print(f"  75th percentile: ${np.percentile(self.cate, 75):,.2f}")
        print("\n" + "="*50)

        return self.cate

    def analyze_heterogeneity(self):
        """
        Analyze and visualize treatment effect heterogeneity.
        """
        print("\n" + "="*50)
        print("HETEROGENEITY ANALYSIS")
        print("="*50 + "\n")

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: CATE distribution
        axes[0, 0].hist(self.cate, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(self.cate.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: ${self.cate.mean():.0f}')
        axes[0, 0].set_xlabel('CATE (dollars)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Conditional Average Treatment Effects')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # TODO: Plot 2: CATE vs Education
        axes[0, 1].scatter(self.df['education'], self.cate, alpha=0.5, s=20)
        axes[0, 1].axhline(self.cate.mean(), color='red', linestyle='--', linewidth=2, label='Mean CATE')
        axes[0, 1].set_xlabel('Years of Education')
        axes[0, 1].set_ylabel('CATE (dollars)')
        axes[0, 1].set_title('CATE vs Education')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # TODO: Plot 3: CATE vs Prior Earnings
        axes[1, 0].scatter(self.df['prior_earnings'], self.cate, alpha=0.5, s=20, color='green')
        axes[1, 0].axhline(self.cate.mean(), color='red', linestyle='--', linewidth=2, label='Mean CATE')
        axes[1, 0].set_xlabel('Prior Earnings ($1000s)')
        axes[1, 0].set_ylabel('CATE (dollars)')
        axes[1, 0].set_title('CATE vs Prior Earnings')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # TODO: Plot 4: CATE by Education Level (< 12 vs >= 12)
        self.df['low_education'] = (self.df['education'] < 12).astype(int)
        data_for_box = [
            self.df[self.df['low_education'] == 1]['cate'],
            self.df[self.df['low_education'] == 0]['cate']
        ]
        axes[1, 1].boxplot(data_for_box, labels=['Education < 12', 'Education ≥ 12'])
        axes[1, 1].set_ylabel('CATE (dollars)')
        axes[1, 1].set_title('CATE by Education Level')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('heterogeneity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Heterogeneity visualizations saved as 'heterogeneity_analysis.png'\n")

        # Additional subgroup analysis
        print("Subgroup Comparison:")
        print(f"  Low education (< 12 years): Mean CATE = ${self.df[self.df['low_education']==1]['cate'].mean():,.2f}")
        print(f"  High education (≥ 12 years): Mean CATE = ${self.df[self.df['low_education']==0]['cate'].mean():,.2f}")
        print(f"  Difference: ${self.df[self.df['low_education']==1]['cate'].mean() - self.df[self.df['low_education']==0]['cate'].mean():,.2f}")
        print("\n" + "="*50)

    def identify_high_responders(self):
        """
        Identify and characterize high responders (those with high CATE).
        """
        print("\n" + "="*50)
        print("PART III: IDENTIFYING HIGH RESPONDERS")
        print("="*50 + "\n")

        # TODO: Calculate percentage with CATE > $3000
        pct_high = (self.cate > 3000).mean() * 100
        print(f"Percentage with CATE > $3,000: {pct_high:.1f}%\n")

        # TODO: Define high and low responders
        p75 = np.percentile(self.cate, 75)
        p25 = np.percentile(self.cate, 25)

        high_responders = self.df[self.cate > p75]
        low_responders = self.df[self.cate < p25]

        # TODO: Create comparison table
        comparison = {
            'Characteristic': [],
            'High Responders (CATE > p75)': [],
            'Low Responders (CATE < p25)': [],
            'Difference': []
        }

        covariates = ['age', 'education', 'prior_earnings', 'female', 'married', 'children']

        for covar in covariates:
            comparison['Characteristic'].append(covar)
            high_mean = high_responders[covar].mean()
            low_mean = low_responders[covar].mean()
            comparison['High Responders (CATE > p75)'].append(f"{high_mean:.2f}")
            comparison['Low Responders (CATE < p25)'].append(f"{low_mean:.2f}")
            comparison['Difference'].append(f"{high_mean - low_mean:.2f}")

        comparison_df = pd.DataFrame(comparison)
        print("High vs Low Responders Comparison:")
        print(comparison_df.to_string(index=False))
        print("\n" + "="*50)

        return comparison_df

    def design_targeting_policy(self, target_pct=0.30):
        """
        Design and evaluate targeting policy.

        Parameters:
        -----------
        target_pct : float
            Fraction of population to treat (default: 0.30 for 30%)
        """
        print("\n" + "="*50)
        print(f"POLICY DESIGN: TARGETING TOP {int(target_pct*100)}%")
        print("="*50 + "\n")

        N = len(self.df)
        n_treat = int(target_pct * N)

        # TODO: Policy 1 - Random selection
        np.random.seed(42)
        random_idx = np.random.choice(N, size=n_treat, replace=False)
        ate_random = self.cate[random_idx].mean()

        # TODO: Policy 2 - Targeting (select top N by CATE)
        # HINT: Use np.argsort() to get indices sorted by CATE
        # HINT: Use [-n_treat:] to get the TOP n_treat individuals
        top_idx = np.argsort(self.cate)[-n_treat:]
        ate_targeted = self.cate[top_idx].mean()

        # TODO: Calculate policy gains
        gain_dollars = ate_targeted - ate_random
        gain_pct = (gain_dollars / ate_random) * 100

        print("Policy Comparison:")
        print(f"  Random Selection ({int(target_pct*100)}%):")
        print(f"    - Average CATE: ${ate_random:,.2f}")
        print(f"    - Total gain (if {n_treat} treated): ${ate_random * n_treat:,.0f}\n")

        print(f"  Targeted Selection (Top {int(target_pct*100)}% by CATE):")
        print(f"    - Average CATE: ${ate_targeted:,.2f}")
        print(f"    - Total gain (if {n_treat} treated): ${ate_targeted * n_treat:,.0f}\n")

        print(f"  Policy Gain from Targeting:")
        print(f"    - Absolute gain: ${gain_dollars:,.2f} per person")
        print(f"    - Percentage gain: {gain_pct:.1f}%")
        print(f"    - Extra earnings generated: ${gain_dollars * n_treat:,.0f}\n")

        print("="*50)

        return {
            'random_ate': ate_random,
            'targeted_ate': ate_targeted,
            'gain_dollars': gain_dollars,
            'gain_pct': gain_pct
        }

    def evaluate_policy(self, training_cost=5000):
        """
        Evaluate policy with cost-benefit analysis and sensitivity.

        Parameters:
        -----------
        training_cost : float
            Cost per person to provide training (default: $5000)
        """
        print("\n" + "="*50)
        print("POLICY EVALUATION")
        print("="*50 + "\n")

        # TODO: Sensitivity analysis across different targeting levels
        print("Sensitivity Analysis: Varying Targeting Levels\n")

        sensitivity_results = []

        for pct in [0.20, 0.30, 0.40, 0.50]:
            N = len(self.df)
            n_treat = int(pct * N)

            # Random baseline
            np.random.seed(42)
            random_idx = np.random.choice(N, size=n_treat, replace=False)
            ate_random = self.cate[random_idx].mean()

            # Targeted
            top_idx = np.argsort(self.cate)[-n_treat:]
            ate_targeted = self.cate[top_idx].mean()

            # Gain
            gain = ate_targeted - ate_random
            gain_pct = (gain / ate_random) * 100

            sensitivity_results.append({
                'Targeting Level': f'Top {int(pct*100)}%',
                'Avg CATE (Targeted)': f'${ate_targeted:,.0f}',
                'Avg CATE (Random)': f'${ate_random:,.0f}',
                'Absolute Gain': f'${gain:,.0f}',
                'Gain %': f'{gain_pct:.1f}%'
            })

        sens_df = pd.DataFrame(sensitivity_results)
        print(sens_df.to_string(index=False))
        print("\n" + "="*50 + "\n")

        # TODO: Cost-benefit analysis
        print("Cost-Benefit Analysis:")
        print(f"Training cost per person: ${training_cost:,.0f}\n")

        # Check if program is cost-effective at different targeting levels
        for pct in [0.30, 0.50]:
            n_treat = int(pct * len(self.df))
            top_idx = np.argsort(self.cate)[-n_treat:]
            ate_targeted = self.cate[top_idx].mean()

            net_benefit = ate_targeted - training_cost
            roi = (net_benefit / training_cost) * 100

            print(f"  Top {int(pct*100)}% targeting:")
            print(f"    - Average CATE: ${ate_targeted:,.0f}")
            print(f"    - Training cost: ${training_cost:,.0f}")
            print(f"    - Net benefit: ${net_benefit:,.0f}")
            print(f"    - ROI: {roi:.1f}%")

            if net_benefit > 0:
                print(f"    - Cost-effective: YES ✓")
            else:
                print(f"    - Cost-effective: NO ✗")
            print()

        print("="*50)

        # TODO: Equity analysis
        print("\n" + "="*50)
        print("EQUITY ANALYSIS")
        print("="*50 + "\n")

        # Compare selected vs not selected under top 30% targeting
        n_treat = int(0.30 * len(self.df))
        top_idx = np.argsort(self.cate)[-n_treat:]

        self.df['selected'] = 0
        self.df.loc[top_idx, 'selected'] = 1

        print("Characteristics of Selected vs Not Selected (Top 30% policy):\n")

        for covar in ['age', 'education', 'prior_earnings', 'female']:
            selected_mean = self.df[self.df['selected'] == 1][covar].mean()
            not_selected_mean = self.df[self.df['selected'] == 0][covar].mean()
            print(f"  {covar}:")
            print(f"    - Selected: {selected_mean:.2f}")
            print(f"    - Not selected: {not_selected_mean:.2f}")
            print(f"    - Difference: {selected_mean - not_selected_mean:.2f}\n")

        print("="*50)

        return sens_df


def main():
    """
    Main function to run the complete analysis.
    """
    # TODO: Set path to data file
    data_path = '../data/job_training.csv'

    # Initialize analyzer
    analyzer = HTEAnalyzer(data_path)

    # PART I: Exploratory Analysis
    print("\n" + "#"*60)
    print("# ASSIGNMENT 4: HETEROGENEOUS TREATMENT EFFECTS")
    print("#"*60)

    analyzer.load_data()
    analyzer.check_covariate_balance()
    analyzer.visualize_outcomes()
    analyzer.calculate_naive_ate()

    # PART II: Causal Forest
    analyzer.train_causal_forest()
    analyzer.analyze_heterogeneity()

    # PART III: Policy Design
    analyzer.identify_high_responders()
    analyzer.design_targeting_policy(target_pct=0.30)
    analyzer.evaluate_policy(training_cost=5000)

    print("\n" + "#"*60)
    print("# ANALYSIS COMPLETE")
    print("#"*60)
    print("\nPlease review the outputs and fill in your report (hw4_report.md)")
    print("Visualizations saved:")
    print("  - outcome_distributions.png")
    print("  - heterogeneity_analysis.png")


if __name__ == "__main__":
    main()
