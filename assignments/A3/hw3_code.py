"""
ECON6083 - Assignment 3: Demand Estimation with IV-DML
Student Template

Instructions:
1. Complete all functions marked with TODO
2. Do not change function names or signatures
3. You may add helper functions if needed
4. Test your code before submission

Submission:
- Rename this file to: hw3_code_<StudentID>_<LastName>.py
- Submit to Moodle along with hw3_report.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')

# Optional: DoubleML (recommended for Part II, Task 2.3)
try:
    from doubleml import DoubleMLData, DoubleMLPLIV
    DOUBLEML_AVAILABLE = True
except ImportError:
    DOUBLEML_AVAILABLE = False
    print("Warning: DoubleML not installed. Install with: pip install doubleml")


class DemandEstimator:
    """
    Estimator for price elasticity of demand using OLS, 2SLS, and DML-IV.
    """

    def __init__(self, data_path='data/grocery_demand.csv', random_state=42):
        """
        Initialize the estimator.

        Parameters:
        -----------
        data_path : str
            Path to the dataset
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.data_path = data_path
        self.df = None
        self.results = {}

    def load_data(self):
        """
        Load the dataset and create log variables.

        Returns:
        --------
        df : pandas.DataFrame
            The loaded dataset with log transformations
        """
        # TODO: Load data from self.data_path
        # YOUR CODE HERE
        self.df = pd.read_csv(self.data_path)

        # TODO: Create log-transformed variables
        # Hint: np.log()
        # YOUR CODE HERE
        self.df['log_quantity'] = np.log(self.df['quantity'])
        self.df['log_price'] = np.log(self.df['price'])

        print(f"Data loaded: {self.df.shape[0]} observations, {self.df.shape[1]} variables")
        print(f"\nColumns: {list(self.df.columns)}")

        return self.df

    def explore_data(self):
        """
        Part I: Data exploration and visualization.
        """
        print("\n" + "="*80)
        print("PART I: DATA EXPLORATION")
        print("="*80)

        # Summary statistics
        print("\nSummary Statistics:")
        print(self.df[['quantity', 'price', 'cost_shock', 'income', 'population']].describe())

        # TODO: Task 1.2 - Plot log(price) vs log(quantity)
        # YOUR CODE HERE
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['log_price'], self.df['log_quantity'], alpha=0.3)
        plt.xlabel('Log(Price)', fontsize=12)
        plt.ylabel('Log(Quantity)', fontsize=12)
        plt.title('Demand Curve: Price vs Quantity (Log Scale)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.savefig('price_quantity_scatter.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved: price_quantity_scatter.png")
        # plt.show()  # Uncomment if running interactively

        # TODO: Task 1.3 - Plot cost_shock vs price (first stage)
        # YOUR CODE HERE
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['cost_shock'], self.df['price'], alpha=0.3)
        plt.xlabel('Cost Shock (Instrument)', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.title('First Stage: IV vs Price', fontsize=14)
        plt.grid(alpha=0.3)
        plt.savefig('first_stage_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved: first_stage_plot.png")
        # plt.show()

    def prepare_controls(self):
        """
        Prepare control variables (X) including fixed effects.

        Returns:
        --------
        X_controls : pandas.DataFrame
            DataFrame with all control variables
        """
        # TODO: Create control variables
        # Include: income, population, trend
        # Add dummies for store_id and product_category (fixed effects)

        # YOUR CODE HERE
        # Start with continuous controls
        controls = self.df[['income', 'population', 'trend']].copy()

        # Add categorical dummies (fixed effects)
        store_dummies = pd.get_dummies(self.df['store_id'], prefix='store', drop_first=True)
        category_dummies = pd.get_dummies(self.df['product_category'], prefix='category', drop_first=True)

        # Combine all controls
        X_controls = pd.concat([controls, store_dummies, category_dummies], axis=1)

        return X_controls

    def estimate_ols(self):
        """
        Part II, Task 2.1: Estimate demand using OLS (ignoring endogeneity).

        Returns:
        --------
        results_dict : dict
            Dictionary with elasticity, std error, and confidence interval
        """
        print("\n" + "="*80)
        print("TASK 2.1: NAIVE OLS ESTIMATION")
        print("="*80)

        # TODO: Prepare variables for OLS
        y = self.df['log_quantity']
        X_controls = self.prepare_controls()

        # TODO: Create X matrix with log_price and controls
        # Hint: Use pd.concat to combine log_price with X_controls
        # Hint: Use sm.add_constant() to add intercept
        # YOUR CODE HERE
        X = pd.concat([self.df[['log_price']], X_controls], axis=1)
        X = sm.add_constant(X)

        # TODO: Run OLS regression with robust standard errors
        # Hint: sm.OLS(y, X).fit(cov_type='HC3')
        # YOUR CODE HERE
        model = sm.OLS(y, X).fit(cov_type='HC3')

        # Extract results
        elasticity = model.params['log_price']
        std_error = model.bse['log_price']
        ci_lower, ci_upper = model.conf_int().loc['log_price']

        # Store results
        self.results['OLS'] = {
            'elasticity': elasticity,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

        print(f"\nOLS Results:")
        print(f"  Price Elasticity: {elasticity:.4f}")
        print(f"  Standard Error:   {std_error:.4f}")
        print(f"  95% CI:           [{ci_lower:.4f}, {ci_upper:.4f}]")

        return self.results['OLS']

    def estimate_2sls(self):
        """
        Part II, Task 2.2: Estimate demand using 2SLS with cost_shock as IV.

        Returns:
        --------
        results_dict : dict
            Dictionary with elasticity, std error, CI, and first stage F-stat
        """
        print("\n" + "="*80)
        print("TASK 2.2: 2SLS ESTIMATION")
        print("="*80)

        # TODO: Prepare variables
        y = self.df['log_quantity']
        X_controls = self.prepare_controls()
        X_endog = self.df[['log_price']]  # Endogenous variable
        Z = self.df[['cost_shock']]  # Instrument

        # TODO: Run 2SLS using linearmodels
        # Hint: IV2SLS(dependent=y, exog=X_controls, endog=X_endog, instruments=Z).fit(cov_type='robust')
        # YOUR CODE HERE
        model = IV2SLS(dependent=y, exog=X_controls, endog=X_endog, instruments=Z).fit(cov_type='robust')

        # Extract results
        elasticity = model.params['log_price']
        std_error = model.std_errors['log_price']
        ci = model.conf_int()
        ci_lower, ci_upper = ci.loc['log_price']

        # TODO: Extract first stage F-statistic
        # Hint: model.first_stage.diagnostics['f.stat'].iloc[0]
        # YOUR CODE HERE
        first_stage_f = model.first_stage.diagnostics['f.stat'].iloc[0]

        # Store results
        self.results['2SLS'] = {
            'elasticity': elasticity,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'first_stage_f': first_stage_f
        }

        print(f"\n2SLS Results:")
        print(f"  Price Elasticity:    {elasticity:.4f}")
        print(f"  Standard Error:      {std_error:.4f}")
        print(f"  95% CI:              [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  First Stage F-stat:  {first_stage_f:.2f}")

        if first_stage_f < 10:
            print("  ⚠️ WARNING: Weak instrument (F < 10)")
        else:
            print("  ✓ Strong instrument (F > 10)")

        return self.results['2SLS']

    def estimate_dml_iv(self, use_doubleml=True):
        """
        Part II, Task 2.3: Estimate demand using DML-IV.

        Parameters:
        -----------
        use_doubleml : bool
            If True, use DoubleML package (recommended)
            If False, use manual implementation

        Returns:
        --------
        results_dict : dict
            Dictionary with elasticity, std error, and confidence interval
        """
        print("\n" + "="*80)
        print("TASK 2.3: DML-IV ESTIMATION")
        print("="*80)

        if use_doubleml and DOUBLEML_AVAILABLE:
            return self._estimate_dml_iv_doubleml()
        else:
            return self._estimate_dml_iv_manual()

    def _estimate_dml_iv_doubleml(self):
        """
        DML-IV using DoubleML package.
        """
        print("Using DoubleML package...")

        # TODO: Prepare data for DoubleML
        # Create a DataFrame with necessary columns
        dml_df = self.df[['log_quantity', 'log_price', 'cost_shock',
                          'income', 'population', 'trend']].copy()

        # TODO: Create DoubleMLData object
        # Hint: DoubleMLData(df, y_col='...', d_cols='...', z_cols='...', x_cols=[...])
        # YOUR CODE HERE
        dml_data = DoubleMLData(
            dml_df,
            y_col='log_quantity',
            d_cols='log_price',
            z_cols='cost_shock',
            x_cols=['income', 'population', 'trend']
        )

        # TODO: Specify ML methods for nuisance parameters
        # ml_g: E[Y | X, Z] (reduced form)
        # ml_m: E[D | X, Z] (first stage)
        # Use RandomForestRegressor or LassoCV
        # YOUR CODE HERE
        ml_g = RandomForestRegressor(n_estimators=100, max_depth=10,
                                      min_samples_leaf=20, random_state=self.random_state)
        ml_m = RandomForestRegressor(n_estimators=100, max_depth=10,
                                      min_samples_leaf=20, random_state=self.random_state)

        # TODO: Create DoubleMLPLIV model
        # Hint: DoubleMLPLIV(dml_data, ml_g, ml_m, n_folds=5)
        # YOUR CODE HERE
        dml_model = DoubleMLPLIV(dml_data, ml_g, ml_m, n_folds=5, n_rep=1)

        # TODO: Fit the model
        # YOUR CODE HERE
        dml_model.fit()

        # Extract results
        elasticity = dml_model.coef[0]
        std_error = dml_model.se[0]
        ci = dml_model.confint()
        ci_lower = ci['2.5 %'].iloc[0]
        ci_upper = ci['97.5 %'].iloc[0]

        # Store results
        self.results['DML-IV'] = {
            'elasticity': elasticity,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

        print(f"\nDML-IV Results (DoubleML):")
        print(f"  Price Elasticity: {elasticity:.4f}")
        print(f"  Standard Error:   {std_error:.4f}")
        print(f"  95% CI:           [{ci_lower:.4f}, {ci_upper:.4f}]")

        return self.results['DML-IV']

    def _estimate_dml_iv_manual(self):
        """
        Manual DML-IV implementation (Advanced).

        This is for students who want to implement DML-IV by hand,
        similar to the in-class exercise.
        """
        print("Manual DML-IV implementation...")
        print("(This is an advanced option. Most students should use DoubleML.)")

        # TODO (Advanced): Implement manual DML-IV with sample splitting
        # See Lecture 8 exercise for reference
        # Steps:
        # 1. Split sample
        # 2. Fit reduced form and first stage with ML on one half
        # 3. Predict on the other half
        # 4. Compute residuals
        # 5. IV regression on residuals
        # 6. Repeat with cross-fitting

        # YOUR CODE HERE (optional)
        print("\n⚠️ Manual implementation not completed.")
        print("   Please use DoubleML instead (set use_doubleml=True)")

        self.results['DML-IV'] = {
            'elasticity': np.nan,
            'std_error': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        }

        return self.results['DML-IV']

    def compare_results(self):
        """
        Part III: Compare all three methods and create summary table.
        """
        print("\n" + "="*80)
        print("PART III: COMPARISON OF METHODS")
        print("="*80)

        # Create comparison table
        comparison = []
        for method in ['OLS', '2SLS', 'DML-IV']:
            if method in self.results:
                res = self.results[method]
                row = {
                    'Method': method,
                    'Elasticity': res['elasticity'],
                    'Std Error': res['std_error'],
                    '95% CI Lower': res['ci_lower'],
                    '95% CI Upper': res['ci_upper']
                }
                if method == '2SLS':
                    row['First Stage F'] = res.get('first_stage_f', np.nan)
                else:
                    row['First Stage F'] = '-'
                comparison.append(row)

        df_comparison = pd.DataFrame(comparison)

        print("\n" + "-"*80)
        print(df_comparison.to_string(index=False))
        print("-"*80)

        # Save to CSV for report
        df_comparison.to_csv('comparison_table.csv', index=False)
        print("\nComparison table saved to: comparison_table.csv")

        return df_comparison

    def run_full_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print("="*80)
        print("ECON6083 - Assignment 3: Demand Estimation with IV-DML")
        print("="*80)

        # Load data
        print("\n[Step 1/5] Loading data...")
        self.load_data()

        # Data exploration
        print("\n[Step 2/5] Exploring data...")
        self.explore_data()

        # OLS
        print("\n[Step 3/5] Running OLS...")
        self.estimate_ols()

        # 2SLS
        print("\n[Step 4/5] Running 2SLS...")
        self.estimate_2sls()

        # DML-IV
        print("\n[Step 5/5] Running DML-IV...")
        self.estimate_dml_iv(use_doubleml=DOUBLEML_AVAILABLE)

        # Comparison
        self.compare_results()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Review the comparison table above")
        print("2. Complete hw3_report.md with your economic analysis")
        print("3. Submit both hw3_code.py and hw3_report.md to Moodle")
        print("\nGood luck! 🚀")


# Main execution
if __name__ == "__main__":
    # Create estimator instance
    estimator = DemandEstimator(data_path='data/grocery_demand.csv', random_state=42)

    # Run full analysis
    estimator.run_full_analysis()

    print("\n✓ All estimators completed successfully!")
