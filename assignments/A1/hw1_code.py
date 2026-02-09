"""
ECON 491: Applied Machine Learning in Economics
Homework 1: Hedonic Pricing with OLS and Penalized Regression

Student Name: [Your Name Here]
Student ID: [Your ID Here]

Instructions:
- Complete all sections marked with TODO
- Do not change the class name or method signatures
- You may add helper methods as needed
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class EconomicHousingModel(BaseEstimator, RegressorMixin):
    """
    Automated Valuation Model (AVM) for housing prices using OLS and Penalized Regression.

    This model demonstrates the bias-variance tradeoff in high-dimensional economic data.
    """

    def __init__(self, model_type='ols', alpha=1.0):
        """
        Initialize the model.

        Parameters:
        -----------
        model_type : str, default='ols'
            Type of regression model: 'ols', 'ridge', or 'lasso'
        alpha : float, default=1.0
            Regularization strength (only used for ridge/lasso)
        """
        self.model_type = model_type
        self.alpha = alpha
        self.pipeline = None
        self.best_alpha = None

    def _create_features(self, df):
        """
        Feature engineering: Create economically meaningful features.

        TODO: Implement the following transformations:
        1. Create TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
        2. Create QualityArea = OverallQual * GrLivArea
        3. Create HouseAge = YrSold - YearBuilt
        4. Add at least one more meaningful feature

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        df : pd.DataFrame
            DataFrame with additional features
        """
        df = df.copy()

        # TODO: Implement feature engineering here
        # Example:
        # df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

        return df

    def _handle_missing_values(self, df):
        """
        Handle missing values with economic intuition.

        TODO: Implement missing value imputation:
        1. For features like PoolQC, Fence, Alley, etc., NA means "None" - fill with 'None'
        2. For numerical features, use median imputation
        3. Document your reasoning for each choice

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        df : pd.DataFrame
            DataFrame with missing values handled
        """
        df = df.copy()

        # TODO: Implement missing value handling here
        # Hint: Features like 'PoolQC', 'Fence', 'Alley', 'MiscFeature', 'FireplaceQu'
        # where NA means "None" (no pool, no fence, etc.)

        return df

    def _get_feature_names(self, df):
        """
        Separate numerical and categorical features.

        TODO: Identify which columns are numerical and which are categorical

        Returns:
        --------
        numerical_features : list
            List of numerical feature names
        categorical_features : list
            List of categorical feature names
        """
        # TODO: Implement feature type identification
        numerical_features = []
        categorical_features = []

        return numerical_features, categorical_features

    def fit(self, X, y, tune_hyperparameters=True):
        """
        Fit the model to training data.

        TODO: Implement the following:
        1. Apply feature engineering
        2. Handle missing values
        3. Create preprocessing pipeline with ColumnTransformer
        4. For ridge/lasso: use GridSearchCV to find optimal alpha
        5. Fit the model

        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series or np.array
            Training target (SalePrice)
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning (for ridge/lasso)

        Returns:
        --------
        self : object
            Fitted model
        """
        # TODO: Implement log transformation of target variable
        # y_transformed = np.log(y)

        # TODO: Apply feature engineering and missing value handling
        # X_processed = self._create_features(X)
        # X_processed = self._handle_missing_values(X_processed)

        # TODO: Get feature names
        # numerical_features, categorical_features = self._get_feature_names(X_processed)

        # TODO: Create preprocessing pipeline
        # Use ColumnTransformer to handle numerical (scaling) and categorical (one-hot) separately

        # TODO: Create full pipeline with preprocessing + model

        # TODO: If ridge/lasso and tune_hyperparameters=True, use GridSearchCV

        # TODO: Fit the model

        return self

    def predict(self, X):
        """
        Make predictions on new data.

        TODO: Implement the following:
        1. Apply same feature engineering as in fit
        2. Handle missing values
        3. Use pipeline to make predictions
        4. Transform predictions back from log scale

        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on

        Returns:
        --------
        predictions : np.array
            Predicted house prices (in original scale, not log)
        """
        # TODO: Implement prediction logic

        pass

    def get_feature_importance(self, feature_names=None, top_n=10):
        """
        Get feature importance (coefficients).

        TODO: Extract coefficients from the fitted model

        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        top_n : int
            Number of top features to return

        Returns:
        --------
        importance_df : pd.DataFrame
            DataFrame with features and their coefficients
        """
        # TODO: Implement feature importance extraction

        pass


def load_and_split_data(test_size=0.2, random_state=42):
    """
    Load Ames Housing data and split into train/test sets.

    TODO: Implement the following:
    1. Load data using sklearn.datasets.fetch_openml
    2. Handle target variable (SalePrice)
    3. Split into train/test

    Returns:
    --------
    X_train, X_test, y_train, y_test : DataFrames and Series
    """
    from sklearn.datasets import fetch_openml

    # TODO: Load Ames Housing data
    # Hint: fetch_openml(name="house_prices", version=1, as_frame=True, parser='auto')

    pass


def compare_models(X_train, X_test, y_train, y_test):
    """
    Compare OLS, Ridge, and Lasso models.

    TODO: Implement the following:
    1. Train all three models
    2. Make predictions on test set
    3. Calculate RMSE for each
    4. Return results as a dictionary

    Returns:
    --------
    results : dict
        Dictionary with model names as keys and RMSE as values
    """
    results = {}

    # TODO: Train and evaluate OLS

    # TODO: Train and evaluate Ridge (with hyperparameter tuning)

    # TODO: Train and evaluate Lasso (with hyperparameter tuning)

    return results


if __name__ == "__main__":
    """
    Main execution block for testing your implementation.
    """
    print("Loading Ames Housing Data...")
    # TODO: Load and split data

    print("\nTraining models...")
    # TODO: Compare models

    print("\nResults:")
    # TODO: Print results

    print("\nFeature Analysis (Lasso):")
    # TODO: Print top features from Lasso model
