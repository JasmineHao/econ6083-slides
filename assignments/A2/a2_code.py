"""
ECON6083 - Assignment 2: Income Prediction with Tree-Based Models
Student Template

Instructions:
1. Complete all functions marked with TODO
2. Do not change function names or signatures
3. You may add helper functions if needed
4. Test your code before submission

Submission:
- Rename this file to: a2_code_<StudentID>_<LastName>.py
- Submit to Moodle along with a2_report.md
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Uncomment if using XGBoost or LightGBM
# import xgboost as xgb
# import lightgbm as lgb


class IncomeClassifier:
    """
    A classifier for predicting income levels using tree-based models.

    This class implements Decision Tree, Random Forest, and Gradient Boosting
    classifiers for the UCI Adult Income dataset.
    """

    def __init__(self, random_state=42):
        """
        Initialize the classifier.

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}  # Dictionary to store trained models
        self.feature_names = []  # List to store feature names
        self.label_encoders = {}  # Dictionary to store label encoders

    def load_data(self):
        """
        Load the UCI Adult Income dataset.

        Returns:
        --------
        df : pandas.DataFrame
            The loaded dataset
        """
        # TODO: Load data from UCI repository
        # URL: "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                   'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                   'income']

        # TODO: Read the CSV file
        # Hint: Use pd.read_csv with na_values=' ?' to handle missing values
        # Hint: Use skipinitialspace=True to remove leading spaces
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

        # YOUR CODE HERE
        df = None  # Replace with your code

        return df

    def _preprocess_data(self, df):
        """
        Preprocess the dataset: handle missing values, encode categoricals,
        engineer features.

        Parameters:
        -----------
        df : pandas.DataFrame
            Raw dataframe

        Returns:
        --------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target variable (binary: 0 for <=50K, 1 for >50K)
        """
        df = df.copy()

        # TODO: 1. Handle missing values
        # Hint: Missing values are marked as ' ?' (with space)
        # Replace ' ?' with np.nan, then fill with mode or drop

        # YOUR CODE HERE


        # TODO: 2. Create target variable (binary classification)
        # Encode income: <=50K -> 0, >50K -> 1

        # YOUR CODE HERE
        y = None  # Replace with your code


        # TODO: 3. Feature Engineering
        # Create at least 3 new features:
        # - capital_gain_indicator: binary (1 if capital-gain > 0, else 0)
        # - hours_category: 'part-time' (<35), 'full-time' (35-45), 'overtime' (>45)
        # - age_group: 'young' (<30), 'middle' (30-50), 'senior' (>50)

        # YOUR CODE HERE


        # TODO: 4. Drop redundant features
        # Drop: fnlwgt (sampling weight, not predictive)
        # Drop either 'education' or 'education-num' (redundant)

        # YOUR CODE HERE


        # TODO: 5. Encode categorical variables
        # Options:
        #   A. Use LabelEncoder for binary features (sex)
        #   B. Use OneHotEncoder or pd.get_dummies for multi-class features
        #   C. Use LabelEncoder for all (simpler but less accurate)

        # Hint: Store label encoders in self.label_encoders for later use

        # YOUR CODE HERE


        # TODO: 6. Separate features and target
        # Make sure to drop 'income' column from X

        X = None  # Replace with your code

        # Store feature names for later analysis
        self.feature_names = X.columns.tolist()

        return X, y

    def split_data(self, X, y, test_size=0.2, val_size=0.0):
        """
        Split data into train, validation (optional), and test sets.

        Parameters:
        -----------
        X : pandas.DataFrame
            Features
        y : pandas.Series
            Target
        test_size : float
            Proportion of data for test set
        val_size : float
            Proportion of data for validation set (0 = no validation set)

        Returns:
        --------
        If val_size > 0: X_train, X_val, X_test, y_train, y_val, y_test
        If val_size = 0: X_train, X_test, y_train, y_test
        """
        # TODO: Split the data
        # If val_size > 0, do a three-way split (train/val/test)
        # Otherwise, do a two-way split (train/test)

        # Hint: Use train_test_split twice for three-way split
        # First split: separate test set
        # Second split: split remaining data into train and val

        # YOUR CODE HERE

        if val_size > 0:
            # Three-way split
            X_train = None
            X_val = None
            X_test = None
            y_train = None
            y_val = None
            y_test = None
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            # Two-way split
            X_train = None
            X_test = None
            y_train = None
            y_test = None
            return X_train, X_test, y_train, y_test

    def train_decision_tree(self, X_train, y_train):
        """
        Train a Decision Tree classifier with hyperparameter tuning.

        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Training labels

        Returns:
        --------
        best_model : DecisionTreeClassifier
            The best model from GridSearchCV
        """
        # TODO: Define parameter grid for GridSearchCV
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_split': [10, 20, 50],
            'min_samples_leaf': [5, 10, 20]
        }

        # TODO: Create DecisionTreeClassifier
        # Hint: Use class_weight='balanced' to handle class imbalance

        dt = None  # YOUR CODE HERE

        # TODO: Create GridSearchCV
        # Use 5-fold cross-validation
        # Use scoring='f1' (better than accuracy for imbalanced data)

        grid_search = None  # YOUR CODE HERE

        # TODO: Fit the grid search
        # YOUR CODE HERE

        # Store the best model
        best_model = grid_search.best_estimator_
        self.models['decision_tree'] = best_model

        print(f"Decision Tree - Best parameters: {grid_search.best_params_}")
        print(f"Decision Tree - Best CV F1-score: {grid_search.best_score_:.4f}")

        return best_model

    def train_random_forest(self, X_train, y_train):
        """
        Train a Random Forest classifier with hyperparameter tuning.

        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Training labels

        Returns:
        --------
        best_model : RandomForestClassifier
            The best model from GridSearchCV
        """
        # TODO: Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10]
        }

        # TODO: Create RandomForestClassifier
        # Hint: Use class_weight='balanced' and n_jobs=-1 for speed

        rf = None  # YOUR CODE HERE

        # TODO: Create GridSearchCV
        # Use 5-fold CV, scoring='f1'

        grid_search = None  # YOUR CODE HERE

        # TODO: Fit the grid search
        # YOUR CODE HERE

        # Store the best model
        best_model = grid_search.best_estimator_
        self.models['random_forest'] = best_model

        print(f"Random Forest - Best parameters: {grid_search.best_params_}")
        print(f"Random Forest - Best CV F1-score: {grid_search.best_score_:.4f}")

        return best_model

    def train_gradient_boosting(self, X_train, y_train, use_xgboost=True):
        """
        Train a Gradient Boosting classifier with hyperparameter tuning.

        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Training labels
        use_xgboost : bool
            If True, use XGBoost; otherwise use sklearn GradientBoostingClassifier

        Returns:
        --------
        best_model : GradientBoostingClassifier or XGBClassifier
            The best model from GridSearchCV
        """
        if use_xgboost:
            # Option 1: XGBoost (recommended)
            # TODO: Import xgboost and create XGBClassifier
            # Uncomment the import at the top of the file

            # TODO: Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }

            # TODO: Create XGBClassifier
            # Hint: Use scale_pos_weight to handle imbalance
            # scale_pos_weight = (count of class 0) / (count of class 1)

            # YOUR CODE HERE
            model = None  # Replace with XGBClassifier

        else:
            # Option 2: Sklearn GradientBoostingClassifier
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }

            # TODO: Create GradientBoostingClassifier
            # YOUR CODE HERE
            model = None  # Replace with GradientBoostingClassifier

        # TODO: Create GridSearchCV
        grid_search = None  # YOUR CODE HERE

        # TODO: Fit the grid search
        # YOUR CODE HERE

        # Store the best model
        best_model = grid_search.best_estimator_
        self.models['gradient_boosting'] = best_model

        print(f"Gradient Boosting - Best parameters: {grid_search.best_params_}")
        print(f"Gradient Boosting - Best CV F1-score: {grid_search.best_score_:.4f}")

        return best_model

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on the test set.

        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            Test labels

        Returns:
        --------
        results : dict
            Dictionary with metrics for each model
        """
        results = {}

        for model_name, model in self.models.items():
            # TODO: Make predictions
            y_pred = None  # YOUR CODE HERE
            y_pred_proba = None  # YOUR CODE HERE (probability for AUC)

            # TODO: Calculate metrics
            # Calculate: accuracy, precision, recall, f1, auc

            accuracy = None  # YOUR CODE HERE
            precision = None  # YOUR CODE HERE
            recall = None  # YOUR CODE HERE
            f1 = None  # YOUR CODE HERE
            auc = None  # YOUR CODE HERE

            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            # Print classification report
            print(f"\n{'='*60}")
            print(f"{model_name.upper()} - Classification Report")
            print(f"{'='*60}")
            print(classification_report(y_test, y_pred))

            # Print confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(cm)

        # Create comparison table
        self._print_comparison_table(results)

        return results

    def _print_comparison_table(self, results):
        """Print a comparison table of all models."""
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
        print(f"{'-'*80}")

        for model_name, metrics in results.items():
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1']:<12.4f} "
                  f"{metrics['auc']:<12.4f}")
        print(f"{'='*80}\n")

    def get_feature_importance(self, model_name='random_forest', top_n=10):
        """
        Extract and display feature importance from a tree-based model.

        Parameters:
        -----------
        model_name : str
            Name of the model ('random_forest' or 'gradient_boosting')
        top_n : int
            Number of top features to display

        Returns:
        --------
        importance_df : pandas.DataFrame
            DataFrame with features and their importance scores
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found. Train the model first.")
            return None

        model = self.models[model_name]

        # TODO: Extract feature importance
        # Hint: Use model.feature_importances_

        # YOUR CODE HERE
        importances = None  # Replace with feature importances

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Print top N features
        print(f"\nTop {top_n} Most Important Features ({model_name}):")
        print(importance_df.head(top_n))

        return importance_df

    def plot_roc_curves(self, X_test, y_test, results):
        """
        Plot ROC curves for all models.

        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            Test labels
        results : dict
            Results dictionary from evaluate_models()
        """
        plt.figure(figsize=(10, 8))

        for model_name, metrics in results.items():
            # TODO: Calculate ROC curve
            # Hint: Use roc_curve(y_test, y_pred_proba)

            # YOUR CODE HERE
            fpr = None
            tpr = None

            # Plot
            plt.plot(fpr, tpr, linewidth=2,
                    label=f"{model_name} (AUC = {metrics['auc']:.3f})")

        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')

        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("\nROC curves saved to 'roc_curves.png'")
        plt.show()

    def analyze_fairness(self, X_test, y_test, sensitive_feature='sex'):
        """
        Analyze fairness metrics across different demographic groups.

        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features (must include sensitive_feature column)
        y_test : pandas.Series
            Test labels
        sensitive_feature : str
            Name of the sensitive feature (e.g., 'sex', 'race')
        """
        if sensitive_feature not in X_test.columns:
            print(f"Feature '{sensitive_feature}' not found in dataset.")
            return

        print(f"\n{'='*60}")
        print(f"FAIRNESS ANALYSIS - {sensitive_feature.upper()}")
        print(f"{'='*60}")

        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)

            print(f"\n{model_name.upper()}:")
            print(f"{'-'*60}")

            # TODO: Calculate fairness metrics for each group
            # For each unique value in sensitive_feature:
            #   - Calculate False Positive Rate (FPR)
            #   - Calculate False Negative Rate (FNR)
            #   - Calculate F1-score

            # YOUR CODE HERE
            # Hint: Use confusion_matrix to get TN, FP, FN, TP
            # FPR = FP / (FP + TN)
            # FNR = FN / (FN + TP)

            print(f"{'Group':<15} {'FPR':<10} {'FNR':<10} {'F1':<10}")
            print(f"{'-'*45}")

            # YOUR CODE HERE
            # Loop through groups and print metrics

    def run_full_pipeline(self):
        """
        Run the complete analysis pipeline.
        """
        print("="*80)
        print("ECON6083 - Assignment 2: Income Prediction with Tree-Based Models")
        print("="*80)

        # Step 1: Load data
        print("\n[1/6] Loading data...")
        df = self.load_data()
        print(f"Dataset shape: {df.shape}")

        # Step 2: Preprocess
        print("\n[2/6] Preprocessing data...")
        X, y = self._preprocess_data(df)
        print(f"Features shape: {X.shape}")
        print(f"Class distribution:\n{y.value_counts(normalize=True)}")

        # Step 3: Split data
        print("\n[3/6] Splitting data...")
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=0.2)
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

        # Step 4: Train models
        print("\n[4/6] Training models...")
        print("\n--- Decision Tree ---")
        self.train_decision_tree(X_train, y_train)

        print("\n--- Random Forest ---")
        self.train_random_forest(X_train, y_train)

        print("\n--- Gradient Boosting ---")
        self.train_gradient_boosting(X_train, y_train, use_xgboost=True)

        # Step 5: Evaluate
        print("\n[5/6] Evaluating models...")
        results = self.evaluate_models(X_test, y_test)

        # Step 6: Analysis
        print("\n[6/6] Additional analysis...")

        # Feature importance
        self.get_feature_importance('random_forest', top_n=10)

        # ROC curves
        self.plot_roc_curves(X_test, y_test, results)

        # Fairness analysis (optional, uncomment if X_test has 'sex' feature)
        # self.analyze_fairness(X_test, y_test, sensitive_feature='sex')

        print("\n" + "="*80)
        print("Analysis complete! Check 'roc_curves.png' for visualizations.")
        print("="*80)


# Main execution
if __name__ == "__main__":
    # Create classifier instance
    classifier = IncomeClassifier(random_state=42)

    # Run full pipeline
    classifier.run_full_pipeline()

    print("\nAll models trained successfully!")
    print("Next steps:")
    print("1. Review the output above")
    print("2. Complete the a2_report.md file")
    print("3. Submit both files to Moodle")
