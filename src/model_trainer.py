"""
Machine Learning Model Training Module.

This module handles:
- Training Logistic Regression and Random Forest models
- Model evaluation using multiple metrics
- Model comparison and selection
- Feature importance analysis
- Model saving and loading

IMPORTANT: All training and evaluation uses only training data.
Test set is used only for final evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
try:
    from config import (
        RANDOM_STATE, MODEL_CONFIG, TARGET_NAME, FEATURE_NAMES,
        MODEL_PATH, PREPROCESSOR_PATH, RISK_THRESHOLDS
    )
except ImportError:
    # Fallback if config module not found
    RANDOM_STATE = 42
    MODEL_CONFIG = {
        "logistic_regression": {
            "max_iter": 1000,
            "random_state": RANDOM_STATE
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": RANDOM_STATE
        }
    }
    TARGET_NAME = "Failure_Risk"
    FEATURE_NAMES = [
        "CPU_Temperature",
        "CPU_Usage", 
        "RAM_Usage",
        "Disk_Usage",
        "Fan_Speed",
        "Error_Count",
        "Runtime_Hours"
    ]
    MODEL_PATH = "models/best_model.joblib"
    PREPROCESSOR_PATH = "models/preprocessor.joblib"
    RISK_THRESHOLDS = {"LOW": 0.3, "MEDIUM": 0.7, "HIGH": 1.0}


class ModelTrainer:
    """
    Train and evaluate machine learning models for failure prediction.
    
    This class handles the complete model training pipeline including
    model training, evaluation, comparison, and selection.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
    
    def get_logistic_regression(self) -> LogisticRegression:
        """
        Create and configure Logistic Regression model.
        
        Logistic Regression is a linear classification algorithm that uses
        the sigmoid function to output probabilities between 0 and 1.
        
        The sigmoid function: σ(z) = 1 / (1 + e^(-z))
        - Converts linear output to probability
        - Output of 0.5 is the decision boundary
        - Values > 0.5 predict class 1 (failure risk)
        - Values < 0.5 predict class 0 (normal)
        
        Returns:
            Configured Logistic Regression model
        """
        config = MODEL_CONFIG.get("logistic_regression", {})
        model = LogisticRegression(
            max_iter=config.get("max_iter", 1000),
            random_state=config.get("random_state", self.random_state)
        )
        return model
    
    def get_random_forest(self) -> RandomForestClassifier:
        """
        Create and configure Random Forest model.
        
        Random Forest is an ensemble method that combines multiple decision trees.
        Each tree is trained on a random subset of data and features, then
        predictions are aggregated (voting for classification).
        
        Advantages:
        - Handles non-linear relationships
        - Robust to overfitting (through ensemble averaging)
        - Provides feature importance scores
        - Works well with different feature scales
        
        Returns:
            Configured Random Forest model
        """
        config = MODEL_CONFIG.get("random_forest", {})
        model = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
            min_samples_split=config.get("min_samples_split", 5),
            random_state=config.get("random_state", self.random_state),
            n_jobs=-1  # Use all available cores
        )
        return model
    
    def train_model(self, model, X_train, y_train, model_name: str):
        """
        Train a single model on training data.
        
        Args:
            model: sklearn model instance
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model for tracking
        """
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        self.models[model_name] = model
        print(f"{model_name} trained successfully.")
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """
        Evaluate a model on test data using multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        print(f"\n{model_name} Evaluation Results:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"True Negatives:  {cm[0,0]}  (Normal correctly predicted)")
        print(f"False Positives: {cm[0,1]}  (Normal incorrectly flagged)")
        print(f"False Negatives: {cm[1,0]}  (Failures missed - CRITICAL)")
        print(f"True Positives:  {cm[1,1]}  (Failures correctly caught)")
        
        return metrics
    
    def plot_confusion_matrix(self, cm, model_name: str):
        """
        Plot confusion matrix with annotations.
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Failure Risk'],
                   yticklabels=['Normal', 'Failure Risk'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, model, X_test, y_test, model_name: str):
        """
        Plot ROC curve for the model.
        
        ROC (Receiver Operating Characteristic) curve shows the trade-off
        between true positive rate and false positive rate at different thresholds.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
        """
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend([f'{model_name}', 'Random'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models using evaluation metrics.
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        # Sort by recall if column exists
        if 'Recall' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('Recall', ascending=False)
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(comparison_df.to_string(index=False))
        print("=" * 70)
        
        return comparison_df
    
    def select_best_model(self):
        """
        Select the best model based on evaluation metrics.
        
        For failure prediction, we prioritize recall to minimize false negatives
        (missed failures), while maintaining reasonable precision.
        
        Selection criteria (weighted):
        - 50% Recall (most important - don't miss failures)
        - 30% F1-Score (balance of precision and recall)
        - 20% ROC-AUC (overall discriminative ability)
        """
        print("\n" + "=" * 70)
        print("MODEL SELECTION")
        print("=" * 70)
        
        # Calculate composite score for each model
        for model_name, metrics in self.results.items():
            composite_score = (
                0.5 * metrics['recall'] +
                0.3 * metrics['f1'] +
                0.2 * metrics['roc_auc']
            )
            self.results[model_name]['composite_score'] = composite_score
            print(f"{model_name} - Composite Score: {composite_score:.4f} (Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f})")
        
        # Select model with highest composite score
        if not self.results:
            print("ERROR: No models found in results!")
            return
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nSelected Model: {best_model_name}")
        print(f"Composite Score: {self.results[best_model_name]['composite_score']:.4f}")
        print(f"  Recall:    {self.results[best_model_name]['recall']:.4f}")
        print(f"  F1-Score:  {self.results[best_model_name]['f1']:.4f}")
        print(f"  ROC-AUC:   {self.results[best_model_name]['roc_auc']:.4f}")
        print("=" * 70)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the best model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.best_model is None:
            raise ValueError("No model selected. Call select_best_model() first.")
        
        # Get feature importance based on model type
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models (Random Forest)
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # Linear models (Logistic Regression)
            importance = np.abs(self.best_model.coef_[0])
        else:
            raise ValueError("Model does not support feature importance.")
        
        importance_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE")
        print("=" * 70)
        print(importance_df.to_string(index=False))
        print("=" * 70)
        
        return importance_df
    
    def plot_feature_importance(self, importance_df: pd.DataFrame):
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Feature Importance - {self.best_model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path: str = MODEL_PATH, preprocessor=None, 
                   preprocessor_path: str = PREPROCESSOR_PATH):
        """
        Save the best model and preprocessor.
        
        Args:
            model_path: Path to save the model
            preprocessor: Fitted preprocessor pipeline
            preprocessor_path: Path to save the preprocessor
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save preprocessor if provided
        if preprocessor is not None:
            preprocessor_path = Path(preprocessor_path)
            preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, preprocessor_path)
            print(f"Preprocessor saved to: {preprocessor_path}")
            self.preprocessor = preprocessor
    
    def load_model(self, model_path: str = MODEL_PATH, 
                   preprocessor_path: str = PREPROCESSOR_PATH):
        """
        Load a saved model and preprocessor.
        
        Args:
            model_path: Path to the saved model
            preprocessor_path: Path to the saved preprocessor
        """
        model_path = Path(model_path)
        self.best_model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        if preprocessor_path and Path(preprocessor_path).exists():
            preprocessor_path = Path(preprocessor_path)
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"Preprocessor loaded from: {preprocessor_path}")
    
    def predict(self, X):
        """
        Make predictions using the best model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (predicted_class, probability, risk_category)
        """
        if self.best_model is None:
            raise ValueError("No model loaded. Call load_model() or train a model first.")
        
        # Apply preprocessor if available
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        
        # Make predictions
        y_pred = self.best_model.predict(X)
        y_pred_proba = self.best_model.predict_proba(X)[:, 1]
        
        # Determine risk category
        risk_categories = []
        for prob in y_pred_proba:
            if prob < RISK_THRESHOLDS["LOW"]:
                risk_categories.append("LOW")
            elif prob < RISK_THRESHOLDS["MEDIUM"]:
                risk_categories.append("MEDIUM")
            else:
                risk_categories.append("HIGH")
        
        return y_pred, y_pred_proba, risk_categories


def main():
    """
    Main function to demonstrate model training workflow.
    """
    print("=" * 70)
    print("MODEL TRAINING DEMO")
    print("=" * 70)
    
    # This is a demo - actual training would be done in train.py
    print("Please use train.py for the complete training pipeline.")


if __name__ == "__main__":
    main()