"""
Explainable AI Module using SHAP.

This module provides model explainability using SHAP (SHapley Additive exPlanations).
SHAP values help understand which features contribute most to individual predictions
and overall model behavior.

IMPORTANT: SHAP explanations are based on the trained model's behavior and
should be interpreted as approximations of feature importance, not definitive causality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
try:
    from config import FEATURE_NAMES, TARGET_NAME
except ImportError:
    FEATURE_NAMES = [
        "CPU_Temperature",
        "CPU_Usage", 
        "RAM_Usage",
        "Disk_Usage",
        "Fan_Speed",
        "Error_Count",
        "Runtime_Hours"
    ]
    TARGET_NAME = "Failure_Risk"


class ModelExplainer:
    """
    Provides model explainability using SHAP values.
    
    This class generates both global explanations (overall feature importance)
    and local explanations (individual prediction explanations).
    """
    
    def __init__(self, model, feature_names: list = None):
        """
        Initialize the model explainer.
        
        Args:
            model: Trained machine learning model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names or FEATURE_NAMES
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, X_background, model_type: str = "auto"):
        """
        Create SHAP explainer based on model type.
        
        Args:
            X_background: Background dataset for SHAP (typically a sample of training data)
            model_type: Type of model ('tree', 'linear', 'auto')
        """
        print("Creating SHAP explainer...")
        
        if model_type == "auto":
            # Auto-detect model type
            if hasattr(self.model, 'estimators_'):
                # Random Forest or similar tree ensemble
                model_type = "tree"
            elif hasattr(self.model, 'coef_'):
                # Linear model
                model_type = "linear"
            else:
                # Default to tree explainer (works for many models)
                model_type = "tree"
        
        if model_type == "tree":
            # TreeExplainer is optimized for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            print("Using TreeExplainer (optimized for tree-based models)")
        elif model_type == "linear":
            # LinearExplainer for linear models
            self.explainer = shap.LinearExplainer(self.model, X_background)
            print("Using LinearExplainer (for linear models)")
        else:
            # KernelExplainer as fallback (model-agnostic but slower)
            self.explainer = shap.KernelExplainer(self.model.predict, X_background)
            print("Using KernelExplainer (model-agnostic)")
    
    def calculate_shap_values(self, X):
        """
        Calculate SHAP values for the given data.
        
        Args:
            X: Data to calculate SHAP values for
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        print("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(X)
        print("SHAP values calculated successfully.")
    
    def plot_global_feature_importance(self, max_display: int = None):
        """
        Plot global feature importance using SHAP values.
        
        This shows which features are most important overall across all predictions.
        
        Args:
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values() first.")
        
        print("\nGenerating global feature importance plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, plot_type="bar", 
                        feature_names=self.feature_names,
                        max_display=max_display, show=False)
        plt.title("Global Feature Importance (SHAP Values)")
        plt.tight_layout()
        plt.show()
        
        print("Global feature importance plot displayed.")
        print("\nInterpretation:")
        print("- Features with higher SHAP values have more impact on predictions")
        print("- This shows overall importance across all predictions")
        print("- Features near the top are most influential")
    
    def plot_summary_plot(self, X, max_display: int = None):
        """
        Plot SHAP summary plot showing feature impact and value relationships.
        
        This shows both feature importance and how feature values affect predictions.
        
        Args:
            X: Data used for SHAP values
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values() first.")
        
        print("\nGenerating SHAP summary plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, 
                        feature_names=self.feature_names,
                        max_display=max_display, show=False)
        plt.title("SHAP Summary Plot: Feature Impact vs Feature Values")
        plt.tight_layout()
        plt.show()
        
        print("SHAP summary plot displayed.")
        print("\nInterpretation:")
        print("- Each dot represents a single prediction")
        print("- Horizontal position: SHAP value (impact on prediction)")
        print("- Color: Feature value (red = high, blue = low)")
        print("- Features on right push prediction toward failure risk")
        print("- Features on left push prediction toward normal")
    
    def explain_single_prediction(self, X_single, feature_values: dict = None):
        """
        Explain a single prediction using SHAP force plot.
        
        Args:
            X_single: Single instance to explain (2D array with shape [1, n_features])
            feature_values: Dictionary of actual feature values for the instance
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        print("\nGenerating single prediction explanation...")
        
        # Calculate SHAP values for single instance
        shap_values_single = self.explainer.shap_values(X_single)
        
        # Create force plot
        plt.figure(figsize=(12, 6))
        shap.force_plot(self.explainer.expected_value[0], shap_values_single[0],
                       X_single[0], feature_names=self.feature_names,
                       matplotlib=True, show=False)
        plt.title("SHAP Force Plot: Single Prediction Explanation")
        plt.tight_layout()
        plt.show()
        
        print("Single prediction explanation displayed.")
        print("\nInterpretation:")
        print("- Base value: Average prediction across training data")
        print("- Red features: Push prediction toward failure risk")
        print("- Blue features: Push prediction toward normal")
        print("- Arrow length: Magnitude of feature's impact")
        
        # Print numerical explanation
        print("\nNumerical Feature Contributions:")
        for i, (feature, shap_val, feature_val) in enumerate(zip(
            self.feature_names, shap_values_single[0], X_single[0]
        )):
            direction = "→ Failure Risk" if shap_val > 0 else "→ Normal"
            print(f"  {feature:20s}: {shap_val:+7.4f} (value: {feature_val:7.2f}) {direction}")
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance as a DataFrame.
        
        Returns:
            DataFrame with feature importance (mean absolute SHAP values)
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values() first.")
        
        # Calculate mean absolute SHAP values for each feature
        mean_shap_values = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_SHAP_Value': mean_shap_values
        }).sort_values('Mean_SHAP_Value', ascending=False)
        
        return importance_df
    
    def explain_in_plain_language(self, X_single, prediction: str, probability: float):
        """
        Provide a plain language explanation of a prediction.
        
        Args:
            X_single: Single instance to explain
            prediction: Model prediction ("Normal" or "Failure Risk")
            probability: Prediction probability
            
        Returns:
            Plain language explanation string
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Calculate SHAP values for single instance
        shap_values_single = self.explainer.shap_values(X_single)
        
        # Get top contributing features
        feature_contributions = list(zip(self.feature_names, shap_values_single[0], X_single[0]))
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Build explanation
        explanation = f"\nPrediction Explanation:\n"
        explanation += f"{'='*50}\n"
        explanation += f"Model Prediction: {prediction}\n"
        explanation += f"Failure Probability: {probability:.1%}\n"
        explanation += f"{'='*50}\n\n"
        
        explanation += "Key Factors Influencing This Prediction:\n"
        
        for i, (feature, shap_val, feature_val) in enumerate(feature_contributions[:5]):
            impact = "increases" if shap_val > 0 else "decreases"
            explanation += f"\n{i+1}. {feature} (value: {feature_val:.2f})\n"
            explanation += f"   This {impact} failure risk (SHAP value: {shap_val:+.4f})\n"
        
        explanation += f"\n{'='*50}\n"
        explanation += "Note: SHAP values show feature contribution to this specific prediction.\n"
        explanation += "Higher absolute values indicate stronger influence.\n"
        
        return explanation


def main():
    """
    Main function to demonstrate explainability workflow.
    """
    print("=" * 70)
    print("EXPLAINABLE AI DEMO")
    print("=" * 70)
    
    # This is a demo - actual usage would be in the Streamlit app
    print("Please use the Streamlit app for interactive explanations.")


if __name__ == "__main__":
    main()