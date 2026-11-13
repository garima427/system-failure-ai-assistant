"""
Data Preprocessing and Exploratory Data Analysis (EDA) Module.

This module handles:
- Data inspection and validation
- Data cleaning
- Exploratory data analysis with visualizations
- Feature preprocessing pipelines
- Train/test split with no data leakage

IMPORTANT: All preprocessing statistics are learned from training data only.
Test data is never used for fitting any transformations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
try:
    from config import (
        RANDOM_STATE, TEST_SIZE, FEATURE_NAMES, TARGET_NAME, 
        EDA_OUTPUT_DIR, FEATURE_DESCRIPTIONS
    )
except ImportError:
    # Fallback if config module not found
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
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
    EDA_OUTPUT_DIR = "eda_output"
    FEATURE_DESCRIPTIONS = {}


class DataPreprocessor:
    """
    Handles data preprocessing and EDA operations.
    
    This class provides methods for data inspection, cleaning, visualization,
    and creating preprocessing pipelines while preventing data leakage.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the data preprocessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = None
        self.preprocessor = None
    
    def inspect_data(self, df: pd.DataFrame) -> dict:
        """
        Perform comprehensive data inspection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing inspection results
        """
        print("=" * 70)
        print("DATA INSPECTION")
        print("=" * 70)
        
        inspection_results = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        print(f"\nDataset Shape: {inspection_results['shape']}")
        print(f"Columns: {len(inspection_results['columns'])}")
        print(f"Memory Usage: {inspection_results['memory_usage']:.2f} MB")
        
        print("\nData Types:")
        for col, dtype in inspection_results['dtypes'].items():
            print(f"  {col}: {dtype}")
        
        print("\nMissing Values:")
        total_missing = sum(inspection_results['missing_values'].values())
        print(f"  Total: {total_missing}")
        for col, count in inspection_results['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count}")
        
        print(f"\nDuplicate Rows: {inspection_results['duplicates']}")
        
        return inspection_results
    
    def analyze_class_distribution(self, df: pd.DataFrame):
        """
        Analyze and visualize class distribution.
        
        Args:
            df: Input DataFrame
        """
        print("\n" + "=" * 70)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("=" * 70)
        
        class_counts = df[TARGET_NAME].value_counts()
        class_percentages = df[TARGET_NAME].value_counts(normalize=True) * 100
        
        print(f"\nClass Counts:")
        for label, count in class_counts.items():
            label_name = "Normal" if label == 0 else "Failure Risk"
            print(f"  {label_name} ({label}): {count}")
        
        print(f"\nClass Percentages:")
        for label, percentage in class_percentages.items():
            label_name = "Normal" if label == 0 else "Failure Risk"
            print(f"  {label_name} ({label}): {percentage:.2f}%")
        
        # Visualize class distribution
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(x=TARGET_NAME, data=df)
        plt.title('Class Distribution (0=Normal, 1=Failure Risk)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Add count labels on bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        # Save if output directory exists
        if EDA_OUTPUT_DIR:
            output_path = Path(EDA_OUTPUT_DIR)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / 'class_distribution.png', dpi=150, bbox_inches='tight')
            print(f"\nClass distribution plot saved to: {output_path / 'class_distribution.png'}")
        
        plt.show()
    
    def analyze_feature_distributions(self, df: pd.DataFrame):
        """
        Analyze and visualize feature distributions.
        
        Args:
            df: Input DataFrame
        """
        print("\n" + "=" * 70)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("=" * 70)
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(df[FEATURE_NAMES].describe().round(2))
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(FEATURE_NAMES):
            if i < len(axes):
                # Plot histogram for normal and failure classes
                for label in [0, 1]:
                    data = df[df[TARGET_NAME] == label][feature]
                    label_name = "Normal" if label == 0 else "Failure Risk"
                    axes[i].hist(data, bins=30, alpha=0.5, label=label_name)
                
                axes[i].set_title(feature)
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
        
        # Remove empty subplot if odd number of features
        if len(FEATURE_NAMES) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        # Save if output directory exists
        if EDA_OUTPUT_DIR:
            output_path = Path(EDA_OUTPUT_DIR)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / 'feature_distributions.png', dpi=150, bbox_inches='tight')
            print(f"\nFeature distributions saved to: {output_path / 'feature_distributions.png'}")
        
        plt.show()
    
    def analyze_correlations(self, df: pd.DataFrame):
        """
        Analyze and visualize feature correlations.
        
        Args:
            df: Input DataFrame
        """
        print("\n" + "=" * 70)
        print("CORRELATION ANALYSIS")
        print("=" * 70)
        
        # Calculate correlation matrix
        corr_matrix = df[FEATURE_NAMES].corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'},
                   square=True)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # Save if output directory exists
        if EDA_OUTPUT_DIR:
            output_path = Path(EDA_OUTPUT_DIR)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
            print(f"\nCorrelation heatmap saved to: {output_path / 'correlation_heatmap.png'}")
        
        plt.show()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by removing duplicates and handling missing values.
        
        IMPORTANT: This should be done before train/test split.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("\n" + "=" * 70)
        print("DATA CLEANING")
        print("=" * 70)
        
        initial_shape = df.shape
        
        # Remove duplicate rows
        duplicates_before = df.duplicated().sum()
        df_clean = df.drop_duplicates()
        duplicates_removed = duplicates_before - df_clean.duplicated().sum()
        
        print(f"Duplicates removed: {duplicates_removed}")
        
        # Handle missing values with median imputation
        missing_before = df_clean.isnull().sum().sum()
        for col in FEATURE_NAMES:
            if col in df_clean.columns and df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"Imputed {col} with median: {median_val:.2f}")
        
        missing_after = df_clean.isnull().sum().sum()
        
        print(f"\nData shape before cleaning: {initial_shape}")
        print(f"Data shape after cleaning: {df_clean.shape}")
        print(f"Missing values before: {missing_before}")
        print(f"Missing values after: {missing_after}")
        
        return df_clean
    
    def split_data(self, df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple:
        """
        Split data into train and test sets.
        
        IMPORTANT: This split happens BEFORE any preprocessing.
        Test set remains completely unseen during preprocessing.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing (0-1)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\n" + "=" * 70)
        print("TRAIN/TEST SPLIT")
        print("=" * 70)
        
        # Separate features and target
        X = df.drop(columns=[TARGET_NAME])
        y = df[TARGET_NAME]
        
        # Perform stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain set size: {len(X_train)} samples ({(1-test_size)*100:.1f}%)")
        print(f"Test set size: {len(X_test)} samples ({test_size*100:.1f}%)")
        
        print(f"\nTrain class distribution:")
        print(y_train.value_counts(normalize=True).round(3) * 100)
        
        print(f"\nTest class distribution:")
        print(y_test.value_counts(normalize=True).round(3) * 100)
        
        return X_train, X_test, y_train, y_test
    
    def create_preprocessing_pipeline(self) -> Pipeline:
        """
        Create preprocessing pipeline for numerical features.
        
        The pipeline includes:
        1. Median imputation for missing values
        2. Standard scaling for normalization
        
        IMPORTANT: This pipeline will be FIT only on training data.
        Test data will only be TRANSFORMED using fitted parameters.
        
        Returns:
            Fitted preprocessing pipeline
        """
        print("\n" + "=" * 70)
        print("PREPROCESSING PIPELINE")
        print("=" * 70)
        
        # Create preprocessing steps
        preprocessing_steps = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', preprocessing_steps, FEATURE_NAMES)
            ],
            remainder='passthrough'
        )
        
        print("Preprocessing pipeline created with:")
        print("  1. Median imputation for missing values")
        print("  2. Standard scaling (z-score normalization)")
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit_preprocessor(self, X_train):
        """
        Fit the preprocessing pipeline on training data only.
        
        Args:
            X_train: Training features
            
        Returns:
            Transformed training features
        """
        print("\nFitting preprocessor on training data...")
        X_train_processed = self.preprocessor.fit_transform(X_train)
        print(f"Training data shape after preprocessing: {X_train_processed.shape}")
        
        return X_train_processed
    
    def transform_data(self, X):
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_preprocessor() first.")
        
        X_processed = self.preprocessor.transform(X)
        return X_processed


def main():
    """
    Main function to demonstrate data preprocessing workflow.
    """
    print("=" * 70)
    print("DATA PREPROCESSING AND EDA DEMO")
    print("=" * 70)
    
    # Load data (assuming it exists)
    try:
        from config import DATA_PATH
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded from: {DATA_PATH}")
    except:
        print("Please generate data first using data_generator.py")
        return
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Perform EDA
    preprocessor.inspect_data(df)
    preprocessor.analyze_class_distribution(df)
    preprocessor.analyze_feature_distributions(df)
    preprocessor.analyze_correlations(df)
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_clean)
    
    # Create and fit preprocessing pipeline
    pipeline = preprocessor.create_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_preprocessor(X_train)
    X_test_processed = preprocessor.transform_data(X_test)
    
    print("\n" + "=" * 70)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()