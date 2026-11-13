"""
Main Training Pipeline for System Failure Prediction AI Assistant.

This script orchestrates the complete machine learning pipeline:
1. Generate synthetic telemetry data
2. Data preprocessing and EDA
3. Train and evaluate ML models
4. Select best model
5. Save model and preprocessor

IMPORTANT: Test set remains completely unseen until final evaluation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_generator import SystemTelemetryGenerator
from src.data_preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer

# Import configuration
try:
    from config import (
        DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH, EDA_OUTPUT_DIR,
        RANDOM_STATE, FEATURE_NAMES, TARGET_NAME
    )
except ImportError:
    DATA_PATH = "data/system_telemetry.csv"
    MODEL_PATH = "models/best_model.joblib"
    PREPROCESSOR_PATH = "models/preprocessor.joblib"
    EDA_OUTPUT_DIR = "eda_output"
    RANDOM_STATE = 42
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


def main():
    """
    Run the complete training pipeline.
    """
    print("=" * 70)
    print("SYSTEM FAILURE PREDICTION AI ASSISTANT - TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1/6] Generating synthetic telemetry data...")
    generator = SystemTelemetryGenerator(random_state=RANDOM_STATE)
    
    # Check if data already exists
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        df = generator.generate_dataset()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to: {data_path}")
    else:
        print(f"Loading existing dataset from: {data_path}")
        df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Failure rate: {df[TARGET_NAME].mean():.2%}")
    
    # Step 2: Data preprocessing and EDA
    print("\n[Step 2/6] Performing data preprocessing and EDA...")
    preprocessor = DataPreprocessor(random_state=RANDOM_STATE)
    
    # Data inspection
    preprocessor.inspect_data(df)
    
    # Class distribution analysis
    preprocessor.analyze_class_distribution(df)
    
    # Feature distribution analysis
    preprocessor.analyze_feature_distributions(df)
    
    # Correlation analysis
    preprocessor.analyze_correlations(df)
    
    # Data cleaning
    df_clean = preprocessor.clean_data(df)
    
    # Step 3: Train/test split
    print("\n[Step 3/6] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_clean)
    
    # Step 4: Create and fit preprocessing pipeline
    print("\n[Step 4/6] Creating and fitting preprocessing pipeline...")
    pipeline = preprocessor.create_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_preprocessor(X_train)
    X_test_processed = preprocessor.transform_data(X_test)
    
    # Step 5: Train and evaluate models
    print("\n[Step 5/6] Training and evaluating models...")
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    
    # Train Logistic Regression
    logistic_model = trainer.get_logistic_regression()
    trainer.train_model(logistic_model, X_train_processed, y_train, "Logistic Regression")
    logistic_metrics = trainer.evaluate_model(
        logistic_model, X_test_processed, y_test, "Logistic Regression"
    )
    
    # Train Random Forest
    rf_model = trainer.get_random_forest()
    trainer.train_model(rf_model, X_train_processed, y_train, "Random Forest")
    rf_metrics = trainer.evaluate_model(
        rf_model, X_test_processed, y_test, "Random Forest"
    )
    
    # Plot confusion matrices
    trainer.plot_confusion_matrix(logistic_metrics['confusion_matrix'], "Logistic Regression")
    trainer.plot_confusion_matrix(rf_metrics['confusion_matrix'], "Random Forest")
    
    # Plot ROC curves
    trainer.plot_roc_curve(logistic_model, X_test_processed, y_test, "Logistic Regression")
    trainer.plot_roc_curve(rf_model, X_test_processed, y_test, "Random Forest")
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Step 6: Select best model and save
    print("\n[Step 6/6] Selecting best model and saving...")
    trainer.select_best_model()
    
    # Get feature importance
    importance_df = trainer.get_feature_importance()
    trainer.plot_feature_importance(importance_df)
    
    # Save model and preprocessor
    trainer.save_model(MODEL_PATH, preprocessor.preprocessor, PREPROCESSOR_PATH)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")
    
    print(f"\nFinal Test Set Performance ({trainer.best_model_name}):")
    best_metrics = trainer.results[trainer.best_model_name]
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  F1-Score:  {best_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    
    print(f"\nEDA visualizations saved to: {EDA_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()