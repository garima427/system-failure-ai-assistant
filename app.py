"""
Streamlit Dashboard for System Failure Prediction AI Assistant.

This dashboard provides an interactive interface for:
- System health prediction using ML models
- Model performance visualization
- Explainable AI (SHAP) analysis
- LLM-powered diagnostic assistance

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.explainability import ModelExplainer
from src.llm_assistant import LLMDiagnosticAssistant

# Import configuration
try:
    from config import (
        DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH, EDA_OUTPUT_DIR,
        FEATURE_NAMES, FEATURE_DESCRIPTIONS, TARGET_NAME,
        RISK_THRESHOLDS, MODEL_CONFIG
    )
except ImportError:
    # Fallback configuration
    DATA_PATH = "data/system_telemetry.csv"
    MODEL_PATH = "models/best_model.joblib"
    PREPROCESSOR_PATH = "models/preprocessor.joblib"
    EDA_OUTPUT_DIR = "eda_output"
    FEATURE_NAMES = [
        "CPU_Temperature",
        "CPU_Usage", 
        "RAM_Usage",
        "Disk_Usage",
        "Fan_Speed",
        "Error_Count",
        "Runtime_Hours"
    ]
    FEATURE_DESCRIPTIONS = {}
    TARGET_NAME = "Failure_Risk"
    RISK_THRESHOLDS = {"LOW": 0.3, "MEDIUM": 0.7, "HIGH": 1.0}
    MODEL_CONFIG = {}


# Page configuration
st.set_page_config(
    page_title="System Failure Prediction AI Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_preprocessor():
    """
    Load the trained model and preprocessor.
    
    Returns:
        Tuple of (model, preprocessor) or (None, None) if loading fails
    """
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data
def load_data():
    """
    Load the dataset.
    
    Returns:
        DataFrame or None if loading fails
    """
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def get_llm_assistant():
    """
    Get LLM assistant instance.
    
    Returns:
        LLMDiagnosticAssistant instance
    """
    return LLMDiagnosticAssistant()


def get_risk_category(probability: float) -> tuple:
    """
    Determine risk category based on probability.
    
    Args:
        probability: Failure probability (0-1)
        
    Returns:
        Tuple of (category, color)
    """
    if probability < RISK_THRESHOLDS["LOW"]:
        return "LOW RISK", "green"
    elif probability < RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM RISK", "orange"
    else:
        return "HIGH RISK", "red"


def main():
    """Main Streamlit application."""
    
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor()
    
    if model is None or preprocessor is None:
        st.error("Model or preprocessor not found. Please run train.py first.")
        st.stop()
    
    # Initialize LLM assistant
    llm_assistant = get_llm_assistant()
    llm_available = llm_assistant.is_available()
    
    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "System Health Prediction", "Model Performance", 
         "Explainability", "Diagnostic Assistant"]
    )
    
    # Page 1: Overview
    if page == "Overview":
        st.markdown('<h1 class="main-header">System Failure Prediction AI Assistant</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        # Project introduction
        st.markdown('<h2 class="section-header">Project Overview</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Objective:</strong> Predict potential system failure risk using system telemetry data 
        and provide understandable explanations and troubleshooting suggestions using Explainable AI and Generative AI.
        </div>
        """, unsafe_allow_html=True)
        
        # Model information
        st.markdown('<h2 class="section-header">Model Information</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Machine Learning Models:**")
            st.markdown("- Logistic Regression (Baseline)")
            st.markdown("- Random Forest Classifier")
            st.markdown("- Model selection based on composite score")
        
        with col2:
            st.markdown("**AI Components:**")
            st.markdown("- SHAP for explainable AI")
            st.markdown("- LLM for diagnostic assistance")
            st.markdown("- Feature importance analysis")
        
        # Dataset summary
        st.markdown('<h2 class="section-header">Dataset Summary</h2>', 
                   unsafe_allow_html=True)
        
        df = load_data()
        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", len(FEATURE_NAMES))
            with col3:
                failure_rate = df[TARGET_NAME].mean() * 100
                st.metric("Failure Rate", f"{failure_rate:.1f}%")
            
            st.markdown("**Features Used:**")
            for feature in FEATURE_NAMES:
                description = FEATURE_DESCRIPTIONS.get(feature, "")
                st.markdown(f"- **{feature}**: {description}")
        
        # Important disclaimer
        st.markdown('<h2 class="section-header">Important Disclaimer</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ This is a demonstration project using synthetic data.</strong><br><br>
        - The dataset is simulated and does not represent actual hardware telemetry<br>
        - Predictions are for educational purposes only<br>
        - Do not use this system for real-world critical infrastructure<br>
        - Always consult with qualified IT professionals for actual system diagnostics
        </div>
        """, unsafe_allow_html=True)
    
    # Page 2: System Health Prediction
    elif page == "System Health Prediction":
        st.markdown('<h1 class="main-header">System Health Prediction</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("Enter the current system telemetry values below:")
        
        # Create input form with sliders
        col1, col2 = st.columns(2)
        
        with col1:
            cpu_temp = st.slider(
                "CPU Temperature (°C)", 
                min_value=30, max_value=95, value=50,
                help=FEATURE_DESCRIPTIONS.get("CPU_Temperature", "")
            )
            cpu_usage = st.slider(
                "CPU Usage (%)", 
                min_value=0, max_value=100, value=40,
                help=FEATURE_DESCRIPTIONS.get("CPU_Usage", "")
            )
            ram_usage = st.slider(
                "RAM Usage (%)", 
                min_value=0, max_value=100, value=45,
                help=FEATURE_DESCRIPTIONS.get("RAM_Usage", "")
            )
            disk_usage = st.slider(
                "Disk Usage (%)", 
                min_value=0, max_value=100, value=50,
                help=FEATURE_DESCRIPTIONS.get("Disk_Usage", "")
            )
        
        with col2:
            fan_speed = st.slider(
                "Fan Speed (RPM)", 
                min_value=0, max_value=5000, value=2000,
                help=FEATURE_DESCRIPTIONS.get("Fan_Speed", "")
            )
            error_count = st.slider(
                "Error Count", 
                min_value=0, max_value=50, value=2,
                help=FEATURE_DESCRIPTIONS.get("Error_Count", "")
            )
            runtime_hours = st.slider(
                "Runtime Hours", 
                min_value=0, max_value=2000, value=300,
                help=FEATURE_DESCRIPTIONS.get("Runtime_Hours", "")
            )
        
        # Predict button
        if st.button("🔍 Predict System Health", type="primary", use_container_width=True):
            # Create input DataFrame
            input_data = pd.DataFrame([{
                "CPU_Temperature": cpu_temp,
                "CPU_Usage": cpu_usage,
                "RAM_Usage": ram_usage,
                "Disk_Usage": disk_usage,
                "Fan_Speed": fan_speed,
                "Error_Count": error_count,
                "Runtime_Hours": runtime_hours
            }])
            
            # Preprocess
            input_processed = preprocessor.transform(input_data)
            
            # Predict
            prediction, probability, risk_categories = model.predict(input_processed)
            prediction_label = "Failure Risk" if prediction[0] == 1 else "Normal"
            risk_category, risk_color = get_risk_category(probability[0])
            
            # Display results
            st.markdown("---")
            st.markdown('<h2 class="section-header">Prediction Results</h2>', 
                       unsafe_allow_html=True)
            
            # Main prediction display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction[0] == 1:
                    st.markdown(f'<div class="danger-box"><strong>PREDICTED CLASS</strong><br><h3>⚠️ FAILURE RISK</h3></div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="success-box"><strong>PREDICTED CLASS</strong><br><h3>✅ NORMAL</h3></div>', 
                               unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "Failure Probability",
                    f"{probability[0]:.1%}",
                    delta=None
                )
            
            with col3:
                st.markdown(f'<div style="color: {risk_color}; font-size: 2rem; font-weight: bold; text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">{risk_category}</div>', 
                           unsafe_allow_html=True)
            
            # Telemetry summary
            st.markdown('<h2 class="section-header">Current Telemetry Summary</h2>', 
                       unsafe_allow_html=True)
            
            telemetry_df = pd.DataFrame([{
                "CPU Temperature": f"{cpu_temp}°C",
                "CPU Usage": f"{cpu_usage}%",
                "RAM Usage": f"{ram_usage}%",
                "Disk Usage": f"{disk_usage}%",
                "Fan Speed": f"{fan_speed} RPM",
                "Error Count": str(error_count),
                "Runtime Hours": f"{runtime_hours}h"
            }])
            
            st.table(telemetry_df.T.rename(columns={0: "Value"}))
    
    # Page 3: Model Performance
    elif page == "Model Performance":
        st.markdown('<h1 class="main-header">Model Performance</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown('<h2 class="section-header">Evaluation Metrics</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Metrics Explained:</strong><br>
        - <strong>Accuracy:</strong> Overall correctness of predictions<br>
        - <strong>Precision:</strong> How many predicted failures were actual failures<br>
        - <strong>Recall:</strong> How many actual failures were correctly predicted (CRITICAL for this use case)<br>
        - <strong>F1-Score:</strong> Harmonic mean of precision and recall<br>
        - <strong>ROC-AUC:</strong> Area under ROC curve (higher is better)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>Why Recall Matters:</strong><br>
        For failure prediction, false negatives (missed failures) are more costly than 
        false positives (false alarms). Missing a system failure can lead to unexpected 
        downtime and data loss, while false alarms only require investigation.
        </div>
        """, unsafe_allow_html=True)
        
        # Placeholder for actual metrics (would be loaded from training results)
        st.markdown('<h2 class="section-header">Model Comparison</h2>', 
                   unsafe_allow_html=True)
        
        st.info("Run train.py to see actual model performance metrics and comparison charts.")
        
        # Display EDA images if available
        st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', 
                   unsafe_allow_html=True)
        
        eda_images = [
            "class_distribution.png",
            "feature_distributions.png", 
            "correlation_heatmap.png"
        ]
        
        eda_path = Path(EDA_OUTPUT_DIR)
        if eda_path.exists():
            for img_name in eda_images:
                img_path = eda_path / img_name
                if img_path.exists():
                    st.subheader(img_name.replace("_", " ").replace(".png", "").title())
                    st.image(str(img_path))
                else:
                    st.info(f"{img_name} not found. Run train.py to generate EDA visualizations.")
        else:
            st.info("EDA output directory not found. Run train.py to generate visualizations.")
    
    # Page 4: Explainability
    elif page == "Explainability":
        st.markdown('<h1 class="main-header">Explainable AI (SHAP)</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("""
        <div class="info-box">
        <strong>What is SHAP?</strong><br>
        SHAP (SHapley Additive exPlanations) is a game theory approach to explain 
        the output of machine learning models. It shows how each feature contributes 
        to a particular prediction.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>Important Limitations:</strong><br>
        - SHAP values are approximations, not exact explanations<br>
        - They show correlation, not causation<br>
        - Explanations are model-specific, not universal truths<br>
        - They should be used as insights, not definitive answers
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">Global Feature Importance</h2>', 
                   unsafe_allow_html=True)
        
        st.info("Global feature importance shows which features are most influential overall across all predictions.")
        
        st.markdown('<h2 class="section-header">Local Prediction Explanation</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("To see explanations for specific predictions, use the System Health Prediction page and then return here.")
        
        # Feature importance from model
        st.markdown('<h2 class="section-header">Model Feature Importance</h2>', 
                   unsafe_allow_html=True)
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Create bar chart
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                         orientation='h', title='Feature Importance',
                         color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            st.table(importance_df)
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            importance_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                         orientation='h', title='Feature Importance (Absolute Coefficients)',
                         color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            st.table(importance_df)
    
    # Page 5: Diagnostic Assistant
    elif page == "Diagnostic Assistant":
        st.markdown('<h1 class="main-header">AI Diagnostic Assistant</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        # LLM availability status
        if llm_available:
            st.markdown(f'<div class="success-box"><strong>✅ LLM Assistant Available</strong><br>Provider: {llm_assistant.provider}<br>Model: {llm_assistant.model}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box"><strong>⚠️ LLM Assistant Unavailable</strong><br>Using rule-based fallback analysis<br>Configure API key in .env file to enable LLM features</div>', 
                       unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">How It Works</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        1. **ML Model Prediction**: The machine learning model assesses failure risk based on telemetry
        2. **Feature Analysis**: SHAP identifies which features influenced the prediction
        3. **LLM Analysis**: The AI assistant explains the prediction and suggests troubleshooting steps
        4. **Safety First**: The LLM does NOT make its own prediction - it only explains the ML model's output
        """)
        
        st.markdown('<h2 class="section-header">Important Safety Considerations</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="danger-box">
        <strong>⚠️ Critical Safety Rules:</strong><br>
        - The LLM must NOT independently predict system failure<br>
        - Use the ML model's actual prediction as the source of risk assessment<br>
        - The LLM must NOT invent sensor readings or claim confirmed hardware faults<br>
        - All recommendations are suggestions, not validated diagnoses<br>
        - Always consult with qualified IT professionals for critical systems
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-header">Get Diagnostic Report</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("To generate a diagnostic report, first make a prediction on the System Health Prediction page, then return here for the full analysis.")
        
        # Example telemetry input for demonstration
        st.markdown('<h2 class="section-header">Example Usage</h2>', 
                   unsafe_allow_html=True)
        
        with st.expander("View Example Diagnostic Report"):
            example_telemetry = {
                "CPU_Temperature": 75.0,
                "CPU_Usage": 85.0,
                "RAM_Usage": 80.0,
                "Disk_Usage": 70.0,
                "Fan_Speed": 3500.0,
                "Error_Count": 12.0,
                "Runtime_Hours": 800.0
            }
            
            example_features = [
                ("CPU_Temperature", 0.8, 75.0),
                ("Error_Count", 0.6, 12.0),
                ("CPU_Usage", 0.4, 85.0)
            ]
            
            if llm_available:
                try:
                    example_report = llm_assistant.generate_diagnostic_report(
                        telemetry_data=example_telemetry,
                        prediction="Failure Risk",
                        probability=0.85,
                        risk_category="HIGH",
                        important_features=example_features
                    )
                    st.markdown(example_report)
                except Exception as e:
                    st.error(f"Error generating example report: {e}")
            else:
                fallback_report = llm_assistant._generate_fallback_report(
                    example_telemetry, "Failure Risk", 0.85, "HIGH", example_features
                )
                st.markdown(fallback_report)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <small>System Failure Prediction AI Assistant - Portfolio Project</small><br>
        <small>This is a demonstration project using synthetic data. Not for production use.</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()