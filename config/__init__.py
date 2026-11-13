"""
Configuration module for System Failure Prediction AI Assistant.
Contains all constants and configuration parameters.
"""

# Random seed for reproducibility
RANDOM_STATE = 42

# Dataset configuration
N_SAMPLES = 9000  # Number of samples to generate
FAILURE_RATIO = 0.15  # 15% failure rate (realistic for predictive maintenance)
TEST_SIZE = 0.2  # 80% train, 20% test

# Feature names
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

# Feature descriptions for dashboard
FEATURE_DESCRIPTIONS = {
    "CPU_Temperature": "CPU temperature in degrees Celsius (normal: 30-65°C)",
    "CPU_Usage": "CPU utilization percentage (0-100%)",
    "RAM_Usage": "RAM utilization percentage (0-100%)",
    "Disk_Usage": "Disk utilization percentage (0-100%)",
    "Fan_Speed": "Fan speed in RPM (normal: 1000-3000 RPM)",
    "Error_Count": "Number of system errors in the last hour",
    "Runtime_Hours": "Total system runtime in hours"
}

# Feature ranges for data generation
FEATURE_RANGES = {
    "CPU_Temperature": (30, 95),
    "CPU_Usage": (0, 100),
    "RAM_Usage": (0, 100),
    "Disk_Usage": (0, 100),
    "Fan_Speed": (0, 5000),
    "Error_Count": (0, 50),
    "Runtime_Hours": (0, 2000)
}

# Risk thresholds
RISK_THRESHOLDS = {
    "LOW": 0.3,
    "MEDIUM": 0.7,
    "HIGH": 1.0
}

# Model configuration
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

# File paths
DATA_PATH = "data/system_telemetry.csv"
MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
EDA_OUTPUT_DIR = "eda_output"

# LLM Configuration
LLM_PROVIDER = "openai"  # Options: "openai", "anthropic", "ollama"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7  # Lower for more deterministic responses
LLM_MAX_TOKENS = 500