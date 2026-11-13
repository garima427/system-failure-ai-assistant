# AI-Powered System Failure Prediction & Diagnostic Assistant

A comprehensive machine learning project that predicts potential system failure risk using system telemetry data and provides understandable explanations and troubleshooting suggestions using Explainable AI (SHAP) and Generative AI (LLM).

## 🎯 Project Objective

Build an application that predicts potential system failure risk using system telemetry data and provides understandable explanations and troubleshooting suggestions using Explainable AI and Generative AI.

### Three Main Components
1. **Machine Learning Model** - Failure prediction using Logistic Regression and Random Forest
2. **Explainable AI** - SHAP-based feature importance and prediction explanations
3. **LLM Diagnostic Assistant** - AI-powered troubleshooting suggestions and diagnostic reports

## 🚀 Technology Stack

- **Python** - Core programming language
- **Pandas & NumPy** - Data manipulation and numerical operations
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms and preprocessing
- **SHAP** - Explainable AI for model interpretation
- **Streamlit** - Interactive dashboard
- **Joblib** - Model persistence
- **OpenAI API** (or alternatives) - LLM for diagnostic assistance

## 📊 Dataset

### Synthetic Data Generation
The project generates synthetic system telemetry data with approximately 9,000 records.

### Features
- **CPU_Temperature** - CPU temperature in degrees Celsius (30-95°C)
- **CPU_Usage** - CPU utilization percentage (0-100%)
- **RAM_Usage** - RAM utilization percentage (0-100%)
- **Disk_Usage** - Disk utilization percentage (0-100%)
- **Fan_Speed** - Fan speed in RPM (0-5000)
- **Error_Count** - Number of system errors (0-50)
- **Runtime_Hours** - Total system runtime in hours (0-2000)

### Target Variable
- **Failure_Risk** - Binary classification (0 = Normal, 1 = Potential Failure Risk)

### Data Generation Method
- Uses probabilistic relationships between features
- Implements realistic correlations (e.g., high CPU temp correlates with high usage)
- Creates overlapping distributions between normal and failure classes
- Adds realistic noise and measurement errors
- Includes missing values (~1%) and duplicates (~0.5%) for realism

**⚠️ Important:** This is synthetic data for educational purposes only. It does not represent actual hardware telemetry.

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory:**
```bash
cd system-failure-ai-assistant
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure LLM API (optional but recommended):**
```bash
cp .env.example .env
# Edit .env file and add your API key
```

For OpenAI:
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

## 🏃 Running the Project

### Step 1: Train the ML Models
```bash
python train.py
```

This will:
- Generate synthetic telemetry data
- Perform exploratory data analysis (EDA)
- Train Logistic Regression and Random Forest models
- Evaluate both models
- Select the best model
- Save the model and preprocessor
- Generate EDA visualizations

### Step 2: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
system-failure-ai-assistant/
│
├── config/
│   └── __init__.py              # Configuration constants
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py        # Synthetic data generation
│   ├── data_preprocessing.py    # EDA and preprocessing
│   ├── model_trainer.py         # ML model training and evaluation
│   ├── explainability.py        # SHAP-based explainability
│   └── llm_assistant.py         # LLM diagnostic assistant
│
├── data/
│   └── system_telemetry.csv     # Generated dataset
│
├── models/
│   ├── best_model.joblib        # Trained ML model
│   └── preprocessor.joblib      # Fitted preprocessor
│
├── eda_output/
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   └── correlation_heatmap.png
│
├── notebooks/                   # Jupyter notebooks for exploration
│
├── train.py                     # Main training pipeline
├── app.py                       # Streamlit dashboard
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🎓 Key Concepts Explained

### 1. Machine Learning Pipeline
The project follows a complete ML pipeline:
```
Data Generation → EDA → Preprocessing → Train/Test Split → 
Model Training → Evaluation → Model Selection → Deployment
```

### 2. Data Leakage Prevention
- **Train/test split happens BEFORE preprocessing**
- Preprocessing is FIT only on training data
- Test data is only TRANSFORMED using fitted parameters
- Test set is used only for final evaluation

### 3. Logistic Regression
**What it is:** A linear classification algorithm that uses the sigmoid function to output probabilities.

**Sigmoid Function:** σ(z) = 1 / (1 + e^(-z))
- Converts linear output to probability (0-1)
- Output of 0.5 is the decision boundary
- Values > 0.5 predict class 1 (failure risk)
- Values < 0.5 predict class 0 (normal)

**Why use it:** Simple, interpretable baseline model that provides probability outputs.

### 4. Random Forest
**What it is:** An ensemble method that combines multiple decision trees.

**How it works:**
- Each tree is trained on a random subset of data and features
- Predictions are aggregated (voting for classification)
- Reduces overfitting through ensemble averaging

**Why use it:** Handles non-linear relationships, robust to overfitting, provides feature importance.

### 5. Model Evaluation Metrics

**Accuracy:** Overall correctness of predictions
```
Accuracy = (TP + TN) / Total
```

**Precision:** How many predicted failures were actual failures
```
Precision = TP / (TP + FP)
```

**Recall:** How many actual failures were correctly predicted (CRITICAL)
```
Recall = TP / (TP + FN)
```

**F1-Score:** Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**ROC-AUC:** Area under ROC curve (higher is better)

### 6. Why Recall Matters
For failure prediction:
- **False Negative** = Missed failure (system fails unexpectedly) → **HIGH COST**
- **False Positive** = False alarm (system flagged but healthy) → **LOW COST**

Therefore, recall is prioritized to minimize missed failures.

### 7. SHAP (Explainable AI)
**What it is:** SHAP (SHapley Additive exPlanations) uses game theory to explain model predictions.

**How it works:**
- Calculates contribution of each feature to individual predictions
- Shows both global (overall) and local (per-prediction) importance
- Provides intuitive visualizations

**Limitations:**
- SHAP values are approximations, not exact explanations
- They show correlation, not causation
- Model-specific, not universal truths

### 8. LLM Diagnostic Assistant
**What it does:** Provides intelligent explanations and troubleshooting suggestions based on ML predictions.

**Safety Features:**
- LLM does NOT independently predict system failure
- Uses ML model's actual prediction as risk assessment source
- Does NOT invent sensor readings or claim confirmed hardware faults
- All recommendations are suggestions, not validated diagnoses

**Fallback:** If LLM is unavailable, uses rule-based analysis.

## 📈 Model Performance

### Model Selection Criteria
Models are selected based on a composite score:
- **50% Recall** (most important - don't miss failures)
- **30% F1-Score** (balance of precision and recall)
- **20% ROC-AUC** (overall discriminative ability)

### Expected Performance
Run `train.py` to see actual performance metrics. The synthetic dataset is designed to be challenging but learnable.

## 🔧 Dashboard Features

### Overview Page
- Project introduction and objectives
- Model information
- Dataset summary
- Important disclaimers

### System Health Prediction Page
- Interactive sliders for all telemetry features
- Real-time prediction
- Failure probability and risk category
- Telemetry summary display

### Model Performance Page
- Evaluation metrics explanation
- Model comparison
- EDA visualizations
- Confusion matrices and ROC curves

### Explainability Page
- SHAP explanation
- Global feature importance
- Local prediction explanations
- Model feature importance

### Diagnostic Assistant Page
- LLM availability status
- AI-powered diagnostic reports
- Troubleshooting suggestions
- Safety considerations

## ⚠️ Important Disclaimers

1. **Synthetic Data:** The dataset is simulated and does not represent actual hardware telemetry.

2. **Educational Purpose:** This project demonstrates ML methodology, not real-world deployment.

3. **Not for Production:** Do not use this system for real-world critical infrastructure.

4. **Professional Consultation:** Always consult with qualified IT professionals for actual system diagnostics.

5. **LLM Limitations:** AI suggestions are recommendations, not validated diagnoses.

## 🔐 Security Considerations

- **API Keys:** Never commit API keys to version control
- **Environment Variables:** Use `.env` file for sensitive configuration
- **Model Security:** Validate inputs before processing
- **Output Validation:** Review LLM outputs for safety

## 🚧 Limitations

1. **Synthetic Data:** Does not represent real-world system behavior
2. **Simplified Features:** Real systems have many more telemetry parameters
3. **No Temporal Analysis:** Doesn't consider time-series patterns
4. **Static Model:** Model doesn't update with new data
5. **Single System Type:** Assumes homogeneous system characteristics
6. **LLM Dependency:** Requires API key for full functionality

## 📚 Future Improvements

1. **Real Data Integration:** Connect to actual system monitoring
2. **Time-Series Analysis:** Incorporate temporal patterns
3. **More Features:** Additional telemetry parameters
4. **Advanced Models:** Experiment with gradient boosting, neural networks
5. **Real-time Monitoring:** Live system integration
6. **Multi-class Prediction:** Predict specific failure types
7. **Model Updates:** Online learning for continuous improvement
8. **Deployment:** Containerize and deploy to production

## 🤝 Contributing

This is a portfolio project for educational purposes. Suggestions and improvements are welcome!

## 📄 License

This is a portfolio project for educational purposes.

## 👨‍💻 Technical Implementation Details

### Data Generation Algorithm
The synthetic data generator uses:
- **Multivariate normal distributions** for correlated features
- **Logistic function** for probabilistic label assignment
- **Interaction terms** for realistic feature relationships
- **Measurement noise** for realism
- **Class imbalance** (~15% failure rate)

### Preprocessing Pipeline
- **Median imputation** for missing values
- **Standard scaling** (z-score normalization)
- **Stratified train/test split** (80/20)
- **No data leakage** (preprocessing fit on train only)

### Model Hyperparameters
- **Logistic Regression:** max_iter=1000
- **Random Forest:** n_estimators=100, max_depth=10, min_samples_split=5

### SHAP Integration
- **TreeExplainer** for Random Forest (optimized)
- **Summary plots** for global importance
- **Force plots** for local explanations
- **Plain language explanations** for non-technical users

### LLM Integration
- **OpenAI API** (default) with fallback to alternatives
- **Structured prompts** for consistent outputs
- **Error handling** with rule-based fallback
- **Safety constraints** in prompt engineering

## 🎯 Learning Outcomes

This project demonstrates:
- Complete ML pipeline from data to deployment
- Explainable AI for model interpretation
- Generative AI integration for assistance
- Professional software engineering practices
- Security best practices
- Beginner-friendly but professional code

## 📞 Support

For questions or issues, please refer to the inline code documentation and comments throughout the project.

---

**Built with ❤️ for learning and demonstration purposes**