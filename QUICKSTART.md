# Quick Start Guide

Get the AI-Powered System Failure Prediction Assistant up and running in minutes.

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
cd system-failure-ai-assistant
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

This will:
- Generate synthetic data (~9,000 samples)
- Perform EDA and create visualizations
- Train and evaluate ML models
- Save the best model

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 🔧 Optional: Enable LLM Features

### For OpenAI (Recommended)
1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

### For Other Providers
See `.env.example` for Anthropic Claude or local Ollama setup.

## 📊 What You'll See

### Dashboard Pages
1. **Overview** - Project introduction and model info
2. **System Health Prediction** - Interactive prediction interface
3. **Model Performance** - Evaluation metrics and EDA plots
4. **Explainability** - SHAP-based model explanations
5. **Diagnostic Assistant** - AI-powered troubleshooting

### Key Features
- ✅ Real-time failure risk prediction
- ✅ Feature importance analysis
- ✅ SHAP explanations
- ✅ LLM diagnostic reports (with API key)
- ✅ Interactive visualizations

## 🎯 Try It Out

1. Go to "System Health Prediction" page
2. Adjust the telemetry sliders
3. Click "Predict System Health"
4. View the prediction and risk category
5. Check "Explainability" for feature analysis
6. Visit "Diagnostic Assistant" for AI suggestions

## 📈 Expected Results

After training, you should see:
- Model accuracy around 85-95%
- Strong recall (minimizing missed failures)
- Clear feature importance rankings
- Interactive SHAP visualizations

## ⚠️ Important Notes

- **Synthetic Data:** This is educational data, not real telemetry
- **Not for Production:** Don't use for actual system monitoring
- **API Key Required:** LLM features need API key configuration
- **First Run:** Training takes 2-5 minutes on first run

## 🆘 Troubleshooting

### Import Errors
```bash
# Make sure you're in the project directory
cd system-failure-ai-assistant

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Model Not Found
```bash
# Run training first
python train.py
```

### LLM Not Working
```bash
# Check your .env file
cat .env

# Ensure API key is set correctly
```

### Port Already in Use
```bash
# Use a different port
streamlit run app.py --port 8502
```

## 📚 Next Steps

1. **Explore the Code:** Check `src/` directory for implementation details
2. **Modify Features:** Edit `config/__init__.py` to change parameters
3. **Add More Models:** Extend `model_trainer.py` with new algorithms
4. **Customize Dashboard:** Modify `app.py` for your needs
5. **Read the Full README:** See `README.md` for detailed documentation

## 🎓 Learning Path

This project teaches:
- Complete ML pipeline implementation
- Data preprocessing and EDA
- Model training and evaluation
- Explainable AI with SHAP
- LLM integration and safety
- Streamlit dashboard development
- Professional software practices

---

**Ready to explore AI-powered system diagnostics? Start with step 1!** 🚀