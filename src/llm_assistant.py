"""
LLM Diagnostic Assistant Module.

This module integrates with Large Language Model APIs to provide
intelligent diagnostic assistance and troubleshooting suggestions.

IMPORTANT SAFETY CONSIDERATIONS:
- The LLM must NOT independently predict system failure
- Use the ML model's actual prediction as the source of risk assessment
- The LLM must NOT invent sensor readings or claim confirmed hardware faults
- All recommendations are suggestions, not validated diagnoses
- API keys must be kept secure in environment variables
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
try:
    from config import (
        LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
        FEATURE_NAMES, FEATURE_DESCRIPTIONS, RISK_THRESHOLDS
    )
except ImportError:
    # Fallback configuration
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 500
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
    RISK_THRESHOLDS = {"LOW": 0.3, "MEDIUM": 0.7, "HIGH": 1.0}


class LLMDiagnosticAssistant:
    """
    LLM-powered diagnostic assistant for system failure prediction.
    
    This class provides intelligent explanations and troubleshooting suggestions
    based on ML model predictions and telemetry data.
    """
    
    def __init__(self, provider: str = None, model: str = None):
        """
        Initialize the LLM diagnostic assistant.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'ollama')
            model: Model name to use
        """
        self.provider = provider or LLM_PROVIDER
        self.model = model or LLM_MODEL
        self.api_key = self._load_api_key()
        self.client = None
        self._initialize_client()
    
    def _load_api_key(self) -> Optional[str]:
        """
        Load API key from environment variables.
        
        Returns:
            API key string or None if not found
        """
        # Try to load from environment variable
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == "ollama":
            # Ollama doesn't require API key
            return None
        else:
            print(f"Warning: Unknown provider '{self.provider}'")
            return None
        
        if not api_key:
            print(f"Warning: API key for {self.provider} not found in environment variables")
            print("LLM features will be disabled. Set up .env file with API key.")
        
        return api_key
    
    def _initialize_client(self):
        """
        Initialize the LLM client based on provider.
        """
        if not self.api_key and self.provider != "ollama":
            print("LLM client not initialized (no API key)")
            return
        
        try:
            if self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(f"OpenAI client initialized with model: {self.model}")
            
            elif self.provider == "anthropic":
                try:
                    from anthropic import Anthropic
                    self.client = Anthropic(api_key=self.api_key)
                    print(f"Anthropic client initialized with model: {self.model}")
                except ImportError:
                    print("Warning: anthropic package not installed")
                    print("Install with: pip install anthropic")
            
            elif self.provider == "ollama":
                try:
                    import requests
                    self.client = requests
                    print(f"Ollama client initialized with model: {self.model}")
                except ImportError:
                    print("Warning: requests package not available for Ollama")
            
        except Exception as e:
            print(f"Error initializing {self.provider} client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """
        Check if LLM service is available.
        
        Returns:
            True if LLM is available, False otherwise
        """
        return self.client is not None
    
    def generate_diagnostic_report(
        self,
        telemetry_data: Dict[str, float],
        prediction: str,
        probability: float,
        risk_category: str,
        important_features: List[Tuple[str, float, float]]
    ) -> str:
        """
        Generate a comprehensive diagnostic report using LLM.
        
        Args:
            telemetry_data: Dictionary of telemetry values
            prediction: Model prediction ("Normal" or "Failure Risk")
            probability: Failure probability (0-1)
            risk_category: Risk category ("LOW", "MEDIUM", "HIGH")
            important_features: List of (feature_name, shap_value, feature_value) tuples
            
        Returns:
            Generated diagnostic report string
        """
        if not self.is_available():
            return self._generate_fallback_report(
                telemetry_data, prediction, probability, 
                risk_category, important_features
            )
        
        # Build the prompt for the LLM
        prompt = self._build_diagnostic_prompt(
            telemetry_data, prediction, probability,
            risk_category, important_features
        )
        
        try:
            if self.provider == "openai":
                return self._call_openai(prompt)
            elif self.provider == "anthropic":
                return self._call_anthropic(prompt)
            elif self.provider == "ollama":
                return self._call_ollama(prompt)
            else:
                return self._generate_fallback_report(
                    telemetry_data, prediction, probability,
                    risk_category, important_features
                )
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return self._generate_fallback_report(
                telemetry_data, prediction, probability,
                risk_category, important_features
            )
    
    def _build_diagnostic_prompt(
        self,
        telemetry_data: Dict[str, float],
        prediction: str,
        probability: float,
        risk_category: str,
        important_features: List[Tuple[str, float, float]]
    ) -> str:
        """
        Build the diagnostic prompt for the LLM.
        
        Args:
            telemetry_data: Dictionary of telemetry values
            prediction: Model prediction
            probability: Failure probability
            risk_category: Risk category
            important_features: List of important features
            
        Returns:
            Formatted prompt string
        """
        prompt = """You are a helpful IT diagnostic assistant. Analyze the following system telemetry data and provide diagnostic insights.

IMPORTANT INSTRUCTIONS:
- The ML model has already assessed the failure risk - do NOT make your own prediction
- Use only the provided telemetry values - do NOT invent additional sensor readings
- Do NOT claim a confirmed hardware fault - these are suggestions for investigation
- Provide practical, actionable troubleshooting steps
- Keep recommendations general and safe

SYSTEM TELEMETRY DATA:
"""
        
        # Add telemetry data
        for feature, value in telemetry_data.items():
            description = FEATURE_DESCRIPTIONS.get(feature, "")
            prompt += f"- {feature}: {value:.2f} {description}\n"
        
        prompt += f"""
ML MODEL ASSESSMENT:
- Prediction: {prediction}
- Failure Probability: {probability:.1%}
- Risk Category: {risk_category}

IMPORTANT FEATURES (influencing this prediction):
"""
        
        # Add important features
        for feature, shap_value, feature_value in important_features[:5]:
            direction = "increases" if shap_value > 0 else "decreases"
            prompt += f"- {feature} (value: {feature_value:.2f}): {direction} failure risk\n"
        
        prompt += """
Please provide:
1. A simple explanation of what the ML model detected
2. Which telemetry values deserve attention and why
3. Possible causes to investigate (3-4 specific suggestions)
4. Suggested troubleshooting steps (practical and safe)
5. Preventive maintenance recommendations

Keep the response clear, professional, and actionable. Use bullet points for readability.
"""
        
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI API.
        
        Args:
            prompt: Prompt string
            
        Returns:
            Generated response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful IT diagnostic assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def _call_anthropic(self, prompt: str) -> str:
        """
        Call Anthropic API.
        
        Args:
            prompt: Prompt string
            
        Returns:
            Generated response
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama (local LLM) API.
        
        Args:
            prompt: Prompt string
            
        Returns:
            Generated response
        """
        try:
            response = self.client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")
    
    def _generate_fallback_report(
        self,
        telemetry_data: Dict[str, float],
        prediction: str,
        probability: float,
        risk_category: str,
        important_features: List[Tuple[str, float, float]]
    ) -> str:
        """
        Generate a fallback diagnostic report without LLM.
        
        This provides useful information when LLM is unavailable.
        
        Args:
            telemetry_data: Dictionary of telemetry values
            prediction: Model prediction
            probability: Failure probability
            risk_category: Risk category
            important_features: List of important features
            
        Returns:
            Generated diagnostic report string
        """
        report = f"""
SYSTEM DIAGNOSTIC REPORT (LLM Unavailable - Using Rule-Based Analysis)
{'='*70}

ML MODEL ASSESSMENT:
- Prediction: {prediction}
- Failure Probability: {probability:.1%}
- Risk Category: {risk_category}

IMPORTANT FEATURES INFLUENCING THIS PREDICTION:
"""
        
        for i, (feature, shap_value, feature_value) in enumerate(important_features[:5], 1):
            direction = "increases" if shap_value > 0 else "decreases"
            impact = "HIGH" if abs(shap_value) > 0.5 else "MODERATE" if abs(shap_value) > 0.2 else "LOW"
            report += f"\n{i}. {feature} (value: {feature_value:.2f})\n"
            report += f"   Impact: {impact} - {direction} failure risk\n"
        
        report += f"""
GENERAL TROUBLESHOOTING SUGGESTIONS:

1. Check Cooling System:
   - Verify fan operation and speed
   - Clean dust from heatsinks and vents
   - Ensure proper airflow around the system

2. Monitor Resource Usage:
   - Identify and close unnecessary processes
   - Check for memory leaks in applications
   - Verify disk space availability

3. Review Error Logs:
   - Check system event logs for error patterns
   - Investigate recent error messages
   - Review application crash reports

4. System Maintenance:
   - Update system drivers and firmware
   - Run disk cleanup and defragmentation
   - Perform virus/malware scan

5. Hardware Inspection:
   - Check for physical damage or wear
   - Verify cable connections
   - Test power supply stability

IMPORTANT NOTES:
- This is a machine learning prediction, not a confirmed diagnosis
- Recommendations are suggestions for investigation
- Consult with IT professionals for critical systems
- Regular preventive maintenance helps prevent failures

{'='*70}
"""
        
        return report


def main():
    """
    Main function to demonstrate LLM assistant.
    """
    print("=" * 70)
    print("LLM DIAGNOSTIC ASSISTANT DEMO")
    print("=" * 70)
    
    # Create assistant
    assistant = LLMDiagnosticAssistant()
    
    if assistant.is_available():
        print("LLM assistant is available!")
        
        # Example usage
        telemetry_data = {
            "CPU_Temperature": 75.0,
            "CPU_Usage": 85.0,
            "RAM_Usage": 80.0,
            "Disk_Usage": 70.0,
            "Fan_Speed": 3500.0,
            "Error_Count": 12.0,
            "Runtime_Hours": 800.0
        }
        
        important_features = [
            ("CPU_Temperature", 0.8, 75.0),
            ("Error_Count", 0.6, 12.0),
            ("CPU_Usage", 0.4, 85.0)
        ]
        
        report = assistant.generate_diagnostic_report(
            telemetry_data=telemetry_data,
            prediction="Failure Risk",
            probability=0.85,
            risk_category="HIGH",
            important_features=important_features
        )
        
        print(report)
    else:
        print("LLM assistant is not available (check API key configuration)")


if __name__ == "__main__":
    main()