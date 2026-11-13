"""
Synthetic Data Generator for System Telemetry.

This module generates realistic synthetic system telemetry data with probabilistic
relationships between features and failure risk. The data is designed to be
challenging but learnable for machine learning models.

IMPORTANT: This is synthetic data for educational purposes only.
It does not represent actual hardware telemetry.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import configuration
try:
    from config import (
        RANDOM_STATE, N_SAMPLES, FAILURE_RATIO, FEATURE_NAMES, 
        TARGET_NAME, FEATURE_RANGES, DATA_PATH
    )
except ImportError:
    # Fallback if config module not found
    RANDOM_STATE = 42
    N_SAMPLES = 9000
    FAILURE_RATIO = 0.15
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
    FEATURE_RANGES = {
        "CPU_Temperature": (30, 95),
        "CPU_Usage": (0, 100),
        "RAM_Usage": (0, 100),
        "Disk_Usage": (0, 100),
        "Fan_Speed": (0, 5000),
        "Error_Count": (0, 50),
        "Runtime_Hours": (0, 2000)
    }
    DATA_PATH = "data/system_telemetry.csv"


class SystemTelemetryGenerator:
    """
    Generates synthetic system telemetry data with realistic patterns.
    
    The data generation uses probabilistic relationships rather than deterministic
    thresholds to create overlapping distributions that make classification
    challenging but learnable.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the data generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define parameters for normal operations (healthy systems)
        self.normal_means = np.array([
            45.0,   # CPU_Temperature
            35.0,   # CPU_Usage
            40.0,   # RAM_Usage
            45.0,   # Disk_Usage
            1800.0, # Fan_Speed
            1.0,    # Error_Count
            300.0   # Runtime_Hours
        ])
        
        # Define parameters for failure-prone operations
        self.failure_means = np.array([
            75.0,   # CPU_Temperature (higher)
            80.0,   # CPU_Usage (higher)
            85.0,   # RAM_Usage (higher)
            75.0,   # Disk_Usage (higher)
            3200.0, # Fan_Speed (higher - trying to cool)
            15.0,   # Error_Count (much higher)
            800.0   # Runtime_Hours (older systems)
        ])
        
        # Standard deviations for normal operations
        self.normal_stds = np.array([
            8.0,    # CPU_Temperature
            12.0,   # CPU_Usage
            10.0,   # RAM_Usage
            12.0,   # Disk_Usage
            400.0,  # Fan_Speed
            2.0,    # Error_Count
            200.0   # Runtime_Hours
        ])
        
        # Standard deviations for failure-prone operations (more variability)
        self.failure_stds = self.normal_stds * 1.3
        
        # Correlation matrix - features are correlated in real systems
        # For example: high CPU temp correlates with high CPU usage and fan speed
        self.correlation_matrix = np.array([
            [1.0, 0.7, 0.5, 0.3, 0.6, 0.4, 0.2],  # CPU_Temperature
            [0.7, 1.0, 0.6, 0.4, 0.5, 0.5, 0.3],  # CPU_Usage
            [0.5, 0.6, 1.0, 0.5, 0.3, 0.4, 0.2],  # RAM_Usage
            [0.3, 0.4, 0.5, 1.0, 0.2, 0.3, 0.4],  # Disk_Usage
            [0.6, 0.5, 0.3, 0.2, 1.0, 0.3, 0.1],  # Fan_Speed
            [0.4, 0.5, 0.4, 0.3, 0.3, 1.0, 0.5],  # Error_Count
            [0.2, 0.3, 0.2, 0.4, 0.1, 0.5, 1.0]   # Runtime_Hours
        ])
        
        # Convert correlation to covariance matrix
        self.normal_cov = self._correlation_to_covariance(
            self.correlation_matrix, self.normal_stds
        )
        self.failure_cov = self._correlation_to_covariance(
            self.correlation_matrix, self.failure_stds
        )
    
    def _correlation_to_covariance(self, corr_matrix, stds):
        """
        Convert correlation matrix to covariance matrix.
        
        Args:
            corr_matrix: Correlation matrix
            stds: Standard deviations
            
        Returns:
            Covariance matrix
        """
        diag_std = np.diag(stds)
        return diag_std @ corr_matrix @ diag_std
    
    def _generate_correlated_features(self, n_samples: int, means, cov_matrix):
        """
        Generate features with specified correlations using multivariate normal distribution.
        
        Args:
            n_samples: Number of samples to generate
            means: Mean values for each feature
            cov_matrix: Covariance matrix
            
        Returns:
            Array of generated features
        """
        return np.random.multivariate_normal(means, cov_matrix, size=n_samples)
    
    def _apply_constraints_and_noise(self, features: np.ndarray):
        """
        Apply realistic constraints and add measurement noise.
        
        This ensures generated values are within physically possible ranges
        and adds realistic measurement noise.
        
        Args:
            features: Generated features
            
        Returns:
            Features with constraints applied and noise added
        """
        # Add realistic measurement noise (small random variations)
        noise = np.random.normal(0, 0.02 * features.std(axis=0), size=features.shape)
        features = features + noise
        
        # Apply realistic physical constraints
        features[:, 0] = np.clip(features[:, 0], FEATURE_RANGES["CPU_Temperature"][0], 
                               FEATURE_RANGES["CPU_Temperature"][1])  # CPU_Temperature
        features[:, 1] = np.clip(features[:, 1], FEATURE_RANGES["CPU_Usage"][0], 
                               FEATURE_RANGES["CPU_Usage"][1])  # CPU_Usage
        features[:, 2] = np.clip(features[:, 2], FEATURE_RANGES["RAM_Usage"][0], 
                               FEATURE_RANGES["RAM_Usage"][1])  # RAM_Usage
        features[:, 3] = np.clip(features[:, 3], FEATURE_RANGES["Disk_Usage"][0], 
                               FEATURE_RANGES["Disk_Usage"][1])  # Disk_Usage
        features[:, 4] = np.clip(features[:, 4], FEATURE_RANGES["Fan_Speed"][0], 
                               FEATURE_RANGES["Fan_Speed"][1])  # Fan_Speed
        features[:, 5] = np.clip(features[:, 5], FEATURE_RANGES["Error_Count"][0], 
                               FEATURE_RANGES["Error_Count"][1])  # Error_Count
        features[:, 6] = np.clip(features[:, 6], FEATURE_RANGES["Runtime_Hours"][0], 
                               FEATURE_RANGES["Runtime_Hours"][1])  # Runtime_Hours
        
        # Round to appropriate precision for realism
        features[:, 0] = np.round(features[:, 0], 1)  # Temperature: 1 decimal
        features[:, 1] = np.round(features[:, 1], 1)  # Usage: 1 decimal
        features[:, 2] = np.round(features[:, 2], 1)  # Usage: 1 decimal
        features[:, 3] = np.round(features[:, 3], 1)  # Usage: 1 decimal
        features[:, 4] = np.round(features[:, 4], 0)  # Fan speed: integer
        features[:, 5] = np.round(features[:, 5], 0)  # Error count: integer
        features[:, 6] = np.round(features[:, 6], 1)  # Runtime: 1 decimal
        
        return features
    
    def _calculate_failure_probability(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate failure probability based on feature interactions.
        
        Uses a logistic function with weighted feature combinations to create
        probabilistic relationships rather than hard thresholds. This creates
        overlapping distributions between normal and failure classes.
        
        Args:
            features: Array of feature values
            
        Returns:
            Array of failure probabilities (0-1)
        """
        # Feature weights - reflect real-world importance
        weights = np.array([
            0.15,   # CPU_Temperature (very important)
            0.12,   # CPU_Usage (important)
            0.10,   # RAM_Usage (important)
            0.08,   # Disk_Usage (moderately important)
            -0.05,  # Fan_Speed (higher fan speed reduces risk - cooling works)
            0.20,   # Error_Count (very strong predictor)
            0.10    # Runtime_Hours (older systems more prone to failure)
        ])
        
        # Normalize features to 0-1 range for probability calculation
        normalized_features = self._normalize_for_probability(features)
        
        # Linear combination of weighted features
        linear_score = np.dot(normalized_features, weights)
        
        # Add interaction terms (realistic: high temp + high usage = much higher risk)
        temp_usage_interaction = (normalized_features[:, 0] * normalized_features[:, 1]) * 0.20
        error_runtime_interaction = (normalized_features[:, 5] * normalized_features[:, 6]) * 0.15
        
        linear_score += temp_usage_interaction + error_runtime_interaction
        
        # Add bias term to adjust overall probability
        bias = -0.6
        
        # Apply sigmoid function to convert to probability (0-1)
        # Sigmoid: 1 / (1 + e^(-x))
        probabilities = 1 / (1 + np.exp(-(linear_score + bias)))
        
        return probabilities
    
    def _normalize_for_probability(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to 0-1 range for probability calculation.
        
        Args:
            features: Array of feature values
            
        Returns:
            Normalized features (0-1 range)
        """
        normalized = np.zeros_like(features)
        ranges = list(FEATURE_RANGES.values())
        
        for i in range(features.shape[1]):
            min_val, max_val = ranges[i]
            # Avoid division by zero
            if max_val > min_val:
                normalized[:, i] = (features[:, i] - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0.5  # Midpoint if range is zero
        
        return normalized
    
    def generate_dataset(self, n_samples: int = N_SAMPLES) -> pd.DataFrame:
        """
        Generate the complete synthetic telemetry dataset.
        
        This method creates a realistic dataset with:
        - Correlated features (as in real systems)
        - Probabilistic failure labels (not simple thresholds)
        - Overlapping distributions (challenging but learnable)
        - Realistic noise and constraints
        
        Args:
            n_samples: Total number of samples to generate
            
        Returns:
            DataFrame with features and target variable
        """
        print(f"Generating {n_samples} synthetic telemetry samples...")
        
        # Calculate number of failure and normal samples
        n_failure = int(n_samples * FAILURE_RATIO)
        n_normal = n_samples - n_failure
        
        # Generate normal samples (healthy systems)
        normal_features = self._generate_correlated_features(
            n_normal, self.normal_means, self.normal_cov
        )
        normal_features = self._apply_constraints_and_noise(normal_features)
        
        # Generate failure-prone samples
        failure_features = self._generate_correlated_features(
            n_failure, self.failure_means, self.failure_cov
        )
        failure_features = self._apply_constraints_and_noise(failure_features)
        
        # Combine all features
        all_features = np.vstack([normal_features, failure_features])
        
        # Calculate failure probabilities using probabilistic model
        failure_probs = self._calculate_failure_probability(all_features)
        
        # Assign labels probabilistically (creates overlap between classes)
        # Higher probability = more likely to be labeled as failure
        random_thresholds = np.random.random(n_samples)
        labels = (failure_probs > random_thresholds).astype(int)
        
        # Ensure approximately correct class balance (adjust if needed)
        current_failure_rate = labels.mean()
        target_failure_rate = FAILURE_RATIO
        
        if current_failure_rate < target_failure_rate * 0.9:
            # Too few failures - adjust some high-probability samples
            adjustment_indices = np.where((failure_probs > 0.4) & (labels == 0))[0]
            if len(adjustment_indices) > 0:
                n_adjust = min(len(adjustment_indices), 
                              int(n_samples * target_failure_rate) - labels.sum())
                labels[adjustment_indices[:n_adjust]] = 1
        elif current_failure_rate > target_failure_rate * 1.1:
            # Too many failures - adjust some low-probability samples
            adjustment_indices = np.where((failure_probs < 0.6) & (labels == 1))[0]
            if len(adjustment_indices) > 0:
                n_adjust = min(len(adjustment_indices), 
                              labels.sum() - int(n_samples * target_failure_rate))
                labels[adjustment_indices[:n_adjust]] = 0
        
        # Create DataFrame
        df = pd.DataFrame(all_features, columns=FEATURE_NAMES)
        df[TARGET_NAME] = labels
        
        # Add realistic data quality issues (as in real-world data)
        # Add some missing values (~1%)
        missing_indices = np.random.choice(len(df), size=int(0.01 * len(df)), replace=False)
        for idx in missing_indices:
            col = np.random.choice(FEATURE_NAMES)
            df.loc[idx, col] = np.nan
        
        # Add some duplicate rows (~0.5%)
        duplicate_indices = np.random.choice(len(df), size=int(0.005 * len(df)), replace=False)
        duplicates = df.iloc[duplicate_indices]
        df = pd.concat([df, duplicates], ignore_index=True)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"Dataset generated successfully!")
        print(f"Total samples: {len(df)}")
        print(f"Failure rate: {df[TARGET_NAME].mean():.2%}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicates: {df.duplicated().sum()}")
        
        return df


def main():
    """
    Main function to generate and save the dataset.
    """
    print("=" * 70)
    print("SYSTEM TELEMETRY DATA GENERATION")
    print("=" * 70)
    print()
    
    # Initialize generator
    generator = SystemTelemetryGenerator()
    
    # Generate dataset
    df = generator.generate_dataset()
    
    # Save dataset
    data_path = Path(DATA_PATH)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    
    print()
    print(f"Dataset saved to: {data_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()