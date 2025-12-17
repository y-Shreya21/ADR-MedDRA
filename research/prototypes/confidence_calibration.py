# Confidence Calibration System
# Research prototype for calibrating model confidence scores

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pickle
import warnings

warnings.filterwarnings('ignore')

class ConfidenceCalibrator:
    """
    Advanced confidence calibration system for ADR-MedDRA predictions.
    
    This prototype implements multiple calibration methods to provide
    reliable confidence estimates for clinical decision-making.
    """
    
    def __init__(self, method='isotonic', bins=10):
        """
        Initialize calibrator with specified method.
        
        Args:
            method: Calibration method ('isotonic', 'platt', 'histogram')
            bins: Number of bins for histogram calibration
        """
        self.method = method
        self.bins = bins
        self.calibrator = None
        self.calibration_curve = None
        self.is_fitted = False
        
        # Initialize calibrator based on method
        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == 'platt':
            self.calibrator = LogisticRegression(random_state=42)
        elif method == 'histogram':
            self.bin_boundaries = None
            self.bin_true_probabilities = None
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def fit(self, scores: np.ndarray, true_labels: np.ndarray, 
            validation_scores: Optional[np.ndarray] = None,
            validation_labels: Optional[np.ndarray] = None):
        """
        Fit calibration curve on validation data.
        
        Args:
            scores: Raw model confidence scores (0-1)
            true_labels: Binary ground truth (1 for correct, 0 for incorrect)
            validation_scores: Additional validation scores for cross-validation
            validation_labels: Additional validation labels
        """
        scores = np.array(scores).reshape(-1, 1) if scores.ndim == 1 else scores
        true_labels = np.array(true_labels)
        
        if self.method == 'isotonic':
            self.calibrator.fit(scores.ravel(), true_labels)
        
        elif self.method == 'platt':
            # Platt scaling: fit logistic regression
            self.calibrator.fit(scores, true_labels)
        
        elif self.method == 'histogram':
            self._fit_histogram_calibration(scores.ravel(), true_labels)
        
        # Store calibration curve for analysis
        self.calibration_curve = self._compute_calibration_curve(scores.ravel(), true_labels)
        self.is_fitted = True
        
        # Evaluate calibration quality
        if validation_scores is not None and validation_labels is not None:
            self.calibration_metrics = self._evaluate_calibration(
                validation_scores, validation_labels
            )
    
    def _fit_histogram_calibration(self, scores: np.ndarray, true_labels: np.ndarray):
        """Fit histogram-based calibration (binning method)"""
        # Create equal-width bins
        self.bin_boundaries = np.linspace(0, 1, self.bins + 1)
        self.bin_true_probabilities = np.zeros(self.bins)
        
        for i in range(self.bins):
            # Find samples in this bin
            bin_mask = (scores >= self.bin_boundaries[i]) & (scores < self.bin_boundaries[i + 1])
            
            if i == self.bins - 1:  # Include right boundary for last bin
                bin_mask |= (scores == 1.0)
            
            if np.sum(bin_mask) > 0:
                # Calculate true probability for this bin
                self.bin_true_probabilities[i] = np.mean(true_labels[bin_mask])
            else:
                # Use linear interpolation for empty bins
                self.bin_true_probabilities[i] = (self.bin_boundaries[i] + self.bin_boundaries[i + 1]) / 2
    
    def calibrate_score(self, score: float) -> float:
        """
        Calibrate a single confidence score.
        
        Args:
            score: Raw confidence score (0-1)
            
        Returns:
            Calibrated probability
        """
        if not self.is_fitted:
            warnings.warn("Calibrator not fitted. Returning original score.")
            return score
        
        score = np.clip(score, 0, 1)  # Ensure valid range
        
        if self.method == 'isotonic':
            return float(self.calibrator.predict([score])[0])
        
        elif self.method == 'platt':
            return float(self.calibrator.predict_proba([[score]])[0, 1])
        
        elif self.method == 'histogram':
            return self._histogram_calibrate_single(score)
        
        return score
    
    def _histogram_calibrate_single(self, score: float) -> float:
        """Calibrate single score using histogram method"""
        # Find appropriate bin
        bin_idx = np.digitize(score, self.bin_boundaries) - 1
        bin_idx = np.clip(bin_idx, 0, self.bins - 1)
        
        return self.bin_true_probabilities[bin_idx]
    
    def calibrate_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        Calibrate confidence scores for a list of predictions.
        
        Args:
            predictions: List of prediction dictionaries with 'score' key
            
        Returns:
            Updated predictions with calibrated scores and confidence levels
        """
        for pred in predictions:
            original_score = pred.get('score', 0.0)
            calibrated_score = self.calibrate_score(original_score)
            
            pred['calibrated_score'] = calibrated_score
            pred['confidence_level'] = self._get_confidence_level(calibrated_score)
            pred['reliability'] = self._assess_reliability(calibrated_score)
            
            # Keep original for comparison
            pred['original_score'] = original_score
            pred['calibration_method'] = self.method
        
        return predictions
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert calibrated score to interpretable confidence level"""
        if score >= 0.95:
            return "Very High"
        elif score >= 0.80:
            return "High"
        elif score >= 0.60:
            return "Medium"
        elif score >= 0.40:
            return "Low"
        else:
            return "Very Low"
    
    def _assess_reliability(self, score: float) -> str:
        """Assess reliability of the calibrated score"""
        if hasattr(self, 'calibration_metrics'):
            ece = self.calibration_metrics.get('expected_calibration_error', 0.1)
            
            if ece < 0.05:
                return "Highly Reliable"
            elif ece < 0.10:
                return "Reliable"
            elif ece < 0.15:
                return "Moderately Reliable"
            else:
                return "Low Reliability"
        
        return "Unknown"
    
    def _compute_calibration_curve(self, scores: np.ndarray, true_labels: np.ndarray) -> Dict:
        """Compute calibration curve for analysis"""
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels, scores, n_bins=self.bins
        )
        
        return {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'perfect_calibration': np.linspace(0, 1, len(mean_predicted_value))
        }
    
    def _evaluate_calibration(self, scores: np.ndarray, true_labels: np.ndarray) -> Dict:
        """Evaluate calibration quality using multiple metrics"""
        calibrated_scores = np.array([self.calibrate_score(s) for s in scores])
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(calibrated_scores, true_labels)
        
        # Brier Score (lower is better)
        brier_score = brier_score_loss(true_labels, calibrated_scores)
        
        # Log Loss (lower is better)
        log_loss_score = log_loss(true_labels, calibrated_scores)
        
        # Reliability diagram statistics
        reliability_stats = self._calculate_reliability_stats(calibrated_scores, true_labels)
        
        return {
            'expected_calibration_error': ece,
            'brier_score': brier_score,
            'log_loss': log_loss_score,
            'reliability_stats': reliability_stats
        }
    
    def _calculate_ece(self, scores: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in bin
            in_bin = (scores > bin_lower) & (scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = scores[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_reliability_stats(self, scores: np.ndarray, true_labels: np.ndarray) -> Dict:
        """Calculate detailed reliability statistics"""
        # Bin predictions
        bin_stats = []
        for i in range(self.bins):
            bin_lower = i / self.bins
            bin_upper = (i + 1) / self.bins
            
            in_bin = (scores > bin_lower) & (scores <= bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_stats.append({
                    'bin_range': (bin_lower, bin_upper),
                    'count': np.sum(in_bin),
                    'accuracy': np.mean(true_labels[in_bin]),
                    'avg_confidence': np.mean(scores[in_bin]),
                    'calibration_error': abs(np.mean(scores[in_bin]) - np.mean(true_labels[in_bin]))
                })
        
        return bin_stats
    
    def plot_calibration_curve(self, save_path: Optional[str] = None):
        """Plot calibration curve for visual analysis"""
        if not self.is_fitted or self.calibration_curve is None:
            raise ValueError("Calibrator must be fitted before plotting")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Calibration curve
        curve = self.calibration_curve
        ax1.plot(curve['mean_predicted_value'], curve['fraction_of_positives'], 
                marker='o', linewidth=2, label=f'{self.method.title()} Calibration')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Calibration Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reliability diagram (if calibration metrics available)
        if hasattr(self, 'calibration_metrics') and 'reliability_stats' in self.calibration_metrics:
            stats = self.calibration_metrics['reliability_stats']
            
            bin_centers = [(s['bin_range'][0] + s['bin_range'][1]) / 2 for s in stats]
            accuracies = [s['accuracy'] for s in stats]
            confidences = [s['avg_confidence'] for s in stats]
            counts = [s['count'] for s in stats]
            
            # Bar plot showing calibration error
            ax2.bar(bin_centers, accuracies, width=0.08, alpha=0.7, 
                   color='blue', label='Accuracy')
            ax2.plot(bin_centers, confidences, 'ro-', label='Confidence')
            ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Accuracy / Confidence')
            ax2.set_title('Reliability Diagram')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_calibrator(self, filepath: str):
        """Save fitted calibrator to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")
        
        calibrator_data = {
            'method': self.method,
            'bins': self.bins,
            'calibrator': self.calibrator,
            'calibration_curve': self.calibration_curve,
            'is_fitted': self.is_fitted
        }
        
        if self.method == 'histogram':
            calibrator_data['bin_boundaries'] = self.bin_boundaries
            calibrator_data['bin_true_probabilities'] = self.bin_true_probabilities
        
        with open(filepath, 'wb') as f:
            pickle.dump(calibrator_data, f)
    
    @classmethod
    def load_calibrator(cls, filepath: str):
        """Load fitted calibrator from disk"""
        with open(filepath, 'rb') as f:
            calibrator_data = pickle.load(f)
        
        calibrator = cls(method=calibrator_data['method'], bins=calibrator_data['bins'])
        calibrator.calibrator = calibrator_data['calibrator']
        calibrator.calibration_curve = calibrator_data['calibration_curve']
        calibrator.is_fitted = calibrator_data['is_fitted']
        
        if calibrator.method == 'histogram':
            calibrator.bin_boundaries = calibrator_data['bin_boundaries']
            calibrator.bin_true_probabilities = calibrator_data['bin_true_probabilities']
        
        return calibrator

# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    
    # Simulate model scores and ground truth
    n_samples = 1000
    
    # Create slightly overconfident scores (common in neural networks)
    true_probabilities = np.random.beta(2, 2, n_samples)
    model_scores = np.clip(true_probabilities + np.random.normal(0, 0.1, n_samples), 0, 1)
    
    # Generate binary outcomes based on true probabilities
    outcomes = np.random.binomial(1, true_probabilities, n_samples)
    
    # Split data for calibration and testing
    split_idx = int(0.7 * n_samples)
    
    cal_scores = model_scores[:split_idx]
    cal_outcomes = outcomes[:split_idx]
    test_scores = model_scores[split_idx:]
    test_outcomes = outcomes[split_idx:]
    
    # Test different calibration methods
    methods = ['isotonic', 'platt', 'histogram']
    
    for method in methods:
        print(f"\n=== Testing {method.upper()} Calibration ===")
        
        # Initialize and fit calibrator
        calibrator = ConfidenceCalibrator(method=method, bins=10)
        calibrator.fit(cal_scores, cal_outcomes, test_scores, test_outcomes)
        
        # Evaluate calibration
        if hasattr(calibrator, 'calibration_metrics'):
            metrics = calibrator.calibration_metrics
            print(f"Expected Calibration Error: {metrics['expected_calibration_error']:.4f}")
            print(f"Brier Score: {metrics['brier_score']:.4f}")
            print(f"Log Loss: {metrics['log_loss']:.4f}")
        
        # Test prediction calibration
        sample_predictions = [
            {'score': 0.3, 'pt_name': 'Nausea'},
            {'score': 0.7, 'pt_name': 'Headache'},
            {'score': 0.9, 'pt_name': 'Muscle pain'},
            {'score': 0.1, 'pt_name': 'Dizziness'}
        ]
        
        calibrated_preds = calibrator.calibrate_predictions(sample_predictions.copy())
        
        print("\nSample Calibrated Predictions:")
        for pred in calibrated_preds:
            print(f"  {pred['pt_name']}: {pred['original_score']:.2f} → "
                  f"{pred['calibrated_score']:.2f} ({pred['confidence_level']})")
    
    # Plot calibration curves for comparison
    print("\n=== Generating Calibration Plots ===")
    for method in methods:
        calibrator = ConfidenceCalibrator(method=method)
        calibrator.fit(cal_scores, cal_outcomes, test_scores, test_outcomes)
        
        try:
            calibrator.plot_calibration_curve(f'calibration_curve_{method}.png')
        except Exception as e:
            print(f"Could not generate plot for {method}: {e}")