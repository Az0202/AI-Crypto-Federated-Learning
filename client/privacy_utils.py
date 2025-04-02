"""
Privacy Utilities for Federated Learning

This module provides differential privacy mechanisms for federated learning,
ensuring that model updates don't leak sensitive information about training data.
"""

import numpy as np
import tensorflow as tf
import logging
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import time

# Try to import tensorflow_privacy if available
try:
    import tensorflow_privacy
    from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
    from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
    _HAS_DP = True
except ImportError:
    _HAS_DP = False
    logging.warning("tensorflow_privacy not installed. Limited differential privacy features available.")

logger = logging.getLogger(__name__)

class DPModelUpdater:
    """
    Applies differential privacy techniques to model updates.
    
    This class provides methods to add noise to gradients, clip gradients, 
    and calculate the privacy budget for given parameters.
    """
    
    def __init__(
        self,
        noise_multiplier: float = 1.0,
        l2_norm_clip: float = 1.0,
        microbatches: int = 1,
        delta: float = 1e-5
    ):
        """
        Initialize the differential privacy updater.
        
        Args:
            noise_multiplier: Amount of noise to add (higher = more privacy)
            l2_norm_clip: Clipping norm for gradients
            microbatches: Number of microbatches to use
            delta: Target delta for privacy guarantee
        """
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.microbatches = microbatches
        self.delta = delta
        
        # Check if TF Privacy is available
        self.has_dp = _HAS_DP
        
        # Performance metrics
        self.metrics = {
            "clip_times": [],
            "noise_times": [],
            "total_processed": 0
        }
    
    def apply_dp_to_optimizer(
        self, 
        optimizer: tf.keras.optimizers.Optimizer
    ) -> tf.keras.optimizers.Optimizer:
        """
        Convert a standard optimizer to a differentially private version.
        
        Args:
            optimizer: Standard TensorFlow optimizer
            
        Returns:
            Differentially private optimizer
        """
        if not self.has_dp:
            logger.warning("tensorflow_privacy not installed. Using standard optimizer instead.")
            return optimizer
        
        try:
            # Create DP optimizer
            if isinstance(optimizer, tf.keras.optimizers.Adam):
                dp_optimizer = DPAdamGaussianOptimizer(
                    l2_norm_clip=self.l2_norm_clip,
                    noise_multiplier=self.noise_multiplier,
                    num_microbatches=self.microbatches,
                    learning_rate=optimizer.learning_rate
                )
                logger.info("Created DP Adam optimizer")
                return dp_optimizer
            
            # If not Adam, try to use generic DP SGD optimizer
            try:
                from tensorflow_privacy.privacy.optimizers.dp_optimizer import make_gaussian_optimizer_class
                DPOptimizerClass = make_gaussian_optimizer_class(type(optimizer))
                
                dp_optimizer = DPOptimizerClass(
                    l2_norm_clip=self.l2_norm_clip,
                    noise_multiplier=self.noise_multiplier,
                    num_microbatches=self.microbatches,
                    learning_rate=optimizer.learning_rate
                )
                logger.info(f"Created DP optimizer from {type(optimizer).__name__}")
                return dp_optimizer
            except Exception as e:
                logger.error(f"Failed to create DP optimizer: {e}")
                logger.warning("Using standard optimizer instead")
                return optimizer
                
        except Exception as e:
            logger.error(f"Error creating DP optimizer: {e}")
            return optimizer
    
    def make_dp_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Convert a standard model to use differentially private training.
        
        Args:
            model: Standard TensorFlow model
            
        Returns:
            Model with DP-enabled optimizer
        """
        if not self.has_dp:
            logger.warning("tensorflow_privacy not installed. Using standard model instead.")
            return model
        
        try:
            # Get current optimizer and metrics
            optimizer = model.optimizer
            loss = model.loss
            metrics = model.metrics
            
            # Create DP optimizer
            dp_optimizer = self.apply_dp_to_optimizer(optimizer)
            
            # Recompile model with DP optimizer
            model.compile(
                optimizer=dp_optimizer,
                loss=loss,
                metrics=metrics
            )
            
            logger.info("Successfully converted model to use DP training")
            return model
        except Exception as e:
            logger.error(f"Error creating DP model: {e}")
            return model
    
    def apply_dp_to_gradients(
        self, 
        grads: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Apply differential privacy to gradients manually.
        
        Args:
            grads: List of gradient arrays
            
        Returns:
            Privacy-preserving gradients
        """
        start_time = time.time()
        
        # First clip gradients
        clipped_grads = self.clip_gradients(grads)
        
        clip_time = time.time() - start_time
        self.metrics["clip_times"].append(clip_time)
        
        # Then add noise
        noisy_grads = self.add_noise(clipped_grads)
        
        noise_time = time.time() - start_time - clip_time
        self.metrics["noise_times"].append(noise_time)
        
        # Track total processed values
        self.metrics["total_processed"] += sum(g.size for g in grads)
        
        return noisy_grads
    
    def clip_gradients(self, grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Clip gradients to limit sensitivity.
        
        Args:
            grads: List of gradient arrays
            
        Returns:
            Clipped gradients
        """
        # Calculate the l2 norm of all gradients
        flat_concat = np.concatenate([g.flatten() for g in grads])
        l2_norm = np.sqrt(np.sum(np.square(flat_concat)))
        
        # Calculate scaling factor
        scale = 1.0
        if l2_norm > self.l2_norm_clip:
            scale = self.l2_norm_clip / (l2_norm + 1e-10)
        
        # Apply scaling to clip gradients
        clipped_grads = [g * scale for g in grads]
        
        return clipped_grads
    
    def add_noise(self, grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Add Gaussian noise to gradients for privacy.
        
        Args:
            grads: List of gradient arrays
            
        Returns:
            Noisy gradients
        """
        noisy_grads = []
        
        for g in grads:
            # Calculate noise standard deviation
            noise_stddev = self.l2_norm_clip * self.noise_multiplier
            
            # Generate noise matching gradient shape
            noise = np.random.normal(0, noise_stddev, g.shape)
            
            # Add noise to gradient
            noisy_g = g + noise
            noisy_grads.append(noisy_g)
        
        return noisy_grads
    
    def compute_privacy_spent(
        self,
        num_examples: int,
        batch_size: int,
        epochs: int
    ) -> Tuple[float, float]:
        """
        Compute the privacy spent (epsilon, delta) for given training parameters.
        
        Args:
            num_examples: Total number of training examples
            batch_size: Batch size used for training
            epochs: Number of training epochs
            
        Returns:
            Tuple of (epsilon, delta)
        """
        if not self.has_dp:
            logger.warning("tensorflow_privacy not installed. Cannot compute privacy spent.")
            return (float('inf'), self.delta)
        
        try:
            # Calculate number of steps per epoch
            steps_per_epoch = num_examples // batch_size
            
            # Calculate total steps
            steps = steps_per_epoch * epochs
            
            # Compute epsilon using TF Privacy's computation
            eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                n=num_examples,
                batch_size=batch_size,
                noise_multiplier=self.noise_multiplier,
                epochs=epochs,
                delta=self.delta
            )
            
            return (eps, self.delta)
        except Exception as e:
            logger.error(f"Failed to compute privacy: {e}")
            return (float('inf'), self.delta)
    
    def apply_dp_to_model_update(
        self, 
        weights_update: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Apply differential privacy to model weight updates.
        
        Args:
            weights_update: List of weight update arrays
            
        Returns:
            Privacy-preserving weight updates
        """
        # For weight updates, we can use the same approach as gradients
        return self.apply_dp_to_gradients(weights_update)
    
    def get_dp_info(self) -> Dict[str, Any]:
        """
        Get information about the current DP configuration.
        
        Returns:
            Dictionary with DP parameters and metrics
        """
        return {
            "dp_available": self.has_dp,
            "noise_multiplier": self.noise_multiplier,
            "l2_norm_clip": self.l2_norm_clip,
            "microbatches": self.microbatches,
            "delta": self.delta,
            "metrics": {
                "avg_clip_time_ms": np.mean(self.metrics["clip_times"]) * 1000 if self.metrics["clip_times"] else 0,
                "avg_noise_time_ms": np.mean(self.metrics["noise_times"]) * 1000 if self.metrics["noise_times"] else 0,
                "total_processed": self.metrics["total_processed"]
            }
        }


class PrivacyReportGenerator:
    """
    Generates privacy reports and recommendations for federated learning.
    """
    
    @staticmethod
    def generate_privacy_report(dp_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a privacy report with interpretations and recommendations.
        
        Args:
            dp_info: Output from DPModelUpdater.get_dp_info()
            
        Returns:
            Dictionary with privacy report
        """
        # Extract parameters
        noise_multiplier = dp_info["noise_multiplier"]
        l2_norm_clip = dp_info["l2_norm_clip"]
        
        # Generate basic risk assessment
        if noise_multiplier < 0.5:
            privacy_level = "Low"
            risk = "High"
            recommendation = "Increase noise_multiplier to at least 1.0 for better privacy protection."
        elif noise_multiplier < 1.0:
            privacy_level = "Moderate"
            risk = "Medium"
            recommendation = "Consider increasing noise_multiplier if privacy is a major concern."
        else:
            privacy_level = "High"
            risk = "Low"
            recommendation = "Current settings provide good privacy protection."
        
        # Generate report
        report = {
            "privacy_level": privacy_level,
            "risk_level": risk,
            "recommendations": [recommendation],
            "parameters": {
                "noise_multiplier": noise_multiplier,
                "l2_norm_clip": l2_norm_clip
            }
        }
        
        # Add additional recommendations based on l2_norm_clip
        if l2_norm_clip > 10.0:
            report["recommendations"].append(
                "The l2_norm_clip value is high, which might reduce the effectiveness of privacy protection. "
                "Consider reducing it to 1.0-5.0 range."
            )
        elif l2_norm_clip < 0.1:
            report["recommendations"].append(
                "The l2_norm_clip value is very low, which might impact model utility. "
                "Consider increasing it to 0.5-1.0 range."
            )
        
        return report


# Utility functions
def optimize_dp_parameters(
    utility_target: float,
    privacy_target: float,
    dataset_size: int
) -> Dict[str, float]:
    """
    Optimize differential privacy parameters to balance utility and privacy.
    
    Args:
        utility_target: Target utility (accuracy) value (0-1)
        privacy_target: Target privacy (epsilon) value
        dataset_size: Size of the dataset
        
    Returns:
        Dictionary with recommended DP parameters
    """
    # This is a simplified heuristic - in a real implementation,
    # you might use more sophisticated optimization methods
    
    # Start with reasonable defaults
    params = {
        "noise_multiplier": 1.0,
        "l2_norm_clip": 1.0,
        "learning_rate": 0.001,
        "batch_size": min(32, dataset_size // 10)  # Ensure reasonable batch size
    }
    
    # Adjust noise multiplier based on privacy target
    # Lower epsilon requires higher noise
    if privacy_target < 1.0:
        params["noise_multiplier"] = 1.5  # More noise for stricter privacy
    elif privacy_target < 3.0:
        params["noise_multiplier"] = 1.0  # Moderate noise
    else:
        params["noise_multiplier"] = 0.7  # Less noise for more utility
    
    # Adjust learning rate and l2_norm_clip based on utility target
    # Higher utility targets might need more aggressive training
    if utility_target > 0.9:
        params["learning_rate"] = 0.01
        params["l2_norm_clip"] = 3.0
    elif utility_target > 0.8:
        params["learning_rate"] = 0.005
        params["l2_norm_clip"] = 2.0
    
    # Adjust based on dataset size
    if dataset_size < 1000:
        # Small datasets need more privacy protection and careful training
        params["noise_multiplier"] *= 1.2
        params["learning_rate"] *= 0.8
    elif dataset_size > 10000:
        # Large datasets can tolerate more aggressive learning
        params["batch_size"] = 64
    
    return params


def make_dp_compatible_model(model: tf.keras.Model, dp_params: Dict[str, float]) -> tf.keras.Model:
    """
    Make a model compatible with differential privacy training.
    
    Args:
        model: Original TensorFlow model
        dp_params: Differential privacy parameters
        
    Returns:
        Model ready for DP training
    """
    if not _HAS_DP:
        logger.warning("tensorflow_privacy not installed. Returning original model.")
        return model
    
    try:
        from tensorflow_privacy.privacy.optimizers.dp_optimizer import make_gaussian_optimizer_class
        
        # Get current optimizer or use Adam as default
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            optimizer = model.optimizer
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=dp_params.get("learning_rate", 0.001))
        
        # Create DP optimizer
        DPOptimizerClass = make_gaussian_optimizer_class(type(optimizer))
        
        dp_optimizer = DPOptimizerClass(
            l2_norm_clip=dp_params.get("l2_norm_clip", 1.0),
            noise_multiplier=dp_params.get("noise_multiplier", 1.0),
            num_microbatches=dp_params.get("microbatches", 1),
            learning_rate=dp_params.get("learning_rate", 0.001)
        )
        
        # Get loss and metrics from original model
        loss = model.loss if hasattr(model, 'loss') else 'categorical_crossentropy'
        metrics = model.metrics if hasattr(model, 'metrics') else ['accuracy']
        
        # Recompile model
        model.compile(
            optimizer=dp_optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info("Successfully created DP-compatible model")
        return model
        
    except Exception as e:
        logger.error(f"Failed to make DP-compatible model: {e}")
        return model


# Example usage
if __name__ == "__main__":
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create DP updater
    dp_updater = DPModelUpdater(
        noise_multiplier=1.2,
        l2_norm_clip=1.0
    )
    
    # Make model DP-compatible
    dp_model = dp_updater.make_dp_model(model)
    
    # Print DP info
    dp_info = dp_updater.get_dp_info()
    print(f"DP Info: {dp_info}")
    
    # Generate privacy report
    report = PrivacyReportGenerator.generate_privacy_report(dp_info)
    print(f"Privacy Report: {report}")
