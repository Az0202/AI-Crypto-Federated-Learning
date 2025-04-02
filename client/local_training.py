"""
Local Training Module for Decentralized Federated Learning Platform

This module implements the client-side training logic with privacy-preserving mechanisms.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow_privacy import PrivacyAwareTrainer
import yaml
import logging
from typing import Dict, Any, Tuple, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalTrainer:
    """
    Implements local model training with privacy-preserving mechanisms.
    """
    
    def __init__(self, config_path: str = "client_config.yaml"):
        """
        Initialize the local trainer with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.dataset = None
        self.client_id = self._generate_client_id()
        
        # Set up TensorFlow for GPU usage if available
        self._setup_tf_environment()
        
        logger.info(f"Initialized LocalTrainer with client ID: {self.client_id}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration parameters
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Use default configuration
            return {
                "privacy": {
                    "use_differential_privacy": True,
                    "noise_multiplier": 1.1,
                    "l2_norm_clip": 1.0
                },
                "training": {
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "epochs": 5,
                    "optimizer": "adam"
                },
                "model": {
                    "architecture": "cnn",
                    "input_shape": [28, 28, 1]
                }
            }
    
    def _setup_tf_environment(self):
        """Configure TensorFlow environment"""
        # Enable GPU growth to avoid allocating all GPU memory
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Found {len(physical_devices)} GPU(s). Enabled memory growth.")
            except Exception as e:
                logger.warning(f"Error configuring GPU: {e}")
    
    def _generate_client_id(self) -> str:
        """
        Generate a unique client ID.
        
        Returns:
            A unique identifier for this client
        """
        import uuid
        return str(uuid.uuid4())
    
    def load_dataset(self, dataset_path: str) -> None:
        """
        Load local dataset for training.
        
        Args:
            dataset_path: Path to the dataset
        """
        # Implementation depends on the specific data format
        # This is a placeholder for actual data loading logic
        logger.info(f"Loading dataset from {dataset_path}")
        
        try:
            # Example for loading image data
            if self.config['model']['architecture'] == 'cnn':
                # For image classification tasks
                self.dataset = tf.keras.preprocessing.image_dataset_from_directory(
                    dataset_path,
                    batch_size=self.config['training']['batch_size'],
                    image_size=self.config['model']['input_shape'][:2]
                )
            else:
                # For tabular data, we'd implement a different loading mechanism
                # Placeholder for tabular data loading
                logger.warning("Tabular data loading not implemented yet")
                
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def build_model(self) -> None:
        """Build the model architecture based on configuration"""
        architecture = self.config['model']['architecture']
        logger.info(f"Building model with {architecture} architecture")
        
        try:
            if architecture == 'cnn':
                # Example CNN for image classification
                self.model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=self.config['model']['input_shape']),
                    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(10, activation="softmax")
                ])
            elif architecture == 'mlp':
                # Simple MLP for tabular data
                self.model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu', input_shape=(self.config['model']['input_shape'][0],)),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")
                
            # Compile the model
            optimizer_name = self.config['training']['optimizer']
            learning_rate = self.config['training']['learning_rate']
            
            if optimizer_name == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name == 'sgd':
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
                
            self.model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Model built and compiled with {optimizer_name} optimizer")
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def train(self) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Train the model on local data with privacy-preserving mechanisms.
        
        Returns:
            Tuple containing (metrics, model_update)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Starting local training")
        
        # Save initial weights for computing updates later
        initial_weights = [w.numpy() for w in self.model.weights]
        
        try:
            # Apply differential privacy if configured
            if self.config['privacy']['use_differential_privacy']:
                logger.info("Using differential privacy for training")
                
                # Create a privacy-aware optimizer
                from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
                
                optimizer = DPAdamGaussianOptimizer(
                    l2_norm_clip=self.config['privacy']['l2_norm_clip'],
                    noise_multiplier=self.config['privacy']['noise_multiplier'],
                    learning_rate=self.config['training']['learning_rate']
                )
                
                # Recompile the model with the privacy-aware optimizer
                self.model.compile(
                    optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            # Train the model
            history = self.model.fit(
                self.dataset,
                epochs=self.config['training']['epochs'],
                verbose=1
            )
            
            # Compute model update (the difference between final and initial weights)
            model_update = []
            for i, final_weight in enumerate(self.model.weights):
                update = final_weight.numpy() - initial_weights[i]
                model_update.append(update)
            
            # Extract metrics from training history
            metrics = {
                'loss': float(history.history['loss'][-1]),
                'accuracy': float(history.history['accuracy'][-1]),
                'client_id': self.client_id,
                'dataset_size': sum(1 for _ in self.dataset)
            }
            
            logger.info(f"Training completed with metrics: {metrics}")
            
            # Add metadata for the contribution
            metadata = {
                'client_id': self.client_id,
                'architecture': self.config['model']['architecture'],
                'timestamp': self._get_current_timestamp(),
                'dp_applied': self.config['privacy']['use_differential_privacy'],
                'dp_noise_multiplier': self.config['privacy']['noise_multiplier'] if self.config['privacy']['use_differential_privacy'] else None
            }
            
            return metrics, {'weights': model_update, 'metadata': metadata}
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _get_current_timestamp(self) -> int:
        """Get current timestamp in seconds"""
        import time
        return int(time.time())
    
    def load_global_model(self, model_weights: List[np.ndarray]) -> None:
        """
        Load global model weights for next round of training.
        
        Args:
            model_weights: List of weight arrays
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        logger.info("Loading global model weights")
        
        try:
            for i, weights in enumerate(model_weights):
                self.model.weights[i].assign(weights)
                
            logger.info("Global model weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load global model weights: {e}")
            raise
    
    def evaluate(self, test_dataset) -> Dict[str, float]:
        """
        Evaluate the model on local test data.
        
        Args:
            test_dataset: Dataset for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        logger.info("Evaluating model")
        
        try:
            results = self.model.evaluate(test_dataset, verbose=1)
            metrics = {
                'test_loss': float(results[0]),
                'test_accuracy': float(results[1])
            }
            
            logger.info(f"Model evaluation results: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    trainer = LocalTrainer()
    trainer.build_model()
    trainer.load_dataset("./data/local_dataset")
    metrics, model_update = trainer.train()
    print(f"Training metrics: {metrics}")
    print(f"Model update metadata: {model_update['metadata']}")
