"""
Healthcare Diagnostic Model Example

This example demonstrates how to use the Federated Learning Client
to train a healthcare diagnostic model while keeping patient data private.
"""

import os
import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import argparse
import json
from typing import Dict, Any, List, Tuple

# Import our client library
from fed_learning_client import FederatedLearningClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthcareDiagnosticClient:
    """
    Federated Learning client for healthcare diagnostic models.
    """
    
    def __init__(self, 
                api_url: str, 
                data_path: str, 
                private_key: str,
                client_id: str = None,
                config_path: str = "client_config.json"):
        """
        Initialize the healthcare diagnostic client.
        
        Args:
            api_url: Base URL for the FL platform API
            data_path: Path to local patient data (CSV)
            private_key: Ethereum private key for authentication
            client_id: Unique identifier for this client (generated if None)
            config_path: Path to configuration file
        """
        self.data_path = data_path
        self.config = self._load_config(config_path)
        
        # Initialize FL client
        self.client = FederatedLearningClient(
            api_url=api_url,
            client_id=client_id
        )
        
        # Load wallet
        self.wallet_address = self.client.load_ethereum_wallet(private_key)
        logger.info(f"Initialized HealthcareDiagnosticClient with wallet: {self.wallet_address}")
        
        # Load and prepare dataset
        self._load_dataset()
        
        # Initialize local model
        self.model = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration parameters
        """
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Use default configuration
            return {
                "training": {
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "epochs": 5,
                    "validation_split": 0.2,
                    "diff_privacy": {
                        "enabled": True,
                        "noise_multiplier": 1.1,
                        "l2_norm_clip": 1.0
                    }
                },
                "model": {
                    "input_shape": None,  # Will be set based on data
                    "hidden_layers": [64, 32],
                    "activation": "relu",
                    "output_classes": 2
                }
            }
    
    def _load_dataset(self):
        """Load and preprocess the local healthcare dataset."""
        try:
            # Load CSV file
            logger.info(f"Loading dataset from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            # Basic preprocessing
            logger.info("Preprocessing dataset")
            # Remove duplicate records
            self.data = self.data.drop_duplicates()
            
            # Handle missing values (simple imputation)
            numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
            self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].mean())
            
            # Split features and target
            # Assuming the last column is the diagnostic outcome (0 or 1)
            self.X = self.data.iloc[:, :-1].values
            self.y = self.data.iloc[:, -1].values
            
            # Standardize features
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
            
            # Split into training and validation sets
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X, self.y, 
                test_size=self.config['training']['validation_split'],
                stratify=self.y,
                random_state=42
            )
            
            # Update model configuration with input shape
            self.config['model']['input_shape'] = self.X.shape[1]
            
            logger.info(f"Dataset loaded: {len(self.X_train)} training samples, {len(self.X_val)} validation samples")
            logger.info(f"Features: {self.X.shape[1]}, Classes: {len(np.unique(self.y))}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _build_model(self, weights: List[np.ndarray] = None):
        """
        Build or update the local model architecture.
        
        Args:
            weights: Initial weights for the model (optional)
        """
        try:
            # Get model configuration
            input_shape = self.config['model']['input_shape']
            hidden_layers = self.config['model']['hidden_layers']
            activation = self.config['model']['activation']
            output_classes = self.config['model']['output_classes']
            
            # Build model
            model = models.Sequential()
            
            # Input layer
            model.add(layers.Dense(hidden_layers[0], activation=activation, input_shape=(input_shape,)))
            
            # Hidden layers
            for units in hidden_layers[1:]:
                model.add(layers.Dense(units, activation=activation))
                model.add(layers.Dropout(0.2))
            
            # Output layer
            if output_classes == 2:
                # Binary classification
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy', tf.keras.metrics.AUC()]
            else:
                # Multi-class classification
                model.add(layers.Dense(output_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['training']['learning_rate']),
                loss=loss,
                metrics=metrics
            )
            
            # Set weights if provided
            if weights is not None:
                model.set_weights(weights)
            
            self.model = model
            logger.info(f"Model built: {model.summary()}")
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def _apply_differential_privacy(self, model):
        """
        Apply differential privacy to the model if enabled.
        
        Args:
            model: Keras model to apply DP to
            
        Returns:
            Model with DP applied
        """
        if not self.config['training']['diff_privacy']['enabled']:
            return model
        
        try:
            import tensorflow_privacy
            from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
            
            # Calculate the parameters for DP
            batch_size = self.config['training']['batch_size']
            noise_multiplier = self.config['training']['diff_privacy']['noise_multiplier']
            l2_norm_clip = self.config['training']['diff_privacy']['l2_norm_clip']
            learning_rate = self.config['training']['learning_rate']
            
            # Create a DP optimizer
            optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=batch_size,
                learning_rate=learning_rate
            )
            
            # Recompile the model with the DP optimizer
            if self.config['model']['output_classes'] == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy', tf.keras.metrics.AUC()]
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
                
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
            
            logger.info(f"Applied differential privacy with noise multiplier: {noise_multiplier}")
            return model
        except ImportError:
            logger.warning("tensorflow_privacy not installed. Differential privacy disabled.")
            return model
        except Exception as e:
            logger.error(f"Failed to apply differential privacy: {e}")
            return model
    
    async def train_and_contribute(self) -> Dict[str, Any]:
        """
        Main function to participate in federated learning.
        
        Returns:
            Dict with contribution results
        """
        try:
            # Authenticate
            await self.client.authenticate()
            
            # Define training function
            def train_function(global_weights, model_info, custom_data=None):
                logger.info(f"Training on global model version {model_info['model_version']}, round {model_info['current_round']}")
                
                # Build model with global weights
                self._build_model(global_weights)
                
                # Apply differential privacy if enabled
                if self.config['training']['diff_privacy']['enabled']:
                    self.model = self._apply_differential_privacy(self.model)
                
                # Save initial weights for computing updates later
                initial_weights = [w.numpy() for w in self.model.weights]
                
                # Train model
                logger.info("Starting local training")
                history = self.model.fit(
                    self.X_train, self.y_train,
                    batch_size=self.config['training']['batch_size'],
                    epochs=self.config['training']['epochs'],
                    validation_data=(self.X_val, self.y_val),
                    verbose=1
                )
                
                # Evaluate model
                evaluation = self.model.evaluate(self.X_val, self.y_val, verbose=0)
                
                # Compute metrics
                metrics = {
                    "loss": float(evaluation[0]),
                    "accuracy": float(evaluation[1]),
                    "dataset_size": len(self.X_train),
                }
                
                # For binary classification, add AUC
                if self.config['model']['output_classes'] == 2 and len(evaluation) > 2:
                    metrics["auc"] = float(evaluation[2])
                
                # Add training history
                metrics["training_history"] = {
                    "loss": [float(x) for x in history.history['loss']],
                    "val_loss": [float(x) for x in history.history['val_loss']],
                    "accuracy": [float(x) for x in history.history['accuracy']],
                    "val_accuracy": [float(x) for x in history.history['val_accuracy']]
                }
                
                # Compute model update (the difference between final and initial weights)
                final_weights = [w.numpy() for w in self.model.weights]
                model_update = []
                for i, final_weight in enumerate(final_weights):
                    update = final_weight - initial_weights[i]
                    model_update.append(update)
                
                logger.info(f"Local training completed with metrics: {metrics}")
                
                return model_update, metrics
            
            # Train and contribute
            result = await self.client.train_and_contribute(train_function)
            
            logger.info(f"Contribution submitted: {result}")
            
            # Get token balance
            balance = await self.client.get_token_balance()
            logger.info(f"Current token balance: {balance}")
            
            return result
        except Exception as e:
            logger.error(f"Training and contribution failed: {e}")
            raise

async def main():
    """Main function to run the healthcare diagnostic client."""
    parser = argparse.ArgumentParser(description='Healthcare Diagnostic Federated Learning Client')
    parser.add_argument('--api_url', required=True, help='Base URL for the federated learning API')
    parser.add_argument('--data_path', required=True, help='Path to local patient data (CSV)')
    parser.add_argument('--private_key', required=True, help='Ethereum private key for authentication')
    parser.add_argument('--client_id', help='Unique identifier for this client (generated if None)')
    parser.add_argument('--config', default='client_config.json', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize client
    client = HealthcareDiagnosticClient(
        api_url=args.api_url,
        data_path=args.data_path,
        private_key=args.private_key,
        client_id=args.client_id,
        config_path=args.config
    )
    
    # Train and contribute
    await client.train_and_contribute()

if __name__ == "__main__":
    asyncio.run(main())
