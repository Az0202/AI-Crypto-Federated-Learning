"""
Unit Tests for Local Trainer Module

This module tests the local training functionality of the decentralized
federated learning platform, including privacy features.
"""

import unittest
import os
import numpy as np
import tensorflow as tf
import tempfile
import yaml
import json
from unittest.mock import patch, MagicMock

# Import the module to test - adjust path as needed
from client.local_training import LocalTrainer

class TestLocalTrainer(unittest.TestCase):
    """Test cases for the LocalTrainer class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary configuration file
        self.config = {
            "privacy": {
                "use_differential_privacy": True,
                "noise_multiplier": 1.1,
                "l2_norm_clip": 1.0
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 2,
                "optimizer": "adam"
            },
            "model": {
                "architecture": "mlp",
                "input_shape": [10]
            }
        }
        
        # Create a temporary config file
        fd, self.config_path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            yaml.dump(self.config, f)
        
        # Create a temporary dataset
        self.X_train = np.random.randn(100, 10).astype(np.float32)
        self.y_train = np.random.randint(0, 2, size=(100,)).astype(np.int32)
        
        # Create a temporary dataset directory and file
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, 'test_dataset')
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Create a TF dataset and save it
        np.savez(
            os.path.join(self.temp_dir, 'test_data.npz'),
            X=self.X_train,
            y=self.y_train
        )
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary files
        os.remove(self.config_path)
        os.remove(os.path.join(self.temp_dir, 'test_data.npz'))
        os.rmdir(self.dataset_path)
        os.rmdir(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization of LocalTrainer."""
        trainer = LocalTrainer(config_path=self.config_path)
        
        # Check that config was loaded correctly
        self.assertEqual(trainer.config['privacy']['noise_multiplier'], 1.1)
        self.assertEqual(trainer.config['training']['batch_size'], 32)
        
        # Check that client_id was generated
        self.assertIsNotNone(trainer.client_id)
        self.assertTrue(len(trainer.client_id) > 0)
    
    def test_build_model(self):
        """Test model building functionality."""
        trainer = LocalTrainer(config_path=self.config_path)
        trainer.build_model()
        
        # Check that model was created
        self.assertIsNotNone(trainer.model)
        
        # Check model architecture
        self.assertEqual(len(trainer.model.layers), 4)  # Input, 2 hidden, output
        
        # Check that the model is compiled
        self.assertIsNotNone(trainer.model.optimizer)
        self.assertIsNotNone(trainer.model.loss)
    
    @patch('tensorflow_privacy.privacy.optimizers.dp_optimizer.DPAdamGaussianOptimizer')
    def test_train_with_differential_privacy(self, mock_dp_optimizer):
        """Test training with differential privacy."""
        # Mock the DP optimizer
        mock_dp_optimizer.return_value = tf.keras.optimizers.Adam()
        
        # Initialize trainer
        trainer = LocalTrainer(config_path=self.config_path)
        
        # Create a simple dataset
        X = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=(32,)).astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(8)
        trainer.dataset = dataset
        
        # Build model
        trainer.build_model()
        
        # Train model
        metrics, model_update = trainer.train()
        
        # Check that metrics were computed
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('client_id', metrics)
        self.assertIn('dataset_size', metrics)
        
        # Check that model update was created
        self.assertIn('weights', model_update)
        self.assertIn('metadata', model_update)
        
        # Check that DP was applied
        self.assertTrue(model_update['metadata']['dp_applied'])
        self.assertEqual(model_update['metadata']['dp_noise_multiplier'], 1.1)
    
    def test_train_without_differential_privacy(self):
        """Test training without differential privacy."""
        # Update config to disable DP
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['privacy']['use_differential_privacy'] = False
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Initialize trainer
        trainer = LocalTrainer(config_path=self.config_path)
        
        # Create a simple dataset
        X = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=(32,)).astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(8)
        trainer.dataset = dataset
        
        # Build model
        trainer.build_model()
        
        # Train model
        metrics, model_update = trainer.train()
        
        # Check that DP was not applied
        self.assertFalse(model_update['metadata']['dp_applied'])
        self.assertIsNone(model_update['metadata']['dp_noise_multiplier'])
    
    def test_load_global_model(self):
        """Test loading global model weights."""
        trainer = LocalTrainer(config_path=self.config_path)
        trainer.build_model()
        
        # Create some mock weights
        original_weights = trainer.model.get_weights()
        new_weights = [w * 2 for w in original_weights]  # Double the weights
        
        # Load the new weights
        trainer.load_global_model(new_weights)
        
        # Check that weights were updated
        updated_weights = trainer.model.get_weights()
        
        for i in range(len(original_weights)):
            # Check that weights are different
            self.assertFalse(np.array_equal(original_weights[i], updated_weights[i]))
            
            # Check that weights match the new weights
            np.testing.assert_array_almost_equal(new_weights[i], updated_weights[i])
    
    def test_evaluate(self):
        """Test model evaluation."""
        trainer = LocalTrainer(config_path=self.config_path)
        
        # Build model
        trainer.build_model()
        
        # Create a test dataset
        X = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=(32,)).astype(np.int32)
        test_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(8)
        
        # Evaluate model
        metrics = trainer.evaluate(test_dataset)
        
        # Check that metrics were computed
        self.assertIn('test_loss', metrics)
        self.assertIn('test_accuracy', metrics)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        trainer = LocalTrainer(config_path=self.config_path)
        
        # Test train without building model
        with self.assertRaises(ValueError):
            trainer.train()
        
        # Test train without loading dataset
        trainer.build_model()
        with self.assertRaises(ValueError):
            trainer.train()
        
        # Test evaluate without building model
        trainer = LocalTrainer(config_path=self.config_path)
        X = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=(32,)).astype(np.int32)
        test_dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(8)
        
        with self.assertRaises(ValueError):
            trainer.evaluate(test_dataset)
        
        # Test load_global_model without building model
        trainer = LocalTrainer(config_path=self.config_path)
        mock_weights = [np.zeros((10, 10)), np.zeros(10)]
        
        with self.assertRaises(ValueError):
            trainer.load_global_model(mock_weights)

if __name__ == '__main__':
    unittest.main()
