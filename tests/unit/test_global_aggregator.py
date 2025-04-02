"""
Unit Tests for Global Aggregator Module

This module tests the global aggregation functionality of the decentralized
federated learning platform, including different aggregation methods and
quality verification.
"""

import unittest
import os
import numpy as np
import tensorflow as tf
import tempfile
import yaml
import json
import time
from unittest.mock import patch, MagicMock, Mock
import threading

# Import the module to test - adjust path as needed
from aggregator.global_aggregator import GlobalAggregator

class TestGlobalAggregator(unittest.TestCase):
    """Test cases for the GlobalAggregator class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary configuration file
        self.config = {
            "aggregation": {
                "method": "fedavg",
                "min_contributions": 3,
                "contribution_timeout_seconds": 3600,
                "rounds_per_epoch": 10
            },
            "quality_verification": {
                "enabled": True,
                "min_accuracy": 0.6,
                "max_loss": 2.0,
                "outlier_detection": True,
                "cosine_similarity_threshold": 0.7
            },
            "model": {
                "architecture": "mlp",
                "input_shape": [10],
                "save_path": tempfile.mkdtemp()
            },
            "blockchain": {
                "enabled": False,
                "provider_url": "http://localhost:8545",
                "contract_address": "0x123...",
                "gas_limit": 3000000
            }
        }
        
        # Create a temporary config file
        fd, self.config_path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            yaml.dump(self.config, f)
        
        # Create a mock model with simple weights
        self.mock_weights = [
            np.ones((10, 10)),
            np.ones(10),
            np.ones((10, 2)),
            np.ones(2)
        ]
        
        # Create mock contributions
        self.mock_contributions = [
            {
                'client_id': 'client_1',
                'metrics': {
                    'loss': 0.5,
                    'accuracy': 0.85,
                    'dataset_size': 1000
                },
                'model_update': {
                    'weights': [w * 0.1 for w in self.mock_weights],
                    'metadata': {
                        'client_id': 'client_1',
                        'timestamp': int(time.time()),
                        'architecture': 'mlp'
                    }
                }
            },
            {
                'client_id': 'client_2',
                'metrics': {
                    'loss': 0.6,
                    'accuracy': 0.82,
                    'dataset_size': 800
                },
                'model_update': {
                    'weights': [w * 0.15 for w in self.mock_weights],
                    'metadata': {
                        'client_id': 'client_2',
                        'timestamp': int(time.time()),
                        'architecture': 'mlp'
                    }
                }
            },
            {
                'client_id': 'client_3',
                'metrics': {
                    'loss': 0.7,
                    'accuracy': 0.8,
                    'dataset_size': 1200
                },
                'model_update': {
                    'weights': [w * 0.12 for w in self.mock_weights],
                    'metadata': {
                        'client_id': 'client_3',
                        'timestamp': int(time.time()),
                        'architecture': 'mlp'
                    }
                }
            }
        ]
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary files
        os.remove(self.config_path)
        os.rmdir(self.config['model']['save_path'])
    
    @patch('tensorflow.keras.models.load_model')
    @patch('tensorflow.keras.Sequential')
    def test_initialization(self, mock_sequential, mock_load_model):
        """Test proper initialization of GlobalAggregator."""
        # Mock model building
        mock_model = Mock()
        mock_model.weights = [tf.Variable(w) for w in self.mock_weights]
        mock_sequential.return_value = mock_model
        mock_load_model.side_effect = FileNotFoundError  # Force new model creation
        
        # Initialize aggregator
        aggregator = GlobalAggregator(config_path=self.config_path)
        
        # Check that config was loaded correctly
        self.assertEqual(aggregator.config['aggregation']['method'], 'fedavg')
        self.assertEqual(aggregator.config['aggregation']['min_contributions'], 3)
        
        # Check that model was initialized
        self.assertIsNotNone(aggregator.global_model)
        
        # Check initial state
        self.assertEqual(aggregator.current_round, 0)
        self.assertEqual(aggregator.model_version, "0.0.1")
        self.assertEqual(len(aggregator.contributions), 0)
    
    @patch('tensorflow.keras.models.load_model')
    @patch('tensorflow.keras.Sequential')
    def test_receive_contribution(self, mock_sequential, mock_load_model):
        """Test receiving and processing a contribution."""
        # Mock model building
        mock_model = Mock()
        mock_model.weights = [tf.Variable(w) for w in self.mock_weights]
        mock_sequential.return_value = mock_model
        mock_load_model.side_effect = FileNotFoundError  # Force new model creation
        
        # Initialize aggregator
        aggregator = GlobalAggregator(config_path=self.config_path)
        
        # Receive a contribution
        contribution = self.mock_contributions[0]
        result = aggregator.receive_contribution(
            client_id=contribution['client_id'],
            metrics=contribution['metrics'],
            model_update=contribution['model_update']
        )
        
        # Check that contribution was stored
        self.assertEqual(len(aggregator.contributions), 1)
        
        # Find the contribution ID from the result
        contribution_id = result['contribution_id']
        
        # Check contribution fields
        stored_contribution = aggregator.contributions[contribution_id]
        self.assertEqual(stored_contribution['client_id'], contribution['client_id'])
        self.assertEqual(stored_contribution['metrics'], contribution['metrics'])
        self.assertEqual(stored_contribution['model_update'], contribution['model_update'])
        self.assertTrue(stored_contribution['verified'])  # Should pass verification
        
        # Check result
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['current_round'], 0)
        self.assertEqual(result['model_version'], "0.0.1")
    
    @patch('tensorflow.keras.models.load_model')
    @patch('tensorflow.keras.Sequential')
    def test_verify_contribution_quality(self, mock_sequential, mock_load_model):
        """Test contribution quality verification."""
        # Mock model building
        mock_model = Mock()
        mock_model.weights = [tf.Variable(w) for w in self.mock_weights]
        mock_sequential.return_value = mock_model
        mock_load_model.side_effect = FileNotFoundError  # Force new model creation
        
        # Initialize aggregator
        aggregator = GlobalAggregator(config_path=self.config_path)
        
        # Test good contribution - should pass verification
        good_contribution = self.mock_contributions[0]
        good_result = aggregator.receive_contribution(
            client_id=good_contribution['client_id'],
            metrics=good_contribution['metrics'],
            model_update=good_contribution['model_update']
        )
        
        good_id = good_result['contribution_id']
        self.assertTrue(aggregator.contributions[good_id]['verified'])
        
        # Test contribution with bad accuracy - should fail verification
        bad_contribution = self.mock_contributions[1].copy()
        bad_contribution['metrics'] = bad_contribution['metrics'].copy()
        bad_contribution['metrics']['accuracy'] = 0.5  # Below min_accuracy threshold
        
        bad_result = aggregator.receive_contribution(
            client_id=bad_contribution['client_id'],
            metrics=bad_contribution['metrics'],
            model_update=bad_contribution['model_update']
        )
        
        bad_id = bad_result['contribution_id']
        self.assertFalse(aggregator.contributions[bad_id]['verified'])
        self.assertEqual(
            aggregator.contributions[bad_id]['verification_result']['reason'],
            "Accuracy below threshold"
        )
        
        # Test contribution with high loss - should fail verification
        high_loss_contribution = self.mock_contributions[2].copy()
        high_loss_contribution['metrics'] = high_loss_contribution['metrics'].copy()
        high_loss_contribution['metrics']['loss'] = 3.0  # Above max_loss threshold
        
        high_loss_result = aggregator.receive_contribution(
            client_id=high_loss_contribution['client_id'],
            metrics=high_loss_contribution['metrics'],
            model_update=high_loss_contribution['model_update']
        )
        
        high_loss_id = high_loss_result['contribution_id']
        self.assertFalse(aggregator.contributions[high_loss_id]['verified'])
        self.assertEqual(
            aggregator.contributions[high_loss_id]['verification_result']['reason'],
            "Loss above threshold"
        )
    
    @patch('tensorflow.keras.models.load_model')
    @patch('tensorflow.keras.Sequential')
    @patch.object(GlobalAggregator, '_record_contribution_on_blockchain')
    @patch.object(GlobalAggregator, 'aggregate_contributions')
    def test_aggregation_triggering(self, mock_aggregate, mock_record, mock_sequential, mock_load_model):
        """Test that aggregation is triggered when enough contributions are received."""
        # Mock model building
        mock_model = Mock()
        mock_model.weights = [tf.Variable(w) for w in self.mock_weights]
        mock_sequential.return_value = mock_model
        mock_load_model.side_effect = FileNotFoundError  # Force new model creation
        
        # Mock threading to avoid actual background thread
        with patch('threading.Thread'):
            # Initialize aggregator
            aggregator = GlobalAggregator(config_path=self.config_path)
            
            # Receive contributions one by one
            for i in range(2):
                contribution = self.mock_contributions[i]
                aggregator.receive_contribution(
                    client_id=contribution['client_id'],
                    metrics=contribution['metrics'],
                    model_update=contribution['model_update']
                )
            
            # Check that aggregation wasn't triggered yet
            mock_aggregate.assert_not_called()
            
            # Receive the third contribution (should trigger aggregation)
            contribution = self.mock_contributions[2]
            aggregator.receive_contribution(
                client_id=contribution['client_id'],
                metrics=contribution['metrics'],
                model_update=contribution['model_update']
            )
            
            # Check that aggregation was triggered
            mock_aggregate.assert_called_once()
    
    @patch('tensorflow.keras.models.load_model')
    @patch('tensorflow.keras.Sequential')
    def test_federated_averaging(self, mock_sequential, mock_load_model):
        """Test the federated averaging aggregation method."""
        # Mock model building
        mock_model = Mock()
        mock_model.weights = [tf.Variable(w) for w in self.mock_weights]
        mock_sequential.return_value = mock_model
        mock_load_model.side_effect = FileNotFoundError  # Force new model creation
        
        # Initialize aggregator
        aggregator = GlobalAggregator(config_path=self.config_path)
        
        # Create test contributions list
        contributions = []
        for contrib_data in self.mock_contributions:
            # Add required fields for the federated averaging function
            contrib = {
                'model_update': contrib_data['model_update'],
                'metrics': contrib_data['metrics']
            }
            contributions.append(contrib)
        
        # Call the federated averaging method
        new_weights = aggregator._federated_averaging(contributions)
        
        # Check that new weights have the right shape
        self.assertEqual(len(new_weights), len(self.mock_weights))
        for i in range(len(self.mock_weights)):
            self.assertEqual(new_weights[i].shape, self.mock_weights[i].shape)
        
        # Expected result: weighted average of updates
        total_samples = sum(c['metrics']['dataset_size'] for c in contributions)
        expected_result = []
        
        for i in range(len(self.mock_weights)):
            # Start with global weights
            expected = self.mock_weights[i].copy()
            
            # Add weighted updates
            for c in contributions:
                update = c['model_update']['weights'][i]
                weight = c['metrics']['dataset_size'] / total_samples
                expected += update * weight
            
            expected_result.append(expected)
        
        # Check that the result matches the expected
        for i in range(len(new_weights)):
            np.testing.assert_array_almost_equal(new_weights[i], expected_result[i])
    
    @patch('tensorflow.keras.models.load_model')
    @patch('tensorflow.keras.Sequential')
    def test_weighted_averaging(self, mock_sequential, mock_load_model):
        """Test the weighted averaging (by accuracy) aggregation method."""
        # Mock model building
        mock_model = Mock()
        mock_model.weights = [tf.Variable(w) for w in self.mock_weights]
        mock_sequential.return_value = mock_model
        mock_load_model.side_effect = FileNotFoundError  # Force new model creation
        
        # Initialize aggregator with weighted_average method
        config = self.config.copy()
        config['aggregation'] = config['aggregation'].copy()
        config['aggregation']['method'] = 'weighted_average'
        
        fd, config_path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            yaml.dump(config, f)
        
        aggregator = GlobalAggregator(config_path=config_path)
        
        # Create test contributions list
        contributions = []
        for contrib_data in self.mock_contributions:
            # Add required fields for the weighted averaging function
            contrib = {
                'model_update': contrib_data['model_update'],
                'metrics': contrib_data['metrics']
            }
            contributions.append(contrib)
        
        # Call the weighted averaging method
        new_weights = aggregator._weighted_averaging(contributions)
        
        # Check that new weights have the right shape
        self.assertEqual(len(new_weights), len(self.mock_weights))
        for i in range(len(self.mock_weights)):
            self.assertEqual(new_weights[i].shape, self.mock_weights[i].shape)
        
        # Clean up
        os.remove(config_path)
    
    @patch('tensorflow.keras.models.load_model')
    @patch('tensorflow.keras.Sequential')
    def test_median_aggregation(self, mock_sequential, mock_load_model):
        """Test the median-based aggregation method."""
        # Mock model building
        mock_model = Mock()
        mock_model.weights = [tf.Variable(w) for w in self.mock_weights]
        mock_sequential.return_value = mock_model
        mock_load_model.side_effect = FileNotFoundError  # Force new model creation
        
        # Initialize aggregator with median method
        config = self.config.copy()
        config['aggregation'] = config['aggregation'].copy()
        config['aggregation']['method'] = 'median'
        
        fd, config_path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            yaml.dump(config, f)
        
        aggregator = GlobalAggregator(config_path=config_path)
        
        # Create test contributions list
        contributions = []
        for contrib_data in self.mock_contributions:
            # Add required fields for the median aggregation function
            contrib = {
                'model_update': contrib_data['model_update'],
                'metrics': contrib_data['metrics']
            }
            contributions.append(contrib)
        
        # Call the median aggregation method
        new_weights = aggregator._median_aggregation(contributions)
        
        # Check that new weights have the right shape
        self.assertEqual(len(new_weights), len(self.mock_weights))
        for i in range(len(self.mock_weights)):
            self.assertEqual(new_weights[i].shape, self.mock_weights[i].shape)
        
        # Clean up
        os.remove(config_path)
    
    @patch('tensorflow.keras.models.load_model')
    @patch('tensorflow.keras.Sequential')
    @patch.object(GlobalAggregator, '_record_aggregation_on_blockchain')
    def test_aggregate_contributions(self, mock_record_aggregation, mock_sequential, mock_load_model):
        """Test the complete aggregation process."""
        # Mock model building
        mock_model = Mock()
        mock_model.weights = [tf.Variable(w) for w in self.mock_weights]
        mock_sequential.return_value = mock_model
        mock_load_model.side_effect = FileNotFoundError  # Force new model creation
        
        # Mock model save
        with patch('tensorflow.keras.models.Sequential.save'):
            # Initialize aggregator
            aggregator = GlobalAggregator(config_path=self.config_path)
            
            # Receive all contributions
            contribution_ids = []
            for contrib_data in self.mock_contributions:
                result = aggregator.receive_contribution(
                    client_id=contrib_data['client_id'],
                    metrics=contrib_data['metrics'],
                    model_update=contrib_data['model_update']
                )
                contribution_ids.append(result['contribution_id'])
            
            # Call aggregation manually
            result = aggregator.aggregate_contributions()
            
            # Check aggregation result
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['round'], 1)  # Round should be incremented
            self.assertEqual(result['contributions_included'], 3)
            
            # Check that contributions were marked as included
            for contrib_id in contribution_ids:
                self.assertTrue(aggregator.contributions[contrib_id]['included_in_aggregation'])
            
            # Check that round and version were updated
            self.assertEqual(aggregator.current_round, 1)
            self.assertEqual(aggregator.model_version, "0.0.1")  # Should not change for first round
            
            # Check that blockchain recording was called
            mock_record_aggregation.assert_called_once()
    
    @patch('tensorflow.keras.models.load_model')
    @patch('tensorflow.keras.Sequential')
    def test_model_version_increment(self, mock_sequential, mock_load_model):
        """Test that model version is incremented after rounds_per_epoch rounds."""
        # Mock model building
        mock_model = Mock()
        mock_model.weights = [tf.Variable(w) for w in self.mock_weights]
        mock_sequential.return_value = mock_model
        mock_load_model.side_effect = FileNotFoundError  # Force new model creation
        
        # Mock model save
        with patch('tensorflow.keras.models.Sequential.save'):
            # Initialize aggregator with smaller rounds_per_epoch
            config = self.config.copy()
            config['aggregation'] = config['aggregation'].copy()
            config['aggregation']['rounds_per_epoch'] = 2  # Change to 2 for testing
            
            fd, config_path = tempfile.mkstemp(suffix='.yaml')
            with os.fdopen(fd, 'w') as f:
                yaml.dump(config, f)
            
            aggregator = GlobalAggregator(config_path=config_path)
            
            # Set up initial state
            aggregator.current_round = 1  # Start at round 1
            
            # Need to have some contributions ready
            for contrib_data in self.mock_contributions:
                result = aggregator.receive_contribution(
                    client_id=contrib_data['client_id'],
                    metrics=contrib_data['metrics'],
                    model_update=contrib_data['model_update']
                )
            
            # Call aggregation to increment round to 2, which should trigger version increment
            result = aggregator.aggregate_contributions()
            
            # Check that version was incremented
            self.assertEqual(aggregator.current_round, 2)
            self.assertEqual(aggregator.model_version, "0.1.0")  # Minor version increment
            
            # Clean up
            os.remove(config_path)

if __name__ == '__main__':
    unittest.main()
