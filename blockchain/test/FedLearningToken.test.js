"""
Unit Tests for Federated Learning Client Library

This module tests the client library that interacts with the decentralized
federated learning platform, including authentication, model downloading,
and contribution submission.
"""

import unittest
import os
import json
import asyncio
import tempfile
import base64
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock, AsyncMock
import io
import eth_account

# Import the module to test - adjust path as needed
from client.fed_learning_client import FederatedLearningClient

class TestFederatedLearningClient(unittest.TestCase):
    """Test cases for the FederatedLearningClient class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Set up API URL
        self.api_url = "http://localhost:8000"
        
        # Create client instance
        self.client = FederatedLearningClient(
            api_url=self.api_url,
            client_id="test_client_123"
        )
        
        # Set up test Ethereum wallet
        self.test_private_key = "0x" + "1" * 64  # Mock private key
        self.test_wallet_address = "0x1234567890123456789012345678901234567890"
        
        # Mock requests
        self.setup_requests_mock()
        
        # Set up test model weights
        self.test_weights = [
            np.ones((10, 10)),
            np.ones(10),
            np.ones((10, 2)),
            np.ones(2)
        ]
    
    def setup_requests_mock(self):
        """Set up mock for requests module."""
        self.requests_patcher = patch('client.fed_learning_client.requests')
        self.mock_requests = self.requests_patcher.start()
        
        # Mock response for successful requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "mock_jwt_token",
            "token_type": "bearer",
            "expires_in": 86400,
            "status": "success",
            "contribution_id": "test_contribution_123",
            "verified": True,
            "current_round": 1,
            "model_version": "0.1.0",
            "balance": str(1000 * 10**18),
            "wallet_address": self.test_wallet_address,
            "weights_base64": self._encode_test_weights(),
            "last_updated": "2023-01-01T00:00:00Z",
            "total_contributions": 10,
            "active_clients": 5,
            "total_rewards_issued": "5000000000000000000000"  # 5000 tokens
        }
        self.mock_requests.post.return_value = mock_response
        self.mock_requests.get.return_value = mock_response
    
    def tearDown(self):
        """Clean up after each test."""
        # Stop patches
        if hasattr(self, 'requests_patcher'):
            self.requests_patcher.stop()
        
        # Clean up any temporary files
        if hasattr(self, 'temp_files'):
            for file_path in self.temp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    def _encode_test_weights(self):
        """Encode test weights to base64 for mocking API responses."""
        buffer = io.BytesIO()
        np.savez(buffer, *self.test_weights)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    @patch('client.fed_learning_client.eth_account.Account.from_key')
    def test_load_ethereum_wallet(self, mock_from_key):
        """Test loading an Ethereum wallet from private key."""
        # Mock eth_account.Account.from_key
        mock_account = MagicMock()
        mock_account.address = self.test_wallet_address
        mock_from_key.return_value = mock_account
        
        # Load wallet
        address = self.client.load_ethereum_wallet(self.test_private_key)
        
        # Check that the wallet was loaded
        self.assertEqual(address, self.test_wallet_address)
        self.assertEqual(self.client.wallet.address, self.test_wallet_address)
        
        # Check that from_key was called with the correct key
        mock_from_key.assert_called_once()
        call_args = mock_from_key.call_args[0]
        # Private key should have 0x prefix
        self.assertTrue(call_args[0].startswith('0x'))
    
    @patch('client.fed_learning_client.eth_account.Account.from_key')
    @patch('client.fed_learning_client.encode_defunct')
    async def test_authenticate(self, mock_encode_defunct, mock_from_key):
        """Test authentication with wallet signature."""
        # Mock account
        mock_account = MagicMock()
        mock_account.address = self.test_wallet_address
        mock_account.sign_message.return_value.signature = b'0x123456'
        mock_from_key.return_value = mock_account
        
        # Mock Web3.to_hex
        with patch('client.fed_learning_client.Web3.to_hex', return_value='0x123456'):
            # Load wallet
            self.client.load_ethereum_wallet(self.test_private_key)
            
            # Authenticate
            result = await self.client.authenticate()
            
            # Check that authentication was successful
            self.assertTrue(result)
            self.assertEqual(self.client.access_token, "mock_jwt_token")
            
            # Check that sign_message was called
            mock_account.sign_message.assert_called_once()
            
            # Check that requests.post was called with the correct data
            self.mock_requests.post.assert_called_once()
            call_args = self.mock_requests.post.call_args
            self.assertEqual(call_args[0][0], f"{self.api_url}/api/login")
            self.assertEqual(call_args[1]['json']['wallet_address'], self.test_wallet_address)
            self.assertEqual(call_args[1]['json']['signed_message'], '0x123456')
    
    async def test_get_authorization_headers(self):
        """Test getting authorization headers."""
        # Set access token
        self.client.access_token = "test_token"
        
        # Get headers
        headers = self.client.get_authorization_headers()
        
        # Check headers
        self.assertEqual(headers, {"Authorization": "Bearer test_token"})
        
        # Test without access token
        self.client.access_token = None
        with self.assertRaises(ValueError):
            self.client.get_authorization_headers()
    
    @patch('client.fed_learning_client.eth_account.Account.from_key')
    async def test_get_global_model(self, mock_from_key):
        """Test downloading the global model."""
        # Mock account
        mock_account = MagicMock()
        mock_account.address = self.test_wallet_address
        mock_from_key.return_value = mock_account
        
        # Load wallet and set access token
        self.client.load_ethereum_wallet(self.test_private_key)
        self.client.access_token = "test_token"
        
        # Mock _decode_model_weights
        with patch.object(self.client, '_decode_model_weights', return_value=self.test_weights):
            # Get global model
            weights, model_info = await self.client.get_global_model()
            
            # Check that requests.get was called with the correct URL and headers
            self.mock_requests.get.assert_called_once()
            call_args = self.mock_requests.get.call_args
            self.assertEqual(call_args[0][0], f"{self.api_url}/api/model")
            self.assertEqual(call_args[1]['headers'], {"Authorization": "Bearer test_token"})
            
            # Check results
            self.assertEqual(model_info['model_version'], "0.1.0")
            self.assertEqual(model_info['current_round'], 1)
            self.assertEqual(len(weights), len(self.test_weights))
            for i in range(len(weights)):
                np.testing.assert_array_equal(weights[i], self.test_weights[i])
    
    def test_encode_decode_model_update(self):
        """Test encoding and decoding model updates."""
        # Create a model update
        model_update = [w * 0.1 for w in self.test_weights]
        
        # Encode model update
        encoded = self.client._encode_model_update(model_update)
        
        # Check that encoding produced a base64 string
        self.assertIsInstance(encoded, str)
        self.assertTrue(len(encoded) > 0)
        
        # Decode model update
        decoded = self.client._decode_model_weights(encoded)
        
        # Check that decoding produced the original weights
        self.assertEqual(len(decoded), len(model_update))
        for i in range(len(decoded)):
            np.testing.assert_array_almost_equal(decoded[i], model_update[i])
    
    @patch('client.fed_learning_client.eth_account.Account.from_key')
    async def test_submit_contribution(self, mock_from_key):
        """Test submitting a training contribution."""
        # Mock account
        mock_account = MagicMock()
        mock_account.address = self.test_wallet_address
        mock_from_key.return_value = mock_account
        
        # Load wallet and set access token
        self.client.load_ethereum_wallet(self.test_private_key)
        self.client.access_token = "test_token"
        
        # Create model update and metrics
        model_update = [w * 0.1 for w in self.test_weights]
        metrics = {
            "loss": 0.5,
            "accuracy": 0.85,
            "dataset_size": 1000
        }
        
        # Mock _encode_model_update
        with patch.object(self.client, '_encode_model_update', return_value="encoded_update"):
            # Submit contribution
            result = await self.client.submit_contribution(model_update, metrics)
            
            # Check that requests.post was called with the correct data
            self.mock_requests.post.assert_called_once()
            call_args = self.mock_requests.post.call_args
            self.assertEqual(call_args[0][0], f"{self.api_url}/api/contributions")
            self.assertEqual(call_args[1]['headers'], {"Authorization": "Bearer test_token"})
            self.assertEqual(call_args[1]['json']['client_id'], self.client.client_id)
            self.assertEqual(call_args[1]['json']['metrics'], metrics)
            self.assertEqual(call_args[1]['json']['model_update_base64'], "encoded_update")
            
            # Check results
            self.assertEqual(result['status'], "success")
            self.assertEqual(result['contribution_id'], "test_contribution_123")
            self.assertEqual(result['verified'], True)
            self.assertEqual(result['current_round'], 1)
            self.assertEqual(result['model_version'], "0.1.0")
    
    @patch('client.fed_learning_client.eth_account.Account.from_key')
    async def test_get_token_balance(self, mock_from_key):
        """Test getting token balance."""
        # Mock account
        mock_account = MagicMock()
        mock_account.address = self.test_wallet_address
        mock_from_key.return_value = mock_account
        
        # Load wallet and set access token
        self.client.load_ethereum_wallet(self.test_private_key)
        self.client.access_token = "test_token"
        
        # Get token balance
        balance = await self.client.get_token_balance()
        
        # Check that requests.get was called with the correct URL and headers
        self.mock_requests.get.assert_called_once()
        call_args = self.mock_requests.get.call_args
        self.assertEqual(call_args[0][0], f"{self.api_url}/api/token/balance")
        self.assertEqual(call_args[1]['headers'], {"Authorization": "Bearer test_token"})
        
        # Check result
        self.assertEqual(balance, str(1000 * 10**18))
    
    async def test_get_platform_stats(self):
        """Test getting platform statistics."""
        # Get platform stats
        stats = await self.client.get_platform_stats()
        
        # Check that requests.get was called with the correct URL
        self.mock_requests.get.assert_called_once()
        call_args = self.mock_requests.get.call_args
        self.assertEqual(call_args[0][0], f"{self.api_url}/api/stats")
        
        # Check results
        self.assertEqual(stats['total_contributions'], 10)
        self.assertEqual(stats['active_clients'], 5)
        self.assertEqual(stats['current_round'], 1)
        self.assertEqual(stats['model_version'], "0.1.0")
        self.assertEqual(stats['total_rewards_issued'], "5000000000000000000000")
    
    @patch('client.fed_learning_client.eth_account.Account.from_key')
    async def test_train_and_contribute(self, mock_from_key):
        """Test the high-level train and contribute function."""
        # Mock account
        mock_account = MagicMock()
        mock_account.address = self.test_wallet_address
        mock_from_key.return_value = mock_account
        
        # Load wallet and set access token
        self.client.load_ethereum_wallet(self.test_private_key)
        self.client.access_token = "test_token"
        
        # Define a mock training function
        async def mock_train_function(weights, model_info, custom_data=None):
            # Simple mock that returns small updates and metrics
            return [w * 0.1 for w in weights], {"loss": 0.5, "accuracy": 0.85, "dataset_size": 1000}
        
        # Patch internal methods
        with patch.object(self.client, 'get_global_model', return_value=(self.test_weights, {"model_version": "0.1.0", "current_round": 1})), \
             patch.object(self.client, 'submit_contribution', return_value={"status": "success", "contribution_id": "test_contribution_123"}):
            
            # Train and contribute
            result = await self.client.train_and_contribute(mock_train_function)
            
            # Check result
            self.assertEqual(result['status'], "success")
            self.assertEqual(result['contribution_id'], "test_contribution_123")
    
    @patch('client.fed_learning_client.eth_account.Account.from_key')
    async def test_vote_on_governance_proposal(self, mock_from_key):
        """Test voting on a governance proposal."""
        # Mock account
        mock_account = MagicMock()
        mock_account.address = self.test_wallet_address
        mock_from_key.return_value = mock_account
        
        # Load wallet and set access token
        self.client.load_ethereum_wallet(self.test_private_key)
        self.client.access_token = "test_token"
        
        # Vote on proposal
        result = await self.client.vote_on_governance_proposal(1, True)
        
        # Check that requests.post was called with the correct data
        self.mock_requests.post.assert_called_once()
        call_args = self.mock_requests.post.call_args
        self.assertEqual(call_args[0][0], f"{self.api_url}/api/governance/vote")
        self.assertEqual(call_args[1]['headers'], {"Authorization": "Bearer test_token"})
        self.assertEqual(call_args[1]['json']['proposal_id'], 1)
        self.assertEqual(call_args[1]['json']['in_support'], True)
        
        # Check result
        self.assertEqual(result['status'], "success")
    
    @patch('client.fed_learning_client.eth_account.Account.from_key')
    async def test_create_governance_proposal(self, mock_from_key):
        """Test creating a governance proposal."""
        # Mock account
        mock_account = MagicMock()
        mock_account.address = self.test_wallet_address
        mock_from_key.return_value = mock_account
        
        # Load wallet and set access token
        self.client.load_ethereum_wallet(self.test_private_key)
        self.client.access_token = "test_token"
        
        # Create proposal
        result = await self.client.create_governance_proposal(
            title="Test Proposal",
            description="This is a test proposal",
            target_contract="0x9876543210987654321098765432109876543210",
            call_data="0x123456789abcdef"
        )
        
        # Check that requests.post was called with the correct data
        self.mock_requests.post.assert_called_once()
        call_args = self.mock_requests.post.call_args
        self.assertEqual(call_args[0][0], f"{self.api_url}/api/governance/propose")
        self.assertEqual(call_args[1]['headers'], {"Authorization": "Bearer test_token"})
        self.assertEqual(call_args[1]['json']['title'], "Test Proposal")
        self.assertEqual(call_args[1]['json']['description'], "This is a test proposal")
        self.assertEqual(call_args[1]['json']['target_contract'], "0x9876543210987654321098765432109876543210")
        self.assertEqual(call_args[1]['json']['call_data'], "0x123456789abcdef")
        
        # Check result
        self.assertEqual(result['status'], "success")
    
    async def test_error_handling(self):
        """Test error handling in API calls."""
        # Mock error response
        error_response = MagicMock()
        error_response.status_code = 400
        error_response.text = "Bad request"
        self.mock_requests.get.return_value = error_response
        
        # No authentication token
        with self.assertRaises(ValueError):
            await self.client.get_global_model()
        
        # Set access token to test API error
        self.client.access_token = "test_token"
        
        # API error
        with self.assertRaises(ValueError):
            await self.client.get_global_model()

if __name__ == '__main__':
    unittest.main()
