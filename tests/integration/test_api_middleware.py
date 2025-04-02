"""
Integration Tests for API-Middleware Interaction

This module tests the integration between the API server and blockchain middleware
components of the decentralized federated learning platform.
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
from fastapi.testclient import TestClient
import jwt
from datetime import datetime, timedelta

# Import components to test - adjust paths as needed
from middleware.api.routes import app
from middleware.blockchain_middleware import BlockchainMiddleware
from middleware.orchestration import OrchestratorService
from aggregator.global_aggregator import GlobalAggregator

class TestAPIMiddlewareIntegration(unittest.TestCase):
    """Test cases for the API-Middleware integration."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create test client
        self.client = TestClient(app)
        
        # Mock JWT settings
        self.jwt_secret = "test_secret_key"
        self.jwt_algorithm = "HS256"
        
        # Create mock wallet and authentication token
        self.test_wallet_address = "0x1234567890123456789012345678901234567890"
        self.test_jwt = self._create_test_jwt(self.test_wallet_address)
        
        # Create sample model weights for testing
        self.test_weights = [
            np.ones((10, 10)),
            np.ones(10),
            np.ones((10, 2)),
            np.ones(2)
        ]
        
        # Create headers with authentication
        self.auth_headers = {"Authorization": f"Bearer {self.test_jwt}"}
        
        # Set up patches for dependencies
        self.mock_global_aggregator = self._setup_mock_aggregator()
        self.mock_blockchain_middleware = self._setup_mock_blockchain_middleware()
        
        # Apply patches
        self.aggregator_patcher = patch('middleware.api.routes.global_aggregator', self.mock_global_aggregator)
        self.blockchain_patcher = patch('middleware.api.routes.blockchain_middleware', self.mock_blockchain_middleware)
        self.jwt_secret_patcher = patch('middleware.api.routes.JWT_SECRET_KEY', self.jwt_secret)
        self.jwt_algo_patcher = patch('middleware.api.routes.JWT_ALGORITHM', self.jwt_algorithm)
        
        self.aggregator_patcher.start()
        self.blockchain_patcher.start()
        self.jwt_secret_patcher.start()
        self.jwt_algo_patcher.start()
    
    def tearDown(self):
        """Clean up after each test."""
        # Stop patches
        self.aggregator_patcher.stop()
        self.blockchain_patcher.stop()
        self.jwt_secret_patcher.stop()
        self.jwt_algo_patcher.stop()
    
    def _create_test_jwt(self, wallet_address):
        """Create a test JWT token."""
        # Define token data
        data = {
            "sub": wallet_address,
            "exp": datetime.utcnow() + timedelta(minutes=30)
        }
        
        # Create token
        token = jwt.encode(data, self.jwt_secret, algorithm=self.jwt_algorithm)
        return token
    
    def _setup_mock_aggregator(self):
        """Set up mock global aggregator."""
        mock_aggregator = MagicMock()
        
        # Mock receive_contribution
        mock_aggregator.receive_contribution.return_value = {
            "status": "success",
            "contribution_id": "test_contribution_123",
            "verified": True,
            "current_round": 1,
            "model_version": "0.1.0"
        }
        
        # Mock get_global_model_weights
        mock_aggregator.get_global_model_weights.return_value = self.test_weights
        
        # Mock get_global_model_info
        mock_aggregator.get_global_model_info.return_value = {
            "model_version": "0.1.0",
            "current_round": 1,
            "architecture": "mlp",
            "input_shape": [10],
            "aggregation_method": "fedavg",
            "last_updated": 1234567890
        }
        
        # Mock get_contribution_stats
        mock_aggregator.get_contribution_stats.return_value = {
            "total_contributions": 10,
            "verified_contributions": 8,
            "included_contributions": 5,
            "by_round": {0: 3, 1: 7},
            "by_client": {"client_1": 5, "client_2": 5},
            "current_round": 1
        }
        
        return mock_aggregator
    
    def _setup_mock_blockchain_middleware(self):
        """Set up mock blockchain middleware."""
        mock_middleware = MagicMock()
        
        # Mock async methods
        mock_middleware.log_contribution = AsyncMock(return_value="0x123...")
        mock_middleware.verify_contribution_quality = AsyncMock(return_value="0x456...")
        mock_middleware.issue_reward = AsyncMock(return_value="0x789...")
        mock_middleware.get_token_balance = AsyncMock(return_value=1000 * 10**18)
        mock_middleware.cast_vote = AsyncMock(return_value="0xabc...")
        mock_middleware.create_governance_proposal = AsyncMock(return_value=5)
        
        return mock_middleware
    
    def _encode_model_update(self, model_update):
        """Encode model update to base64 string for API testing."""
        try:
            # Use BytesIO instead of temporary file
            import io
            buffer = io.BytesIO()
            np.savez(buffer, *model_update)
            buffer.seek(0)
            
            # Encode to base64
            base64_data = base64.b64encode(buffer.read()).decode('utf-8')
            return base64_data
        except Exception as e:
            self.fail(f"Failed to encode model update: {e}")
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
    
    def test_login_success(self):
        """Test successful login with valid wallet signature."""
        # Mock the verify_ethereum_signature function to return True
        with patch('middleware.api.routes.verify_ethereum_signature', return_value=True):
            response = self.client.post(
                "/api/login",
                json={
                    "wallet_address": self.test_wallet_address,
                    "signed_message": "0x123..."  # Mock signature
                }
            )
            
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn("access_token", data)
            self.assertEqual(data["token_type"], "bearer")
            self.assertGreater(data["expires_in"], 0)
    
    def test_login_failure(self):
        """Test login failure with invalid signature."""
        # Mock the verify_ethereum_signature function to return False
        with patch('middleware.api.routes.verify_ethereum_signature', return_value=False):
            response = self.client.post(
                "/api/login",
                json={
                    "wallet_address": self.test_wallet_address,
                    "signed_message": "0xINVALID"  # Invalid signature
                }
            )
            
            self.assertEqual(response.status_code, 401)
    
    def test_get_global_model(self):
        """Test retrieving the global model."""
        response = self.client.get(
            "/api/model",
            headers=self.auth_headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["model_version"], "0.1.0")
        self.assertEqual(data["current_round"], 1)
        self.assertIn("weights_base64", data)
        self.assertIn("last_updated", data)
        
        # Verify aggregator method was called
        self.mock_global_aggregator.get_global_model_weights.assert_called_once()
        self.mock_global_aggregator.get_global_model_info.assert_called_once()
    
    def test_get_stats(self):
        """Test retrieving platform statistics."""
        response = self.client.get("/api/stats")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["total_contributions"], 10)
        self.assertEqual(data["active_clients"], 2)  # From by_client count
        self.assertEqual(data["current_round"], 1)
        self.assertEqual(data["model_version"], "0.1.0")
        
        # Verify aggregator method was called
        self.mock_global_aggregator.get_contribution_stats.assert_called_once()
        self.mock_global_aggregator.get_global_model_info.assert_called_once()
    
    def test_get_token_balance(self):
        """Test retrieving token balance."""
        response = self.client.get(
            "/api/token/balance",
            headers=self.auth_headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["balance"], str(1000 * 10**18))
        self.assertEqual(data["wallet_address"], self.test_wallet_address)
        
        # Verify blockchain middleware method was called
        self.mock_blockchain_middleware.get_token_balance.assert_called_once_with(self.test_wallet_address)
    
    def test_submit_contribution(self):
        """Test submitting a model update contribution."""
        # Encode model update
        model_update = [w * 0.1 for w in self.test_weights]  # Small updates
        model_update_base64 = self._encode_model_update(model_update)
        
        # Prepare contribution data
        contribution_data = {
            "client_id": "test_client_1",
            "metrics": {
                "loss": 0.5,
                "accuracy": 0.85,
                "dataset_size": 1000
            },
            "model_update_base64": model_update_base64
        }
        
        # Submit contribution
        response = self.client.post(
            "/api/contributions",
            json=contribution_data,
            headers=self.auth_headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["verified"], True)
        self.assertEqual(data["current_round"], 1)
        self.assertEqual(data["model_version"], "0.1.0")
        
        # Verify aggregator method was called with correct client_id and metrics
        self.mock_global_aggregator.receive_contribution.assert_called_once()
        call_args = self.mock_global_aggregator.receive_contribution.call_args[1]
        self.assertEqual(call_args["client_id"], "test_client_1")
        self.assertEqual(call_args["metrics"], contribution_data["metrics"])
        
        # Background tasks are harder to test directly in FastAPI
        # In a real test, we might need to use pytest-asyncio and inspect the task queue
    
    def test_vote_on_proposal(self):
        """Test voting on a governance proposal."""
        vote_data = {
            "proposal_id": 1,
            "in_support": True
        }
        
        response = self.client.post(
            "/api/governance/vote",
            json=vote_data,
            headers=self.auth_headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["proposal_id"], 1)
        
        # Background tasks are harder to test directly in FastAPI
        # In a real test, we'd need to ensure the task was queued properly
    
    def test_create_proposal(self):
        """Test creating a governance proposal."""
        proposal_data = {
            "title": "Test Proposal",
            "description": "This is a test proposal",
            "target_contract": "0x9876543210987654321098765432109876543210",
            "call_data": "0x123456789abcdef"
        }
        
        response = self.client.post(
            "/api/governance/propose",
            json=proposal_data,
            headers=self.auth_headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["proposal_id"], 123)  # Mock value from routes.py
        
        # Background tasks are harder to test directly in FastAPI
        # In a real test, we'd need to ensure the task was queued properly
    
    def test_unauthorized_access(self):
        """Test that protected endpoints reject unauthorized requests."""
        # Try to get model without auth
        response = self.client.get("/api/model")
        self.assertEqual(response.status_code, 403)
        
        # Try to get token balance without auth
        response = self.client.get("/api/token/balance")
        self.assertEqual(response.status_code, 403)
        
        # Try to submit contribution without auth
        contribution_data = {
            "client_id": "test_client_1",
            "metrics": {
                "loss": 0.5,
                "accuracy": 0.85,
                "dataset_size": 1000
            },
            "model_update_base64": "invalid"
        }
        
        response = self.client.post(
            "/api/contributions",
            json=contribution_data
        )
        self.assertEqual(response.status_code, 403)

if __name__ == '__main__':
    unittest.main()
