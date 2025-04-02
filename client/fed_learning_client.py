"""
Federated Learning Client Library

This module provides a client library for interacting with the decentralized
federated learning platform. It handles authentication, model downloading,
local training, and contribution submission.
"""

import os
import requests
import json
import base64
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
import eth_account
from eth_account.messages import encode_defunct
from web3 import Web3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FederatedLearningClient:
    """
    Client library for decentralized federated learning platform.
    """
    
    def __init__(self, api_url: str, client_id: Optional[str] = None):
        """
        Initialize the client with API URL and optional client ID.
        
        Args:
            api_url: Base URL for the API
            client_id: Unique identifier for this client (generated if None)
        """
        self.api_url = api_url.rstrip('/')
        self.client_id = client_id or f"client_{int(time.time())}_{os.urandom(4).hex()}"
        self.access_token = None
        self.wallet = None
        
        logger.info(f"Initialized FederatedLearningClient with ID: {self.client_id}")
    
    def load_ethereum_wallet(self, private_key: str) -> str:
        """
        Load an Ethereum wallet from private key.
        
        Args:
            private_key: Ethereum private key (with or without '0x' prefix)
            
        Returns:
            Wallet address
        """
        try:
            # Add '0x' prefix if missing
            if not private_key.startswith('0x'):
                private_key = f"0x{private_key}"
            
            # Create account
            self.wallet = eth_account.Account.from_key(private_key)
            logger.info(f"Loaded wallet with address: {self.wallet.address}")
            
            return self.wallet.address
        except Exception as e:
            logger.error(f"Failed to load wallet: {e}")
            raise
    
    async def authenticate(self) -> bool:
        """
        Authenticate with the API using wallet signature.
        
        Returns:
            True if authentication was successful
        """
        if not self.wallet:
            raise ValueError("Wallet not loaded. Call load_ethereum_wallet() first.")
        
        try:
            # Create message to sign
            timestamp = int(time.time())
            message = f"Authenticate with Federated Learning Platform\nClient ID: {self.client_id}\nTimestamp: {timestamp}"
            
            # Sign message
            message_hash = encode_defunct(text=message)
            signed_message = self.wallet.sign_message(message_hash)
            signature_hex = Web3.to_hex(signed_message.signature)
            
            # Prepare request data
            login_data = {
                "wallet_address": self.wallet.address,
                "signed_message": signature_hex
            }
            
            # Send login request
            response = requests.post(
                f"{self.api_url}/api/login",
                json=login_data
            )
            
            if response.status_code != 200:
                logger.error(f"Authentication failed: {response.text}")
                return False
            
            # Parse response
            data = response.json()
            self.access_token = data["access_token"]
            
            logger.info("Authentication successful")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def get_authorization_headers(self) -> Dict[str, str]:
        """
        Get headers for authenticated requests.
        
        Returns:
            Dictionary of HTTP headers
        """
        if not self.access_token:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        return {
            "Authorization": f"Bearer {self.access_token}"
        }
    
    async def get_global_model(self) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Download the latest global model.
        
        Returns:
            Tuple of (model_weights, model_info)
        """
        try:
            # Get authorization headers
            headers = self.get_authorization_headers()
            
            # Send request
            response = requests.get(
                f"{self.api_url}/api/model",
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get global model: {response.text}")
                raise ValueError(f"Failed to get global model: {response.text}")
            
            # Parse response
            data = response.json()
            
            # Extract model info
            model_info = {
                "model_version": data["model_version"],
                "current_round": data["current_round"],
                "last_updated": data["last_updated"]
            }
            
            # Decode model weights
            weights_base64 = data["weights_base64"]
            weights = self._decode_model_weights(weights_base64)
            
            logger.info(f"Downloaded global model version {model_info['model_version']}, round {model_info['current_round']}")
            
            return weights, model_info
        except Exception as e:
            logger.error(f"Failed to get global model: {e}")
            raise
    
    def _decode_model_weights(self, base64_data: str) -> List[np.ndarray]:
        """
        Decode base64 model weights back to numpy arrays.
        
        Args:
            base64_data: Base64 encoded model weights
            
        Returns:
            List of numpy arrays representing model weights
        """
        try:
            # Decode base64 to bytes
            binary_data = base64.b64decode(base64_data)
            
            # Load from bytes (assuming numpy .npz format)
            with np.load(binary_data) as data:
                # Convert file-like object to list of arrays
                arrays = [data[key] for key in data.files]
            
            return arrays
        except Exception as e:
            logger.error(f"Failed to decode model weights: {e}")
            raise ValueError(f"Failed to decode model weights: {e}")
    
    def _encode_model_update(self, model_update: List[np.ndarray]) -> str:
        """
        Encode model update to base64 string.
        
        Args:
            model_update: List of numpy arrays representing model update
            
        Returns:
            Base64 encoded model update
        """
        try:
            # Save numpy arrays to in-memory file
            with np.memmap('.temp.npz', dtype='float32', mode='w+', shape=(1,)) as temp:
                np.savez(temp, *model_update)
            
            # Read the file and encode to base64
            with open('.temp.npz', 'rb') as f:
                binary_data = f.read()
            
            # Remove temporary file
            os.remove('.temp.npz')
            
            # Encode to base64
            base64_data = base64.b64encode(binary_data).decode('utf-8')
            
            return base64_data
        except Exception as e:
            logger.error(f"Failed to encode model update: {e}")
            raise ValueError(f"Failed to encode model update: {e}")
    
    async def submit_contribution(
        self,
        model_update: List[np.ndarray],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit a training contribution to the platform.
        
        Args:
            model_update: List of numpy arrays representing model update
            metrics: Dictionary of training metrics
            
        Returns:
            Dictionary with submission status and details
        """
        try:
            # Get authorization headers
            headers = self.get_authorization_headers()
            
            # Encode model update
            model_update_base64 = self._encode_model_update(model_update)
            
            # Prepare request data
            contribution_data = {
                "client_id": self.client_id,
                "metrics": metrics,
                "model_update_base64": model_update_base64
            }
            
            # Send request
            response = requests.post(
                f"{self.api_url}/api/contributions",
                json=contribution_data,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Contribution submission failed: {response.text}")
                raise ValueError(f"Contribution submission failed: {response.text}")
            
            # Parse response
            data = response.json()
            
            logger.info(f"Contribution submitted successfully: {data['contribution_id']}")
            
            return data
        except Exception as e:
            logger.error(f"Contribution submission failed: {e}")
            raise
    
    async def get_token_balance(self) -> str:
        """
        Get token balance for the authenticated wallet.
        
        Returns:
            Token balance as string
        """
        try:
            # Get authorization headers
            headers = self.get_authorization_headers()
            
            # Send request
            response = requests.get(
                f"{self.api_url}/api/token/balance",
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get token balance: {response.text}")
                raise ValueError(f"Failed to get token balance: {response.text}")
            
            # Parse response
            data = response.json()
            
            logger.info(f"Token balance: {data['balance']}")
            
            return data['balance']
        except Exception as e:
            logger.error(f"Failed to get token balance: {e}")
            raise
    
    async def get_platform_stats(self) -> Dict[str, Any]:
        """
        Get platform statistics.
        
        Returns:
            Dictionary with platform statistics
        """
        try:
            # Send request (no authentication required)
            response = requests.get(f"{self.api_url}/api/stats")
            
            if response.status_code != 200:
                logger.error(f"Failed to get platform stats: {response.text}")
                raise ValueError(f"Failed to get platform stats: {response.text}")
            
            # Parse response
            data = response.json()
            
            return data
        except Exception as e:
            logger.error(f"Failed to get platform stats: {e}")
            raise
    
    async def train_and_contribute(
        self,
        train_function: Callable[[List[np.ndarray], Dict[str, Any]], Tuple[List[np.ndarray], Dict[str, Any]]],
        custom_data: Any = None
    ) -> Dict[str, Any]:
        """
        High-level function to download global model, train locally, and submit contribution.
        
        Args:
            train_function: Function that takes (model_weights, model_info) and returns (model_update, metrics)
            custom_data: Custom data to pass to the train function
            
        Returns:
            Dictionary with contribution details
        """
        try:
            # Download global model
            global_weights, model_info = await self.get_global_model()
            
            # Perform local training
            logger.info("Starting local training")
            model_update, metrics = train_function(global_weights, model_info, custom_data)
            logger.info(f"Local training completed with metrics: {metrics}")
            
            # Submit contribution
            result = await self.submit_contribution(model_update, metrics)
            
            return result
        except Exception as e:
            logger.error(f"Train and contribute failed: {e}")
            raise
    
    async def vote_on_governance_proposal(self, proposal_id: int, in_support: bool) -> Dict[str, Any]:
        """
        Vote on a governance proposal.
        
        Args:
            proposal_id: ID of the proposal to vote on
            in_support: Whether the vote is in support
            
        Returns:
            Dictionary with vote details
        """
        try:
            # Get authorization headers
            headers = self.get_authorization_headers()
            
            # Prepare request data
            vote_data = {
                "proposal_id": proposal_id,
                "in_support": in_support
            }
            
            # Send request
            response = requests.post(
                f"{self.api_url}/api/governance/vote",
                json=vote_data,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Vote submission failed: {response.text}")
                raise ValueError(f"Vote submission failed: {response.text}")
            
            # Parse response
            data = response.json()
            
            logger.info(f"Vote submitted for proposal {proposal_id}")
            
            return data
        except Exception as e:
            logger.error(f"Vote submission failed: {e}")
            raise
    
    async def create_governance_proposal(
        self,
        title: str,
        description: str,
        target_contract: str,
        call_data: str
    ) -> Dict[str, Any]:
        """
        Create a governance proposal.
        
        Args:
            title: Short title of the proposal
            description: Detailed description of the proposal
            target_contract: Address of the contract to call if proposal passes
            call_data: Hex-encoded function call data for execution
            
        Returns:
            Dictionary with proposal details
        """
        try:
            # Get authorization headers
            headers = self.get_authorization_headers()
            
            # Prepare request data
            proposal_data = {
                "title": title,
                "description": description,
                "target_contract": target_contract,
                "call_data": call_data
            }
            
            # Send request
            response = requests.post(
                f"{self.api_url}/api/governance/propose",
                json=proposal_data,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Proposal creation failed: {response.text}")
                raise ValueError(f"Proposal creation failed: {response.text}")
            
            # Parse response
            data = response.json()
            
            logger.info(f"Proposal created with ID {data.get('proposal_id')}")
            
            return data
        except Exception as e:
            logger.error(f"Proposal creation failed: {e}")
            raise

# Example usage
async def example_usage():
    # Initialize client
    client = FederatedLearningClient(
        api_url="http://localhost:8000",
        client_id="example_client_1"
    )
    
    # Load wallet
    client.load_ethereum_wallet("your_private_key_here")
    
    # Authenticate
    await client.authenticate()
    
    # Define a training function
    def train(global_weights, model_info, custom_data=None):
        # This is where you would implement your actual training logic
        print(f"Training on global model version {model_info['model_version']}")
        
        # Create a simple model update (just adding noise to weights)
        model_update = []
        for weight in global_weights:
            # Add small random updates (simulating training)
            update = np.random.normal(0, 0.01, size=weight.shape)
            model_update.append(update)
        
        # Return model update and metrics
        metrics = {
            "loss": 0.1,
            "accuracy": 0.85,
            "dataset_size": 1000
        }
        
        return model_update, metrics
    
    # Train and contribute
    result = await client.train_and_contribute(train)
    print(f"Contribution result: {result}")
    
    # Get token balance
    balance = await client.get_token_balance()
    print(f"Token balance: {balance}")
    
    # Get platform stats
    stats = await client.get_platform_stats()
    print(f"Platform stats: {stats}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
