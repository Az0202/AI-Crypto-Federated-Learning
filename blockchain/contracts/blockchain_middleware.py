"""
Blockchain Middleware Service for Decentralized Federated Learning Platform

This module implements the middleware that connects the federated learning
components with the blockchain infrastructure, handling contract interactions.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_account.signers.local import LocalAccount

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BlockchainMiddleware:
    """
    Middleware service to connect federated learning with blockchain contracts.
    """
    
    def __init__(self, config_path: str = "blockchain_config.json"):
        """
        Initialize the blockchain middleware with configuration.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        self.config = self._load_config(config_path)
        self.w3 = self._initialize_web3()
        self.account = self._load_account()
        self.contracts = self._load_contracts()
        
        logger.info(f"Initialized BlockchainMiddleware, connected to: {self.config['provider_url']}")
    
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
                "provider_url": "http://localhost:8545",
                "chain_id": 1337,
                "gas_limit": 3000000,
                "key_env_var": "PRIVATE_KEY",
                "contracts": {
                    "ContributionLogging": {
                        "address": "0x1234567890123456789012345678901234567890",
                        "abi_path": "./contracts/abi/ContributionLogging.json"
                    },
                    "QualityVerification": {
                        "address": "0x2345678901234567890123456789012345678901",
                        "abi_path": "./contracts/abi/QualityVerification.json"
                    },
                    "RewardDistribution": {
                        "address": "0x3456789012345678901234567890123456789012",
                        "abi_path": "./contracts/abi/RewardDistribution.json"
                    },
                    "FedLearningToken": {
                        "address": "0x4567890123456789012345678901234567890123",
                        "abi_path": "./contracts/abi/FedLearningToken.json"
                    },
                    "FLGovernance": {
                        "address": "0x5678901234567890123456789012345678901234",
                        "abi_path": "./contracts/abi/FLGovernance.json"
                    }
                }
            }
    
    def _initialize_web3(self) -> Web3:
        """
        Initialize Web3 connection to blockchain.
        
        Returns:
            Web3 instance
        """
        try:
            w3 = Web3(HTTPProvider(self.config['provider_url']))
            
            # Add PoA middleware for networks like Polygon, BSC, etc.
            if self.config.get('use_poa_middleware', False):
                w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not w3.is_connected():
                raise ConnectionError(f"Failed to connect to provider at {self.config['provider_url']}")
                
            logger.info(f"Connected to blockchain at {self.config['provider_url']}")
            logger.info(f"Current block number: {w3.eth.block_number}")
            
            return w3
        except Exception as e:
            logger.error(f"Failed to initialize Web3: {e}")
            raise
    
    def _load_account(self) -> LocalAccount:
        """
        Load Ethereum account from private key.
        
        Returns:
            Ethereum account for signing transactions
        """
        try:
            # Get private key from environment variable
            private_key = os.environ.get(self.config['key_env_var'])
            
            if not private_key:
                raise ValueError(f"Private key not found in environment variable {self.config['key_env_var']}")
            
            # Add '0x' prefix if missing
            if not private_key.startswith('0x'):
                private_key = f"0x{private_key}"
            
            account = Account.from_key(private_key)
            logger.info(f"Loaded account: {account.address}")
            
            return account
        except Exception as e:
            logger.error(f"Failed to load account: {e}")
            raise
    
    def _load_contracts(self) -> Dict[str, Any]:
        """
        Load contract ABIs and instances.
        
        Returns:
            Dict of contract instances
        """
        contracts = {}
        
        try:
            for contract_name, contract_info in self.config['contracts'].items():
                address = contract_info['address']
                abi_path = contract_info['abi_path']
                
                # Load ABI from file
                try:
                    with open(abi_path, 'r') as file:
                        contract_json = json.load(file)
                        
                    # Handle different JSON formats
                    if isinstance(contract_json, dict) and 'abi' in contract_json:
                        abi = contract_json['abi']
                    else:
                        abi = contract_json
                        
                    # Create contract instance
                    contract = self.w3.eth.contract(address=address, abi=abi)
                    contracts[contract_name] = contract
                    
                    logger.info(f"Loaded contract {contract_name} at {address}")
                except Exception as e:
                    logger.error(f"Failed to load ABI for {contract_name}: {e}")
            
            return contracts
        except Exception as e:
            logger.error(f"Failed to load contracts: {e}")
            raise
    
    async def send_transaction(self, contract_name: str, function_name: str, *args, **kwargs) -> str:
        """
        Send a transaction to a contract function.
        
        Args:
            contract_name: Name of the contract to call
            function_name: Name of the function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Transaction hash
        """
        try:
            # Get contract instance
            contract = self.contracts.get(contract_name)
            if not contract:
                raise ValueError(f"Contract '{contract_name}' not found")
            
            # Get function
            contract_function = getattr(contract.functions, function_name)
            if not contract_function:
                raise ValueError(f"Function '{function_name}' not found in contract '{contract_name}'")
            
            # Build transaction
            function_call = contract_function(*args, **kwargs)
            
            # Get gas price
            gas_price = self.w3.eth.gas_price
            
            # Add gas multiplier for faster confirmation if specified
            if self.config.get('gas_price_multiplier', 1) > 1:
                gas_price = int(gas_price * self.config['gas_price_multiplier'])
            
            # Estimate gas (with a buffer)
            try:
                gas_estimate = function_call.estimate_gas({'from': self.account.address})
                gas_limit = int(gas_estimate * 1.2)  # Add 20% buffer
            except Exception as e:
                logger.warning(f"Gas estimation failed: {e}. Using default gas limit.")
                gas_limit = self.config['gas_limit']
            
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            # Build transaction dictionary
            tx = function_call.build_transaction({
                'from': self.account.address,
                'gas': gas_limit,
                'gasPrice': gas_price,
                'nonce': nonce,
                'chainId': self.config['chain_id']
            })
            
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.account.key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            hex_tx_hash = self.w3.to_hex(tx_hash)
            
            logger.info(f"Transaction sent: {hex_tx_hash}")
            
            # Wait for receipt if configured
            if self.config.get('wait_for_receipt', True):
                receipt = await self._wait_for_transaction_receipt(hex_tx_hash)
                if receipt['status'] == 0:
                    logger.error(f"Transaction failed: {hex_tx_hash}")
                    raise ValueError(f"Transaction failed: {hex_tx_hash}")
                else:
                    logger.info(f"Transaction confirmed in block {receipt['blockNumber']}")
            
            return hex_tx_hash
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
            raise
    
    async def _wait_for_transaction_receipt(self, tx_hash: str, timeout: int = 120, poll_interval: int = 0.1) -> Dict[str, Any]:
        """
        Wait for transaction receipt with timeout.
        
        Args:
            tx_hash: Transaction hash
            timeout: Timeout in seconds
            poll_interval: Polling interval in seconds
            
        Returns:
            Transaction receipt
        """
        start_time = time.time()
        while True:
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                if receipt:
                    return dict(receipt)
            except Exception:
                pass
            
            await asyncio.sleep(poll_interval)
            
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for transaction receipt: {tx_hash}")
    
    async def call_function(self, contract_name: str, function_name: str, *args, **kwargs) -> Any:
        """
        Call a read-only contract function.
        
        Args:
            contract_name: Name of the contract to call
            function_name: Name of the function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
        """
        try:
            # Get contract instance
            contract = self.contracts.get(contract_name)
            if not contract:
                raise ValueError(f"Contract '{contract_name}' not found")
            
            # Get function
            contract_function = getattr(contract.functions, function_name)
            if not contract_function:
                raise ValueError(f"Function '{function_name}' not found in contract '{contract_name}'")
            
            # Call function
            result = contract_function(*args, **kwargs).call()
            
            return result
        except Exception as e:
            logger.error(f"Failed to call function: {e}")
            raise
    
    async def log_contribution(self, contribution_id: str, client_id: str, round_num: int, 
                              metrics_json: str, model_version: str, update_hash: str) -> str:
        """
        Log a contribution on the blockchain.
        
        Args:
            contribution_id: Unique identifier for the contribution
            client_id: Client identifier
            round_num: Training round number
            metrics_json: JSON string with training metrics
            model_version: Model version
            update_hash: Hash of model update
            
        Returns:
            Transaction hash
        """
        try:
            return await self.send_transaction(
                'ContributionLogging',
                'logContribution',
                contribution_id,
                client_id,
                round_num,
                metrics_json,
                model_version,
                update_hash
            )
        except Exception as e:
            logger.error(f"Failed to log contribution: {e}")
            raise
    
    async def verify_contribution_quality(self, contribution_id: str, passed: bool) -> str:
        """
        Verify the quality of a contribution.
        
        Args:
            contribution_id: Unique identifier for the contribution
            passed: Whether the contribution passed quality checks
            
        Returns:
            Transaction hash
        """
        try:
            return await self.send_transaction(
                'QualityVerification',
                'verifyContribution',
                contribution_id,
                "{}",  # Metrics JSON (empty for now)
                passed,
                "Automated verification" if passed else "Failed quality checks"
            )
        except Exception as e:
            logger.error(f"Failed to verify contribution quality: {e}")
            raise
    
    async def issue_reward(self, contribution_id: str, recipient: str, 
                          accuracy: int, dataset_size: int, round_num: int) -> str:
        """
        Issue a reward for a contribution.
        
        Args:
            contribution_id: Unique identifier for the contribution
            recipient: Address to receive the reward
            accuracy: Accuracy of the contribution (scaled by 1000)
            dataset_size: Size of the dataset used for training
            round_num: Training round number
            
        Returns:
            Transaction hash
        """
        try:
            return await self.send_transaction(
                'RewardDistribution',
                'issueReward',
                contribution_id,
                recipient,
                accuracy,
                dataset_size,
                round_num
            )
        except Exception as e:
            logger.error(f"Failed to issue reward: {e}")
            raise
    
    async def log_aggregation(self, aggregation_id: str, round_num: int, 
                             model_version: str, model_hash: str, 
                             contribution_ids: List[str]) -> str:
        """
        Log a model aggregation on the blockchain.
        
        Args:
            aggregation_id: Unique identifier for the aggregation
            round_num: Training round number
            model_version: Resulting model version
            model_hash: Hash of aggregated model
            contribution_ids: IDs of included contributions
            
        Returns:
            Transaction hash
        """
        try:
            return await self.send_transaction(
                'ContributionLogging',
                'logAggregation',
                aggregation_id,
                round_num,
                model_version,
                model_hash,
                contribution_ids
            )
        except Exception as e:
            logger.error(f"Failed to log aggregation: {e}")
            raise
    
    async def create_governance_proposal(self, title: str, description: str, 
                                        target_contract: str, call_data: bytes) -> int:
        """
        Create a governance proposal.
        
        Args:
            title: Short title of the proposal
            description: Detailed description of the proposal
            target_contract: Address of the contract to call if proposal passes
            call_data: Function call data for execution
            
        Returns:
            Proposal ID
        """
        try:
            tx_hash = await self.send_transaction(
                'FLGovernance',
                'createProposal',
                title,
                description,
                target_contract,
                call_data
            )
            
            # Get proposal ID from events (simplified approach)
            receipt = await self._wait_for_transaction_receipt(tx_hash)
            contract = self.contracts['FLGovernance']
            
            # Parse events from receipt
            logs = contract.events.ProposalCreated().process_receipt(receipt)
            if logs:
                proposal_id = logs[0]['args']['proposalId']
                return proposal_id
            else:
                raise ValueError("Could not find proposal ID in transaction logs")
        except Exception as e:
            logger.error(f"Failed to create governance proposal: {e}")
            raise
    
    async def cast_vote(self, proposal_id: int, in_support: bool) -> str:
        """
        Cast a vote on a governance proposal.
        
        Args:
            proposal_id: ID of the proposal to vote on
            in_support: Whether the vote is in support
            
        Returns:
            Transaction hash
        """
        try:
            return await self.send_transaction(
                'FLGovernance',
                'castVote',
                proposal_id,
                in_support
            )
        except Exception as e:
            logger.error(f"Failed to cast vote: {e}")
            raise
    
    async def get_contribution_details(self, contribution_id: str) -> Dict[str, Any]:
        """
        Get details about a contribution.
        
        Args:
            contribution_id: Unique identifier for the contribution
            
        Returns:
            Dictionary with contribution details
        """
        try:
            result = await self.call_function(
                'ContributionLogging',
                'contributions',
                contribution_id
            )
            
            # Convert tuple to dictionary
            return {
                'contributor': result[0],
                'clientId': result[1],
                'round': result[2],
                'timestamp': result[3],
                'metricsJson': result[4],
                'modelVersion': result[5],
                'updateHash': result[6],
                'qualityVerified': result[7],
                'rewardIssued': result[8],
                'rewardAmount': result[9],
                'aggregationId': result[10]
            }
        except Exception as e:
            logger.error(f"Failed to get contribution details: {e}")
            raise
    
    async def get_proposal_details(self, proposal_id: int) -> Dict[str, Any]:
        """
        Get details about a governance proposal.
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            Dictionary with proposal details
        """
        try:
            result = await self.call_function(
                'FLGovernance',
                'getProposalDetails',
                proposal_id
            )
            
            # Convert tuple to dictionary
            return {
                'title': result[0],
                'description': result[1],
                'proposer': result[2],
                'createdAt': result[3],
                'votingEndsAt': result[4],
                'status': result[5],
                'yesVotes': result[6],
                'noVotes': result[7],
                'executed': result[8]
            }
        except Exception as e:
            logger.error(f"Failed to get proposal details: {e}")
            raise
    
    async def get_reward_policy(self) -> Dict[str, Any]:
        """
        Get the current reward policy parameters.
        
        Returns:
            Dictionary with reward policy details
        """
        try:
            result = await self.call_function(
                'RewardDistribution',
                'policy'
            )
            
            # Convert tuple to dictionary
            return {
                'baseReward': result[0],
                'accuracyMultiplier': result[1],
                'datasetSizeMultiplier': result[2],
                'maxReward': result[3],
                'roundRewardBudget': result[4]
            }
        except Exception as e:
            logger.error(f"Failed to get reward policy: {e}")
            raise
    
    async def get_token_balance(self, address: Optional[str] = None) -> int:
        """
        Get token balance for an address.
        
        Args:
            address: Address to check (default: middleware account address)
            
        Returns:
            Token balance as integer
        """
        try:
            address = address or self.account.address
            
            result = await self.call_function(
                'FedLearningToken',
                'balanceOf',
                address
            )
            
            return result
        except Exception as e:
            logger.error(f"Failed to get token balance: {e}")
            raise
    
    async def calculate_reward(self, accuracy: int, dataset_size: int) -> int:
        """
        Calculate reward amount based on contribution metrics.
        
        Args:
            accuracy: Accuracy of the contribution (scaled by 1000)
            dataset_size: Size of the dataset used for training
            
        Returns:
            Calculated reward amount
        """
        try:
            result = await self.call_function(
                'RewardDistribution',
                'calculateReward',
                accuracy,
                dataset_size
            )
            
            return result
        except Exception as e:
            logger.error(f"Failed to calculate reward: {e}")
            raise
    
    def create_contract_call_data(self, contract_name: str, function_name: str, *args) -> bytes:
        """
        Create encoded call data for a contract function.
        
        Args:
            contract_name: Name of the contract
            function_name: Name of the function
            *args: Function arguments
            
        Returns:
            Encoded call data
        """
        try:
            # Get contract instance
            contract = self.contracts.get(contract_name)
            if not contract:
                raise ValueError(f"Contract '{contract_name}' not found")
            
            # Get function
            contract_function = getattr(contract.functions, function_name)
            if not contract_function:
                raise ValueError(f"Function '{function_name}' not found in contract '{contract_name}'")
            
            # Encode call data
            call_data = contract_function(*args).build_transaction({
                'gas': 0,
                'gasPrice': 0,
                'nonce': 0,
                'value': 0
            })['data']
            
            return call_data
        except Exception as e:
            logger.error(f"Failed to create contract call data: {e}")
            raise
    
    async def mint_tokens(self, to_address: str, amount: int, reason: str) -> str:
        """
        Mint new tokens (only for accounts with minter role).
        
        Args:
            to_address: Address to receive the tokens
            amount: Amount of tokens to mint
            reason: Reason for minting
            
        Returns:
            Transaction hash
        """
        try:
            return await self.send_transaction(
                'FedLearningToken',
                'mint',
                to_address,
                amount,
                reason
            )
        except Exception as e:
            logger.error(f"Failed to mint tokens: {e}")
            raise
    
    async def check_role(self, contract_name: str, role_name: str, address: Optional[str] = None) -> bool:
        """
        Check if an address has a specific role in a contract.
        
        Args:
            contract_name: Name of the contract
            role_name: Name of the role to check
            address: Address to check (default: middleware account address)
            
        Returns:
            Boolean indicating if the address has the role
        """
        try:
            address = address or self.account.address
            
            # Get role hash
            role_hash = None
            if role_name == "DEFAULT_ADMIN_ROLE":
                role_hash = "0x0000000000000000000000000000000000000000000000000000000000000000"
            else:
                # Try to get the role constant from the contract
                try:
                    role_hash = await self.call_function(
                        contract_name,
                        role_name
                    )
                except Exception:
                    # Calculate role hash using keccak256
                    role_hash = Web3.keccak(text=role_name).hex()
            
            # Check if the address has the role
            result = await self.call_function(
                contract_name,
                'hasRole',
                role_hash,
                address
            )
            
            return result
        except Exception as e:
            logger.error(f"Failed to check role: {e}")
            raise
    
    async def get_event_logs(self, contract_name: str, event_name: str, 
                            from_block: int = 0, to_block: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get event logs for a specific event.
        
        Args:
            contract_name: Name of the contract
            event_name: Name of the event
            from_block: Starting block number
            to_block: Ending block number (default: latest)
            
        Returns:
            List of event logs
        """
        try:
            # Get contract instance
            contract = self.contracts.get(contract_name)
            if not contract:
                raise ValueError(f"Contract '{contract_name}' not found")
            
            # Get event object
            event = getattr(contract.events, event_name)
            if not event:
                raise ValueError(f"Event '{event_name}' not found in contract '{contract_name}'")
            
            # Get to_block
            if to_block is None:
                to_block = self.w3.eth.block_number
            
            # Create filter
            event_filter = event.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            # Get logs
            logs = event_filter.get_all_entries()
            
            # Process logs
            result = []
            for log in logs:
                # Convert event data to dictionary
                event_data = dict(log['args'])
                event_data['blockNumber'] = log['blockNumber']
                event_data['transactionHash'] = log['transactionHash'].hex()
                event_data['logIndex'] = log['logIndex']
                result.append(event_data)
            
            return result
        except Exception as e:
            logger.error(f"Failed to get event logs: {e}")
            raise

# Example usage
async def example_usage():
    # Initialize middleware
    middleware = BlockchainMiddleware()
    
    # Example: Log a contribution
    contribution_id = f"contribution_{int(time.time())}"
    tx_hash = await middleware.log_contribution(
        contribution_id=contribution_id,
        client_id="client_123",
        round_num=1,
        metrics_json='{"accuracy": 0.85, "loss": 0.12, "dataset_size": 1000}',
        model_version="0.1.0",
        update_hash="0x123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )
    print(f"Contribution logged with transaction: {tx_hash}")
    
    # Example: Verify contribution quality
    tx_hash = await middleware.verify_contribution_quality(
        contribution_id=contribution_id,
        passed=True
    )
    print(f"Contribution verified with transaction: {tx_hash}")
    
    # Example: Issue reward
    tx_hash = await middleware.issue_reward(
        contribution_id=contribution_id,
        recipient=middleware.account.address,
        accuracy=850,  # 85.0%
        dataset_size=1000,
        round_num=1
    )
    print(f"Reward issued with transaction: {tx_hash}")
    
    # Example: Get contribution details
    details = await middleware.get_contribution_details(contribution_id)
    print(f"Contribution details: {details}")

if __name__ == "__main__":
    # Run async example
    asyncio.run(example_usage())
