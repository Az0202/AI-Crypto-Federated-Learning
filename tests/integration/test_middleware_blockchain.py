"""
Integration Tests for Middleware-Blockchain Interaction

This module tests the integration between the blockchain middleware and
the smart contracts deployed on the blockchain network.
"""

import unittest
import os
import json
import asyncio
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
import eth_account
from eth_account.messages import encode_defunct
from web3 import Web3

# Import components to test - adjust paths as needed
from blockchain.contracts.blockchain_middleware import BlockchainMiddleware

class TestMiddlewareBlockchainIntegration(unittest.TestCase):
    """Test cases for the Middleware-Blockchain integration."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary configuration
        self.config = {
            "provider_url": "http://localhost:8545",  # Local Ganache or Hardhat node
            "chain_id": 1337,
            "gas_limit": 3000000,
            "key_env_var": "TEST_PRIVATE_KEY",
            "contracts": {
                "ContributionLogging": {
                    "address": "0x1234567890123456789012345678901234567890",
                    "abi_path": self._create_mock_abi_file("ContributionLogging")
                },
                "QualityVerification": {
                    "address": "0x2345678901234567890123456789012345678901",
                    "abi_path": self._create_mock_abi_file("QualityVerification")
                },
                "RewardDistribution": {
                    "address": "0x3456789012345678901234567890123456789012",
                    "abi_path": self._create_mock_abi_file("RewardDistribution")
                },
                "FedLearningToken": {
                    "address": "0x4567890123456789012345678901234567890123",
                    "abi_path": self._create_mock_abi_file("FedLearningToken")
                },
                "FLGovernance": {
                    "address": "0x5678901234567890123456789012345678901234",
                    "abi_path": self._create_mock_abi_file("FLGovernance")
                }
            }
        }
        
        # Create temporary config file
        fd, self.config_path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(self.config, f)
        
        # Set up test private key
        self.test_private_key = "0x" + "1" * 64  # Mock private key
        os.environ["TEST_PRIVATE_KEY"] = self.test_private_key
        
        # Set up test account
        self.account = eth_account.Account.from_key(self.test_private_key)
        
        # Mock Web3 provider
        self.setup_web3_mock()
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary files
        os.remove(self.config_path)
        for contract_info in self.config["contracts"].values():
            os.remove(contract_info["abi_path"])
        
        # Remove environment variable
        del os.environ["TEST_PRIVATE_KEY"]
        
        # Stop Web3 patcher
        self.web3_patcher.stop()
    
    def _create_mock_abi_file(self, contract_name):
        """Create a mock ABI file for testing."""
        # Generate appropriate mock ABI based on contract name
        if contract_name == "ContributionLogging":
            abi = [
                {
                    "inputs": [
                        {"name": "contributionId", "type": "string"},
                        {"name": "clientId", "type": "string"},
                        {"name": "round", "type": "uint256"},
                        {"name": "metricsJson", "type": "string"},
                        {"name": "modelVersion", "type": "string"},
                        {"name": "updateHash", "type": "string"}
                    ],
                    "name": "logContribution",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"name": "aggregationId", "type": "string"},
                        {"name": "round", "type": "uint256"},
                        {"name": "modelVersion", "type": "string"},
                        {"name": "modelHash", "type": "string"},
                        {"name": "contributionIds", "type": "string[]"}
                    ],
                    "name": "logAggregation",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"name": "contributionId", "type": "string"}],
                    "name": "contributions",
                    "outputs": [
                        {"name": "contributor", "type": "address"},
                        {"name": "clientId", "type": "string"},
                        {"name": "round", "type": "uint256"},
                        {"name": "timestamp", "type": "uint256"},
                        {"name": "metricsJson", "type": "string"},
                        {"name": "modelVersion", "type": "string"},
                        {"name": "updateHash", "type": "string"},
                        {"name": "qualityVerified", "type": "bool"},
                        {"name": "rewardIssued", "type": "bool"},
                        {"name": "rewardAmount", "type": "uint256"},
                        {"name": "aggregationId", "type": "string"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
        elif contract_name == "QualityVerification":
            abi = [
                {
                    "inputs": [
                        {"name": "contributionId", "type": "string"},
                        {"name": "metrics", "type": "string"},
                        {"name": "passed", "type": "bool"},
                        {"name": "reason", "type": "string"}
                    ],
                    "name": "verifyContribution",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
        elif contract_name == "RewardDistribution":
            abi = [
                {
                    "inputs": [
                        {"name": "contributionId", "type": "string"},
                        {"name": "recipient", "type": "address"},
                        {"name": "accuracy", "type": "uint256"},
                        {"name": "datasetSize", "type": "uint256"},
                        {"name": "round", "type": "uint256"}
                    ],
                    "name": "issueReward",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"name": "accuracy", "type": "uint256"},
                        {"name": "datasetSize", "type": "uint256"}
                    ],
                    "name": "calculateReward",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
        elif contract_name == "FedLearningToken":
            abi = [
                {
                    "inputs": [{"name": "account", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"name": "to", "type": "address"},
                        {"name": "amount", "type": "uint256"},
                        {"name": "reason", "type": "string"}
                    ],
                    "name": "mint",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
        elif contract_name == "FLGovernance":
            abi = [
                {
                    "inputs": [
                        {"name": "title", "type": "string"},
                        {"name": "description", "type": "string"},
                        {"name": "targetContract", "type": "address"},
                        {"name": "callData", "type": "bytes"}
                    ],
                    "name": "createProposal",
                    "outputs": [{"name": "proposalId", "type": "uint256"}],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"name": "proposalId", "type": "uint256"},
                        {"name": "inSupport", "type": "bool"}
                    ],
                    "name": "castVote",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
        else:
            abi = []
        
        # Write ABI to temporary file
        fd, abi_path = tempfile.mkstemp(suffix=f'_{contract_name}_abi.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(abi, f)
        
        return abi_path
    
    def setup_web3_mock(self):
        """Set up mock Web3 provider and components."""
        # Create main patcher for Web3
        self.web3_patcher = patch('blockchain.contracts.blockchain_middleware.Web3')
        self.mock_web3_class = self.web3_patcher.start()
        
        # Create mock Web3 instance
        self.mock_web3 = MagicMock()
        self.mock_web3_class.return_value = self.mock_web3
        
        # Mock connection status
        self.mock_web3.is_connected.return_value = True
        
        # Mock eth module
        mock_eth = MagicMock()
        self.mock_web3.eth = mock_eth
        
        # Mock gas price and block number
        mock_eth.gas_price = 20000000000  # 20 Gwei
        mock_eth.block_number = 12345
        
        # Mock get_transaction_count
        mock_eth.get_transaction_count.return_value = 1
        
        # Mock account
        mock_eth.account = MagicMock()
        mock_eth.account.sign_transaction.return_value = MagicMock(
            rawTransaction=b'0x123456'
        )
        
        # Mock send_raw_transaction
        mock_eth.send_raw_transaction.return_value = b'0xabcdef'
        
        # Mock to_hex
        self.mock_web3.to_hex.return_value = '0xabcdef'
        
        # Mock contract
        mock_contract = MagicMock()
        self.mock_web3.eth.contract.return_value = mock_contract
        
        # Set up mock contract functions
        self.mock_contract_functions = {}
        mock_contract.functions = MagicMock()
        
        # Configure each function to return a mock builder
        for contract_name, contract_info in self.config["contracts"].items():
            with open(contract_info["abi_path"], 'r') as f:
                abi = json.load(f)
                
            for func in abi:
                if func["type"] == "function":
                    func_name = func["name"]
                    
                    # Create mock function builder
                    mock_function = MagicMock()
                    
                    # Configure return values for view functions
                    if contract_name == "FedLearningToken" and func_name == "balanceOf":
                        mock_function.return_value.call.return_value = 1000 * 10**18
                    elif contract_name == "RewardDistribution" and func_name == "calculateReward":
                        mock_function.return_value.call.return_value = 10 * 10**18
                    elif contract_name == "ContributionLogging" and func_name == "contributions":
                        mock_function.return_value.call.return_value = (
                            self.account.address,
                            "client_123",
                            1,
                            123456789,
                            '{"accuracy": 0.85}',
                            "0.1.0",
                            "0x123...",
                            True,
                            False,
                            0,
                            ""
                        )
                    elif contract_name == "FLGovernance" and func_name == "createProposal":
                        # For functions with return values we need to mock the events
                        tx_receipt = {
                            "logs": [{
                                "topics": [
                                    # Mock event signature topic
                                    "0x123456789abcdef"
                                ],
                                "data": "0x0000000000000000000000000000000000000000000000000000000000000005"
                                # data encodes proposal ID 5
                            }]
                        }
                        mock_eth.wait_for_transaction_receipt.return_value = tx_receipt
                        
                        # Mock the event processor
                        mock_event = MagicMock()
                        mock_event.process_receipt.return_value = [{
                            "args": {"proposalId": 5}
                        }]
                        mock_contract.events.ProposalCreated.return_value = mock_event
                    
                    # Configure build_transaction
                    mock_function.return_value.build_transaction.return_value = {
                        'to': contract_info["address"],
                        'data': '0x123456789abcdef',
                        'gas': 3000000,
                        'gasPrice': 20000000000,
                        'nonce': 1,
                        'chainId': 1337,
                        'value': 0
                    }
                    
                    # Store in dictionary
                    self.mock_contract_functions[(contract_name, func_name)] = mock_function
                    
                    # Add to mock contract functions
                    setattr(mock_contract.functions, func_name, mock_function)
        
        # Set up get_transaction_receipt mock
        mock_eth.get_transaction_receipt.return_value = {
            "status": 1,  # Success
            "blockNumber": 12346,
            "logs": []
        }
        
        # Set up wait_for_transaction_receipt mock
        mock_eth.wait_for_transaction_receipt.return_value = {
            "status": 1,  # Success
            "blockNumber": 12346,
            "logs": []
        }
    
    def test_initialization(self):
        """Test proper initialization of BlockchainMiddleware."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Check that config was loaded correctly
        self.assertEqual(middleware.config['provider_url'], "http://localhost:8545")
        self.assertEqual(middleware.config['chain_id'], 1337)
        
        # Check that Web3 was initialized
        self.mock_web3_class.assert_called_once()
        self.mock_web3.is_connected.assert_called_once()
        
        # Check that account was loaded
        self.assertEqual(middleware.account.address, self.account.address)
        
        # Check that contracts were loaded
        self.mock_web3.eth.contract.call_count = len(self.config["contracts"])
        self.assertEqual(len(middleware.contracts), len(self.config["contracts"]))
    
    async def test_send_transaction(self):
        """Test sending a transaction to a contract function."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Send a test transaction using keyword arguments
        tx_hash = await middleware.send_transaction(
            contract_name="ContributionLogging",
            function_name="logContribution",
            arg1="contribution_123",
            arg2="client_123",
            arg3=1,
            arg4='{"accuracy": 0.85}',
            arg5="0.1.0",
            arg6="0x123..."
        )
        
        # Check that transaction was sent
        self.assertEqual(tx_hash, "0xabcdef")
        
        # Check that the contract function was called
        mock_function = self.mock_contract_functions[("ContributionLogging", "logContribution")]
        mock_function.assert_called_once_with(
            "contribution_123",
            "client_123",
            1,
            '{"accuracy": 0.85}',
            "0.1.0",
            "0x123..."
        )
        
        # Check that build_transaction was called
        mock_function.return_value.build_transaction.assert_called_once()
        
        # Check that sign_transaction was called
        self.mock_web3.eth.account.sign_transaction.assert_called_once()
        
        # Check that send_raw_transaction was called
        self.mock_web3.eth.send_raw_transaction.assert_called_once()
    
    async def test_call_function(self):
        """Test calling a read-only function."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Call a test function
        result = await middleware.call_function(
            contract_name="FedLearningToken",
            function_name="balanceOf",
            arg1=self.account.address
        )
        
        # Check that the function was called
        mock_function = self.mock_contract_functions[("FedLearningToken", "balanceOf")]
        mock_function.assert_called_once_with(self.account.address)
        
        # Check that call() was called
        mock_function.return_value.call.assert_called_once()
        
        # Check the result
        self.assertEqual(result, 1000 * 10**18)
    
    async def test_log_contribution(self):
        """Test logging a contribution on the blockchain."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Log a contribution using keyword arguments
        tx_hash = await middleware.log_contribution(
            contribution_id="contribution_123",
            client_id="client_123",
            round_num=1,
            metrics_json='{"accuracy": 0.85, "loss": 0.15}',
            model_version="0.1.0",
            update_hash="0x123..."
        )
        
        # Check that the transaction was sent
        self.assertEqual(tx_hash, "0xabcdef")
        
        # Check that the contract function was called
        mock_function = self.mock_contract_functions[("ContributionLogging", "logContribution")]
        mock_function.assert_called_once_with(
            "contribution_123",
            "client_123",
            1,
            '{"accuracy": 0.85, "loss": 0.15}',
            "0.1.0",
            "0x123..."
        )
    
    async def test_verify_contribution_quality(self):
        """Test verifying contribution quality on the blockchain."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Verify a contribution
        tx_hash = await middleware.verify_contribution_quality(
            contribution_id="contribution_123",
            passed=True
        )
        
        # Check that the transaction was sent
        self.assertEqual(tx_hash, "0xabcdef")
        
        # Check that the contract function was called
        mock_function = self.mock_contract_functions[("QualityVerification", "verifyContribution")]
        mock_function.assert_called_once()
        
        # Check the call arguments
        call_args, _ = mock_function.call_args
        self.assertEqual(call_args[0], "contribution_123")  # contribution_id
        self.assertEqual(call_args[2], True)  # passed
    
    async def test_issue_reward(self):
        """Test issuing a reward on the blockchain."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Issue a reward
        tx_hash = await middleware.issue_reward(
            contribution_id="contribution_123",
            recipient=self.account.address,
            accuracy=850,  # 85.0%
            dataset_size=1000,
            round_num=1
        )
        
        # Check that the transaction was sent
        self.assertEqual(tx_hash, "0xabcdef")
        
        # Check that the contract function was called
        mock_function = self.mock_contract_functions[("RewardDistribution", "issueReward")]
        mock_function.assert_called_once_with(
            "contribution_123",
            self.account.address,
            850,
            1000,
            1
        )
    
    async def test_log_aggregation(self):
        """Test logging an aggregation on the blockchain."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Log an aggregation
        tx_hash = await middleware.log_aggregation(
            aggregation_id="aggregation_123",
            round_num=1,
            model_version="0.1.0",
            model_hash="0x123...",
            contribution_ids=["contribution_1", "contribution_2", "contribution_3"]
        )
        
        # Check that the transaction was sent
        self.assertEqual(tx_hash, "0xabcdef")
        
        # Check that the contract function was called
        mock_function = self.mock_contract_functions[("ContributionLogging", "logAggregation")]
        mock_function.assert_called_once_with(
            "aggregation_123",
            1,
            "0.1.0",
            "0x123...",
            ["contribution_1", "contribution_2", "contribution_3"]
        )
    
    async def test_get_token_balance(self):
        """Test getting token balance from the blockchain."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Get token balance
        balance = await middleware.get_token_balance(address=self.account.address)
        
        # Check that the contract function was called
        mock_function = self.mock_contract_functions[("FedLearningToken", "balanceOf")]
        mock_function.assert_called_once_with(self.account.address)
        
        # Check the result
        self.assertEqual(balance, 1000 * 10**18)
    
    async def test_create_governance_proposal(self):
        """Test creating a governance proposal on the blockchain."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Create a proposal
        proposal_id = await middleware.create_governance_proposal(
            title="Test Proposal",
            description="This is a test proposal",
            target_contract="0x9876543210987654321098765432109876543210",
            call_data=b'0x123456789abcdef'
        )
        
        # Check that the contract function was called
        mock_function = self.mock_contract_functions[("FLGovernance", "createProposal")]
        mock_function.assert_called_once_with(
            "Test Proposal",
            "This is a test proposal",
            "0x9876543210987654321098765432109876543210",
            b'0x123456789abcdef'
        )
        
        # Check the result
        self.assertEqual(proposal_id, 5)
    
    async def test_cast_vote(self):
        """Test casting a vote on a governance proposal."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Cast a vote
        tx_hash = await middleware.cast_vote(
            proposal_id=1,
            in_support=True
        )
        
        # Check that the transaction was sent
        self.assertEqual(tx_hash, "0xabcdef")
        
        # Check that the contract function was called
        mock_function = self.mock_contract_functions[("FLGovernance", "castVote")]
        mock_function.assert_called_once_with(1, True)
    
    async def test_calculate_reward(self):
        """Test reward calculation."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Calculate reward
        reward = await middleware.calculate_reward(
            accuracy=850,  # 85.0%
            dataset_size=1000
        )
        
        # Check that the contract function was called
        mock_function = self.mock_contract_functions[("RewardDistribution", "calculateReward")]
        mock_function.assert_called_once_with(850, 1000)
        
        # Check the result
        self.assertEqual(reward, 10 * 10**18)
    
    async def test_get_contribution_details(self):
        """Test getting contribution details from the blockchain."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Get contribution details
        details = await middleware.get_contribution_details(contribution_id="contribution_123")
        
        # Check that the contract function was called
        mock_function = self.mock_contract_functions[("ContributionLogging", "contributions")]
        mock_function.assert_called_once_with("contribution_123")
        
        # Check the result structure
        self.assertIn("contributor", details)
        self.assertIn("clientId", details)
        self.assertIn("round", details)
        self.assertIn("timestamp", details)
        self.assertIn("metricsJson", details)
        self.assertIn("modelVersion", details)
        self.assertIn("updateHash", details)
        self.assertIn("qualityVerified", details)
        self.assertIn("rewardIssued", details)
        self.assertIn("rewardAmount", details)
        
        # Check some values
        self.assertEqual(details["contributor"], self.account.address)
        self.assertEqual(details["clientId"], "client_123")
        self.assertEqual(details["round"], 1)
    
    async def test_error_handling(self):
        """Test error handling in blockchain interactions."""
        middleware = BlockchainMiddleware(config_path=self.config_path)
        
        # Mock a transaction failure
        self.mock_web3.eth.wait_for_transaction_receipt.return_value = {
            "status": 0,  # Failure
            "blockNumber": 12346
        }
        
        # Attempt to send a transaction
        with self.assertRaises(ValueError):
            await middleware.send_transaction(
                contract_name="ContributionLogging",
                function_name="logContribution",
                arg1="contribution_123",
                arg2="client_123",
                arg3=1,
                arg4='{"accuracy": 0.85}',
                arg5="0.1.0",
                arg6="0x123..."
            )
        
        # Mock a function not found error
        with self.assertRaises(ValueError):
            await middleware.send_transaction(
                contract_name="ContributionLogging",
                function_name="nonExistentFunction",
                arg1="arg1"
            )
        
        # Mock a contract not found error
        with self.assertRaises(ValueError):
            await middleware.send_transaction(
                contract_name="NonExistentContract",
                function_name="someFunction",
                arg1="arg1"
            )

if __name__ == '__main__':
    unittest.main() 