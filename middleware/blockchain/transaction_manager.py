"""
Optimized Blockchain Transaction Manager

This module provides optimized blockchain transaction handling with features like:
- Automatic nonce management for concurrent transactions
- Exponential backoff retry with dynamic gas price adjustment
- Transaction batching for gas optimization
- Asynchronous transaction confirmation
- Transaction queue management
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import aiohttp
import json
from web3 import Web3
from web3.exceptions import TransactionNotFound, TimeExhausted
from eth_account.signers.local import LocalAccount
import threading
from dataclasses import dataclass
from enum import Enum
import heapq
import random

logger = logging.getLogger(__name__)

class TransactionStatus(Enum):
    """Status of a blockchain transaction."""
    QUEUED = 0
    PENDING = 1
    SUBMITTED = 2
    CONFIRMED = 3
    FAILED = 4
    DROPPED = 5

@dataclass
class TransactionRequest:
    """A request to execute a blockchain transaction."""
    contract_name: str
    function_name: str
    args: tuple
    kwargs: dict
    priority: int = 1  # 1 is highest priority, larger numbers are lower priority
    max_attempts: int = 3
    attempts: int = 0
    created_at: float = 0.0
    submitted_at: float = 0.0
    tx_hash: Optional[str] = None
    status: TransactionStatus = TransactionStatus.QUEUED
    callback: Optional[Callable] = None
    result: Any = None
    error: Optional[Exception] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
    
    # For priority queue comparison
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at

class BlockchainTransactionManager:
    """
    Manager for optimized blockchain transactions.
    
    Features:
    - Transaction queue for prioritization
    - Automatic nonce management
    - Retry mechanism with backoff
    - Gas price adjustment
    - Transaction batching
    """
    
    def __init__(
        self, 
        web3: Web3, 
        account: LocalAccount,
        chain_id: int,
        contracts: Dict[str, Any],
        max_concurrent_txs: int = 5,
        base_gas_price_multiplier: float = 1.1,
        max_gas_price_multiplier: float = 2.0,
        gas_price_bump_percent: int = 10,
        confirmation_blocks: int = 2,
        request_timeout: int = 120,
        min_priority_fee: int = 1500000000  # 1.5 gwei
    ):
        """
        Initialize the transaction manager.
        
        Args:
            web3: Web3 instance
            account: Account to use for transactions
            chain_id: Blockchain network chain ID
            contracts: Dictionary of contract instances
            max_concurrent_txs: Maximum number of concurrent transactions
            base_gas_price_multiplier: Multiplier for base gas price
            max_gas_price_multiplier: Maximum multiplier for gas price
            gas_price_bump_percent: Percentage to bump gas price on retry
            confirmation_blocks: Number of blocks to wait for confirmation
            request_timeout: Timeout for transaction requests in seconds
            min_priority_fee: Minimum priority fee in wei
        """
        self.web3 = web3
        self.account = account
        self.chain_id = chain_id
        self.contracts = contracts
        self.max_concurrent_txs = max_concurrent_txs
        self.base_gas_price_multiplier = base_gas_price_multiplier
        self.max_gas_price_multiplier = max_gas_price_multiplier
        self.gas_price_bump_percent = gas_price_bump_percent
        self.confirmation_blocks = confirmation_blocks
        self.request_timeout = request_timeout
        self.min_priority_fee = min_priority_fee
        
        # Transaction queue (priority queue)
        self._tx_queue = []
        
        # Active transactions
        self._active_txs = {}
        
        # Nonce management
        self._last_used_nonce = -1
        self._nonce_lock = threading.Lock()
        self._nonce_cache = {}
        
        # Threading and event management
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread = None
        
        # Gas price tracker
        self._current_gas_price = None
        self._last_gas_price_update = 0
        self._gas_price_update_interval = 60  # seconds
        
        # Performance metrics
        self._tx_stats = {
            "submitted": 0,
            "confirmed": 0,
            "failed": 0,
            "retried": 0,
            "avg_confirmation_time": 0,
            "total_confirmation_time": 0
        }
    
    def start(self):
        """Start the transaction manager worker thread."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            logger.info("Transaction manager worker thread started")
    
    def stop(self):
        """Stop the transaction manager worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            self._stop_event.set()
            self._queue_event.set()  # Wake up the worker
            self._worker_thread.join(timeout=10)
            logger.info("Transaction manager worker thread stopped")
    
    async def send_transaction(
        self, 
        contract_name: str, 
        function_name: str, 
        *args, 
        priority: int = 1,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """
        Send a transaction asynchronously.
        
        Args:
            contract_name: Name of the contract
            function_name: Name of the function to call
            *args: Positional arguments for the function
            priority: Transaction priority (1 is highest)
            callback: Optional callback function to call when transaction is confirmed
            **kwargs: Keyword arguments for the function
            
        Returns:
            Transaction hash
        """
        # Create transaction request
        tx_request = TransactionRequest(
            contract_name=contract_name,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            callback=callback
        )
        
        # Add to queue
        with self._queue_lock:
            heapq.heappush(self._tx_queue, tx_request)
        
        # Notify worker thread
        self._queue_event.set()
        
        # Create a future to wait for transaction submission
        future = asyncio.Future()
        
        # Wait for transaction to be submitted
        # We use a custom callback that will be called when the tx is submitted
        def on_submit(tx_hash, error=None):
            if error:
                future.set_exception(error)
            else:
                future.set_result(tx_hash)
        
        tx_request.on_submit = on_submit
        
        try:
            # Wait for transaction to be submitted with timeout
            tx_hash = await asyncio.wait_for(future, timeout=self.request_timeout)
            return tx_hash
        except asyncio.TimeoutError:
            logger.error(f"Transaction submission timed out: {contract_name}.{function_name}")
            # Remove from queue if it hasn't been processed yet
            with self._queue_lock:
                try:
                    self._tx_queue.remove(tx_request)
                except ValueError:
                    pass  # Not in queue anymore
            raise TimeoutError(f"Transaction submission timed out: {contract_name}.{function_name}")
    
    async def send_transaction_and_wait(
        self, 
        contract_name: str, 
        function_name: str, 
        *args, 
        priority: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a transaction and wait for confirmation.
        
        Args:
            contract_name: Name of the contract
            function_name: Name of the function to call
            *args: Positional arguments for the function
            priority: Transaction priority (1 is highest)
            **kwargs: Keyword arguments for the function
            
        Returns:
            Transaction receipt
        """
        # Create future for the receipt
        future = asyncio.Future()
        
        # Define callback
        def on_complete(receipt, error=None):
            if error:
                future.set_exception(error)
            else:
                future.set_result(receipt)
        
        # Send transaction
        tx_hash = await self.send_transaction(
            contract_name,
            function_name,
            *args,
            priority=priority,
            callback=on_complete,
            **kwargs
        )
        
        # Wait for confirmation with timeout
        try:
            receipt = await asyncio.wait_for(future, timeout=self.request_timeout * 2)
            return receipt
        except asyncio.TimeoutError:
            logger.error(f"Transaction confirmation timed out: {tx_hash}")
            raise TimeoutError(f"Transaction confirmation timed out: {tx_hash}")
    
    async def call_function(
        self, 
        contract_name: str, 
        function_name: str, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Call a read-only contract function.
        
        Args:
            contract_name: Name of the contract
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
    
    def get_transaction_count(self) -> int:
        """
        Get the current transaction count (nonce) for the account.
        
        Returns:
            Current nonce
        """
        # Check cache first if recent
        current_time = time.time()
        if self._nonce_cache.get('timestamp', 0) > current_time - 10:
            return self._nonce_cache['nonce']
        
        # Get from blockchain with retries
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                nonce = self.web3.eth.get_transaction_count(self.account.address)
                
                # Update cache
                self._nonce_cache = {
                    'nonce': nonce,
                    'timestamp': current_time
                }
                
                return nonce
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to get transaction count: {e}")
                    raise
                
                # Exponential backoff
                time.sleep(0.5 * (2 ** retry_count))
    
    def get_next_nonce(self) -> int:
        """
        Get the next available nonce for transactions.
        
        Returns:
            Next available nonce
        """
        with self._nonce_lock:
            # If we haven't initialized the nonce counter yet
            if self._last_used_nonce < 0:
                # Get current nonce from blockchain
                blockchain_nonce = self.get_transaction_count()
                self._last_used_nonce = blockchain_nonce - 1
            
            # Increment and return
            self._last_used_nonce += 1
            return self._last_used_nonce
    
    def get_gas_price(self, increase_percent: int = 0) -> int:
        """
        Get the current gas price with optional percentage increase.
        
        Args:
            increase_percent: Percentage to increase the gas price by
            
        Returns:
            Gas price in wei
        """
        current_time = time.time()
        
        # Update gas price if necessary
        if (self._current_gas_price is None or
                current_time - self._last_gas_price_update > self._gas_price_update_interval):
            try:
                # Get gas price from blockchain
                if hasattr(self.web3.eth, 'gas_price'):
                    base_gas_price = self.web3.eth.gas_price
                else:
                    # Fallback for older web3.py versions
                    base_gas_price = self.web3.eth.gasPrice
                
                # Apply base multiplier
                self._current_gas_price = int(base_gas_price * self.base_gas_price_multiplier)
                self._last_gas_price_update = current_time
                
                logger.debug(f"Updated gas price: {self._current_gas_price} wei")
            except Exception as e:
                logger.warning(f"Failed to update gas price: {e}")
                # Use previous price or fallback
                if self._current_gas_price is None:
                    self._current_gas_price = 50000000000  # 50 gwei fallback
        
        # Get current gas price with increase if specified
        gas_price = self._current_gas_price
        if increase_percent > 0:
            gas_price = int(gas_price * (1 + (increase_percent / 100)))
        
        # Ensure gas price doesn't exceed maximum
        max_gas_price = int(self._current_gas_price * self.max_gas_price_multiplier)
        gas_price = min(gas_price, max_gas_price)
        
        return gas_price
    
    def prepare_eip1559_fee_params(self, base_fee_increase_percent: int = 0) -> Dict[str, int]:
        """
        Prepare EIP-1559 fee parameters (maxFeePerGas and maxPriorityFeePerGas).
        
        Args:
            base_fee_increase_percent: Percentage to increase the base fee by
            
        Returns:
            Dictionary with fee parameters
        """
        # Check if EIP-1559 is supported
        if not hasattr(self.web3.eth, 'max_priority_fee'):
            # Fall back to legacy gas price
            return {"gasPrice": self.get_gas_price(base_fee_increase_percent)}
        
        try:
            # Get the priority fee
            priority_fee = max(self.web3.eth.max_priority_fee, self.min_priority_fee)
            
            # Get the latest block to calculate base fee
            latest_block = self.web3.eth.get_block('latest')
            base_fee_per_gas = latest_block.get('baseFeePerGas', 0)
            
            # Calculate max fee per gas with safety margin and increase if specified
            increase_multiplier = 1 + (base_fee_increase_percent / 100)
            max_fee_per_gas = int((base_fee_per_gas * 2 * increase_multiplier) + priority_fee)
            
            return {
                "maxFeePerGas": max_fee_per_gas,
                "maxPriorityFeePerGas": priority_fee
            }
        except Exception as e:
            logger.warning(f"Failed to prepare EIP-1559 fee params: {e}")
            # Fall back to legacy gas price
            return {"gasPrice": self.get_gas_price(base_fee_increase_percent)}
    
    def _worker_loop(self):
        """Main worker loop for processing transaction queue."""
        logger.info("Transaction worker loop started")
        
        while not self._stop_event.is_set():
            # Wait for queue event or stop event
            self._queue_event.wait(timeout=1.0)
            self._queue_event.clear()
            
            if self._stop_event.is_set():
                break
            
            # Process queue
            try:
                self._process_queue()
            except Exception as e:
                logger.error(f"Error in transaction worker: {e}")
                time.sleep(1.0)  # Prevent rapid cycling on persistent errors
            
            # Check confirmations for active transactions
            try:
                self._check_confirmations()
            except Exception as e:
                logger.error(f"Error checking confirmations: {e}")
        
        logger.info("Transaction worker loop stopped")
    
    def _process_queue(self):
        """Process the transaction queue."""
        # Count active transactions
        active_count = len(self._active_txs)
        if active_count >= self.max_concurrent_txs:
            return  # At maximum concurrent transactions
        
        # Calculate how many transactions we can process
        slots_available = self.max_concurrent_txs - active_count
        
        # Process up to available slots
        for _ in range(slots_available):
            with self._queue_lock:
                if not self._tx_queue:
                    break  # Queue is empty
                
                # Get highest priority transaction
                tx_request = heapq.heappop(self._tx_queue)
            
            # Process the transaction
            try:
                tx_hash = self._submit_transaction(tx_request)
                
                if tx_hash:
                    # Update request status
                    tx_request.status = TransactionStatus.SUBMITTED
                    tx_request.submitted_at = time.time()
                    tx_request.tx_hash = tx_hash
                    
                    # Add to active transactions
                    self._active_txs[tx_hash] = tx_request
                    
                    # Call on_submit callback if exists
                    if hasattr(tx_request, 'on_submit') and callable(tx_request.on_submit):
                        tx_request.on_submit(tx_hash)
                    
                    # Update stats
                    self._tx_stats["submitted"] += 1
                else:
                    # Failed to submit
                    tx_request.attempts += 1
                    if tx_request.attempts < tx_request.max_attempts:
                        # Re-queue with lower priority
                        tx_request.priority += 1
                        with self._queue_lock:
                            heapq.heappush(self._tx_queue, tx_request)
                    else:
                        # Max attempts reached
                        logger.error(f"Max attempts reached for tx: {tx_request.contract_name}.{tx_request.function_name}")
                        tx_request.status = TransactionStatus.FAILED
                        tx_request.error = Exception("Max attempts reached")
                        
                        # Call on_submit callback with error if exists
                        if hasattr(tx_request, 'on_submit') and callable(tx_request.on_submit):
                            tx_request.on_submit(None, tx_request.error)
                        
                        # Call callback with error if exists
                        if tx_request.callback:
                            tx_request.callback(None, tx_request.error)
                        
                        # Update stats
                        self._tx_stats["failed"] += 1
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
                
                # Update request
                tx_request.attempts += 1
                tx_request.error = e
                
                if tx_request.attempts < tx_request.max_attempts:
                    # Re-queue with lower priority
                    tx_request.priority += 1
                    with self._queue_lock:
                        heapq.heappush(self._tx_queue, tx_request)
                        
                    # Update stats
                    self._tx_stats["retried"] += 1
                else:
                    # Max attempts reached
                    tx_request.status = TransactionStatus.FAILED
                    
                    # Call on_submit callback with error if exists
                    if hasattr(tx_request, 'on_submit') and callable(tx_request.on_submit):
                        tx_request.on_submit(None, e)
                    
                    # Call callback with error if exists
                    if tx_request.callback:
                        tx_request.callback(None, e)
                    
                    # Update stats
                    self._tx_stats["failed"] += 1
    
    def _submit_transaction(self, tx_request: TransactionRequest) -> Optional[str]:
        """
        Submit a transaction to the blockchain.
        
        Args:
            tx_request: Transaction request to submit
            
        Returns:
            Transaction hash if successful, None otherwise
        """
        try:
            # Get contract instance
            contract = self.contracts.get(tx_request.contract_name)
            if not contract:
                raise ValueError(f"Contract '{tx_request.contract_name}' not found")
            
            # Get function
            contract_function = getattr(contract.functions, tx_request.function_name)
            if not contract_function:
                raise ValueError(f"Function '{tx_request.function_name}' not found in contract '{tx_request.contract_name}'")
            
            # Build function call
            function_call = contract_function(*tx_request.args, **tx_request.kwargs)
            
            # Get nonce
            nonce = self.get_next_nonce()
            
            # Get fee parameters based on network and retry count
            if tx_request.attempts > 0:
                # Increase gas price for retries
                gas_increase = tx_request.attempts * self.gas_price_bump_percent
                fee_params = self.prepare_eip1559_fee_params(gas_increase)
            else:
                fee_params = self.prepare_eip1559_fee_params()
            
            # Estimate gas (with buffer)
            try:
                gas_estimate = function_call.estimate_gas({
                    'from': self.account.address,
                    **fee_params
                })
                gas_limit = int(gas_estimate * 1.2)  # Add 20% buffer
            except Exception as e:
                logger.warning(f"Gas estimation failed: {e}. Using default gas limit.")
                gas_limit = 3000000  # Default gas limit
            
            # Build transaction
            tx = function_call.build_transaction({
                'from': self.account.address,
                'gas': gas_limit,
                'nonce': nonce,
                'chainId': self.chain_id,
                **fee_params
            })
            
            # Sign transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, private_key=self.account.key)
            
            # Send transaction
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            hex_tx_hash = self.web3.to_hex(tx_hash)
            
            logger.info(f"Transaction sent: {hex_tx_hash}")
            return hex_tx_hash
            
        except Exception as e:
            logger.error(f"Failed to submit transaction: {e}")
            return None
    
    def _check_confirmations(self):
        """Check confirmations for active transactions."""
        if not self._active_txs:
            return
        
        current_time = time.time()
        completed_txs = []
        
        for tx_hash, tx_request in self._active_txs.items():
            try:
                # Check if transaction is confirmed
                try:
                    receipt = self.web3.eth.get_transaction_receipt(tx_hash)
                    
                    if receipt:
                        # Check if enough confirmations
                        if self.web3.eth.block_number >= receipt.blockNumber + self.confirmation_blocks:
                            # Transaction confirmed
                            confirmation_time = current_time - tx_request.submitted_at
                            
                            # Update metrics
                            self._tx_stats["confirmed"] += 1
                            self._tx_stats["total_confirmation_time"] += confirmation_time
                            self._tx_stats["avg_confirmation_time"] = (
                                self._tx_stats["total_confirmation_time"] / self._tx_stats["confirmed"]
                            )
                            
                            # Check if transaction was successful
                            if receipt.status == 1:
                                # Success
                                logger.info(f"Transaction confirmed: {tx_hash} (took {confirmation_time:.2f}s)")
                                tx_request.status = TransactionStatus.CONFIRMED
                                tx_request.result = receipt
                                
                                # Call callback if exists
                                if tx_request.callback:
                                    tx_request.callback(receipt)
                            else:
                                # Failed
                                logger.warning(f"Transaction failed: {tx_hash}")
                                tx_request.status = TransactionStatus.FAILED
                                tx_request.error = Exception("Transaction failed (status=0)")
                                
                                # Call callback with error if exists
                                if tx_request.callback:
                                    tx_request.callback(None, tx_request.error)
                            
                            # Mark for removal
                            completed_txs.append(tx_hash)
                
                except TransactionNotFound:
                    # Transaction not yet mined
                    # If it's been too long, consider it dropped
                    if current_time - tx_request.submitted_at > self.request_timeout:
                        logger.warning(f"Transaction may be dropped: {tx_hash}")
                        
                        # Resubmit with higher gas price
                        tx_request.attempts += 1
                        tx_request.status = TransactionStatus.DROPPED
                        
                        if tx_request.attempts < tx_request.max_attempts:
                            # Re-queue with higher priority
                            logger.info(f"Re-queuing dropped transaction: {tx_hash}")
                            tx_request.priority = 1  # Highest priority
                            with self._queue_lock:
                                heapq.heappush(self._tx_queue, tx_request)
                                
                            # Update stats
                            self._tx_stats["retried"] += 1
                        else:
                            # Max attempts reached
                            logger.error(f"Max attempts reached for dropped tx: {tx_hash}")
                            tx_request.status = TransactionStatus.FAILED
                            tx_request.error = Exception("Transaction dropped")
                            
                            # Call callback with error if exists
                            if tx_request.callback:
                                tx_request.callback(None, tx_request.error)
                            
                            # Update stats
                            self._tx_stats["failed"] += 1
                        
                        # Mark for removal
                        completed_txs.append(tx_hash)
            
            except Exception as e:
                logger.error(f"Error checking transaction {tx_hash}: {e}")
        
        # Remove completed transactions
        for tx_hash in completed_txs:
            if tx_hash in self._active_txs:
                del self._active_txs[tx_hash]
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the transaction queue.
        
        Returns:
            Dictionary with queue status
        """
        with self._queue_lock:
            queue_size = len(self._tx_queue)
        
        active_size = len(self._active_txs)
        
        return {
            "queue_size": queue_size,
            "active_transactions": active_size,
            "stats": self._tx_stats.copy()
        }
    
    def batch_transactions(self, batch_size: int = 5) -> bool:
        """
        Attempt to batch multiple transactions into a single multicall.
        
        Note: This is an advanced feature and requires a multicall contract.
        
        Args:
            batch_size: Maximum number of transactions to batch
            
        Returns:
            True if batching was successful
        """
        # This is a placeholder for multicall implementation
        # In a real implementation, this would use something like Gnosis MulticallV2
        return False


# Example usage
async def example_usage():
    # Set up Web3 provider
    from web3 import Web3
    web3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
    
    # Set up account
    from eth_account import Account
    private_key = "0x" + "1" * 64  # Replace with actual private key
    account = Account.from_key(private_key)
    
    # Load contract ABIs
    import json
    with open("ContributionLogging.json") as f:
        contribution_abi = json.load(f)["abi"]
    
    # Create contract instances
    contribution_contract = web3.eth.contract(
        address="0x1234567890123456789012345678901234567890",
        abi=contribution_abi
    )
    
    # Create contract dict
    contracts = {
        "ContributionLogging": contribution_contract
    }
    
    # Create transaction manager
    tx_manager = BlockchainTransactionManager(
        web3=web3,
        account=account,
        chain_id=1337,  # Local testnet
        contracts=contracts
    )
    
    # Start the transaction manager
    tx_manager.start()
    
    try:
        # Send a transaction - fix the positional arguments issue
        tx_hash = await tx_manager.send_transaction(
            "ContributionLogging",
            "logContribution",
            "contribution_123",
            "client_123",
            1,
            '{"accuracy": 0.85}',
            "0.1.0",
            "0x123..."
        )
        
        print(f"Transaction sent: {tx_hash}")
        
        # Wait for it to be mined
        receipt = await web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Transaction mined: {receipt.blockNumber}")
        
        # Get queue status
        status = tx_manager.get_queue_status()
        print(f"Queue status: {status}")
        
    finally:
        # Stop the transaction manager
        tx_manager.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
