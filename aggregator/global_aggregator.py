"""
Global Aggregator Module for Decentralized Federated Learning Platform

This module implements the global model aggregation logic with quality verification.
It collects model updates from clients, verifies their quality, aggregates them using
federated averaging or other methods, and creates new global model versions.
"""

import os
import numpy as np
import tensorflow as tf
import yaml
import logging
import json
from typing import Dict, Any, List, Tuple, Optional
import time
from collections import defaultdict
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GlobalAggregator:
    """
    Implements global model aggregation with quality verification.
    """
    
    def __init__(self, config_path: str = "aggregator_config.yaml"):
        """
        Initialize the global aggregator with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.global_model = None
        self.current_round = 0
        self.contributions = {}
        self.model_version = "0.0.1"
        self.lock = threading.Lock()  # For thread safety
        
        # Load or initialize global model
        self._initialize_global_model()
        
        logger.info(f"Initialized GlobalAggregator with model version: {self.model_version}")
    
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
                "aggregation": {
                    "method": "fedavg",  # Options: fedavg, weighted_average, median
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
                    "architecture": "cnn",
                    "input_shape": [28, 28, 1],
                    "save_path": "./models/global/"
                },
                "blockchain": {
                    "enabled": True,
                    "provider_url": "http://localhost:8545",
                    "contract_address": "0x123...",
                    "gas_limit": 3000000
                }
            }
    
    def _initialize_global_model(self) -> None:
        """Initialize or load the global model"""
        architecture = self.config['model']['architecture']
        logger.info(f"Initializing global model with {architecture} architecture")
        
        try:
            # Check if we have a saved model
            save_path = self.config['model']['save_path']
            model_path = os.path.join(save_path, f"global_model_{self.model_version}")
            
            if os.path.exists(model_path):
                logger.info(f"Loading saved model from {model_path}")
                self.global_model = tf.keras.models.load_model(model_path)
            else:
                logger.info("No saved model found. Initializing new model.")
                self._build_new_model()
                
                # Create directory if it doesn't exist
                os.makedirs(save_path, exist_ok=True)
                
                # Save the initial model
                self.global_model.save(model_path)
                
            logger.info("Global model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize global model: {e}")
            raise
    
    def _build_new_model(self) -> None:
        """Build a new model based on configuration"""
        architecture = self.config['model']['architecture']
        
        if architecture == 'cnn':
            # Example CNN for image classification
            self.global_model = tf.keras.Sequential([
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
            self.global_model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(self.config['model']['input_shape'][0],)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
            
        # Compile the model
        self.global_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def receive_contribution(self, client_id: str, metrics: Dict[str, float], model_update: Dict) -> Dict[str, Any]:
        """
        Receive and process a model update contribution from a client.
        
        Args:
            client_id: Unique identifier for the contributing client
            metrics: Dictionary of training metrics
            model_update: Dictionary containing model weight updates and metadata
            
        Returns:
            Dictionary with status and contribution ID
        """
        logger.info(f"Received contribution from client {client_id}")
        
        with self.lock:
            # Generate a unique contribution ID
            contribution_id = f"{client_id}_{self.current_round}_{int(time.time())}"
            
            # Store the contribution
            self.contributions[contribution_id] = {
                'client_id': client_id,
                'round': self.current_round,
                'timestamp': time.time(),
                'metrics': metrics,
                'model_update': model_update,
                'verified': False,
                'included_in_aggregation': False
            }
            
            # Verify contribution quality if enabled
            if self.config['quality_verification']['enabled']:
                verification_result = self._verify_contribution_quality(contribution_id)
                self.contributions[contribution_id]['verification_result'] = verification_result
                self.contributions[contribution_id]['verified'] = verification_result['accepted']
                
                # Log the verification result
                if verification_result['accepted']:
                    logger.info(f"Contribution {contribution_id} passed quality verification")
                else:
                    logger.warning(f"Contribution {contribution_id} failed quality verification: {verification_result['reason']}")
            else:
                # If verification is disabled, accept all contributions
                self.contributions[contribution_id]['verified'] = True
                self.contributions[contribution_id]['verification_result'] = {'accepted': True, 'reason': 'Verification disabled'}
            
            # Record the contribution on blockchain if enabled
            if self.config['blockchain']['enabled'] and self.contributions[contribution_id]['verified']:
                try:
                    # This is a placeholder for actual blockchain integration
                    # In a real implementation, this would interact with a smart contract
                    blockchain_tx = self._record_contribution_on_blockchain(contribution_id)
                    self.contributions[contribution_id]['blockchain_tx'] = blockchain_tx
                    logger.info(f"Recorded contribution {contribution_id} on blockchain: {blockchain_tx}")
                except Exception as e:
                    logger.error(f"Failed to record contribution on blockchain: {e}")
            
            # Trigger aggregation if we have enough contributions
            if self._should_aggregate():
                threading.Thread(target=self.aggregate_contributions).start()
            
            # Return the status
            return {
                'status': 'success',
                'contribution_id': contribution_id,
                'verified': self.contributions[contribution_id]['verified'],
                'current_round': self.current_round,
                'model_version': self.model_version
            }
    
    def _verify_contribution_quality(self, contribution_id: str) -> Dict[str, Any]:
        """
        Verify the quality of a contribution.
        
        Args:
            contribution_id: ID of the contribution to verify
            
        Returns:
            Dictionary with verification results
        """
        contribution = self.contributions[contribution_id]
        metrics = contribution['metrics']
        model_update = contribution['model_update']
        
        # Basic quality checks
        if metrics['loss'] > self.config['quality_verification']['max_loss']:
            return {'accepted': False, 'reason': f"Loss too high: {metrics['loss']}"}
        
        if metrics.get('accuracy', 0) < self.config['quality_verification']['min_accuracy']:
            return {'accepted': False, 'reason': f"Accuracy too low: {metrics.get('accuracy', 0)}"}
        
        # Outlier detection if enabled
        if self.config['quality_verification']['outlier_detection']:
            try:
                # This is a basic check that could be replaced with more sophisticated outlier detection
                is_outlier, outlier_score = self._detect_outlier(model_update)
                if is_outlier:
                    return {'accepted': False, 'reason': f"Detected as outlier with score: {outlier_score}"}
            except Exception as e:
                logger.error(f"Error in outlier detection: {e}")
        
        # All checks passed
        return {'accepted': True, 'reason': "All quality checks passed"}
    
    def _detect_outlier(self, model_update: Dict) -> Tuple[bool, float]:
        """
        Detect if a model update is an outlier.
        
        Args:
            model_update: Dictionary containing model weight updates
            
        Returns:
            Tuple of (is_outlier, outlier_score)
        """
        # Get all verified contributions from this round
        verified_contributions = [
            c for c in self.contributions.values() 
            if c['verified'] and c['round'] == self.current_round
        ]
        
        if len(verified_contributions) < 2:
            # Not enough data to detect outliers
            return False, 0.0
        
        # Calculate cosine similarity with other updates
        similarities = []
        
        for contrib in verified_contributions:
            # Skip comparing with itself
            if contrib['model_update'] is model_update:
                continue
                
            # Calculate cosine similarity for each layer
            layer_similarities = []
            for i, layer_update in enumerate(model_update['weights']):
                other_update = contrib['model_update']['weights'][i]
                
                # Flatten the arrays for cosine similarity calculation
                flat_update = layer_update.flatten()
                flat_other = other_update.flatten()
                
                # Calculate cosine similarity
                similarity = np.dot(flat_update, flat_other) / (
                    np.linalg.norm(flat_update) * np.linalg.norm(flat_other) + 1e-9
                )
                layer_similarities.append(similarity)
            
            # Average similarity across layers
            avg_similarity = np.mean(layer_similarities)
            similarities.append(avg_similarity)
        
        # Calculate the average similarity
        avg_similarity = np.mean(similarities)
        
        # Check if below threshold
        threshold = self.config['quality_verification']['cosine_similarity_threshold']
        is_outlier = avg_similarity < threshold
        
        return is_outlier, avg_similarity
    
    def _record_contribution_on_blockchain(self, contribution_id: str) -> str:
        """
        Record contribution metadata on blockchain.
        
        Args:
            contribution_id: ID of the contribution to record
            
        Returns:
            Transaction hash or identifier
        """
        # This is a placeholder for actual blockchain interaction
        # In a real implementation, this would use web3.py or similar to interact with smart contracts
        
        contribution = self.contributions[contribution_id]
        
        # Prepare data for blockchain
        blockchain_data = {
            'client_id': contribution['client_id'],
            'round': contribution['round'],
            'timestamp': contribution['timestamp'],
            'metrics': {
                'loss': contribution['metrics']['loss'],
                'accuracy': contribution['metrics'].get('accuracy', 0),
                'dataset_size': contribution['metrics'].get('dataset_size', 0)
            },
            'model_version': self.model_version,
            # We would add a hash of model updates instead of the actual weights
            'update_hash': self._hash_model_update(contribution['model_update']['weights'])
        }
        
        # Mock blockchain transaction
        tx_hash = f"0x{os.urandom(32).hex()}"
        
        # In a real implementation, we would use web3.py:
        """
        from web3 import Web3
        
        # Connect to Ethereum node
        w3 = Web3(Web3.HTTPProvider(self.config['blockchain']['provider_url']))
        
        # Load contract ABI and address
        with open('ContributionLogging.json') as f:
            contract_json = json.load(f)
        contract_abi = contract_json['abi']
        contract_address = self.config['blockchain']['contract_address']
        
        # Create contract instance
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        
        # Prepare transaction
        tx = contract.functions.logContribution(
            contribution['client_id'],
            contribution['round'],
            int(contribution['timestamp']),
            json.dumps(blockchain_data['metrics']),
            self.model_version,
            blockchain_data['update_hash']
        ).build_transaction({
            'from': w3.eth.accounts[0],
            'gas': self.config['blockchain']['gas_limit'],
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(w3.eth.accounts[0])
        })
        
        # Sign and send transaction
        signed_tx = w3.eth.account.sign_transaction(tx, private_key='your_private_key')
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for transaction receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_hash.hex()
        """
        
        return tx_hash
    
    def _hash_model_update(self, model_update: List[np.ndarray]) -> str:
        """
        Create a hash of model update weights.
        
        Args:
            model_update: List of weight arrays
            
        Returns:
            Hash string
        """
        import hashlib
        
        # Concatenate all weights into a single byte array
        all_weights = b''
        for layer_weights in model_update:
            # Convert numpy array to bytes
            all_weights += layer_weights.tobytes()
        
        # Create SHA-256 hash
        hasher = hashlib.sha256()
        hasher.update(all_weights)
        
        return hasher.hexdigest()
    
    def _should_aggregate(self) -> bool:
        """
        Determine if we should trigger aggregation based on current contributions.
        
        Returns:
            Boolean indicating if aggregation should be performed
        """
        # Count verified contributions in the current round
        verified_count = sum(
            1 for c in self.contributions.values() 
            if c['verified'] and c['round'] == self.current_round and not c['included_in_aggregation']
        )
        
        # Check if we have enough contributions
        return verified_count >= self.config['aggregation']['min_contributions']
        
    def aggregate_contributions(self) -> Dict[str, Any]:
        """
        Aggregate verified contributions to create a new global model.
        
        Returns:
            Dictionary with aggregation results
        """
        logger.info("Starting contribution aggregation")
        
        with self.lock:
            # Get all verified contributions for this round that haven't been included yet
            contributions_to_aggregate = [
                c for c in self.contributions.values()
                if c['verified'] and c['round'] == self.current_round and not c['included_in_aggregation']
            ]
            
            if not contributions_to_aggregate:
                logger.warning("No contributions to aggregate")
                return {'status': 'failed', 'reason': 'No contributions to aggregate'}
            
            logger.info(f"Aggregating {len(contributions_to_aggregate)} contributions")
            
            try:
                # Choose aggregation method based on configuration
                aggregation_method = self.config['aggregation']['method']
                
                if aggregation_method == 'fedavg':
                    new_weights = self._federated_averaging(contributions_to_aggregate)
                elif aggregation_method == 'weighted_average':
                    new_weights = self._weighted_averaging(contributions_to_aggregate)
                elif aggregation_method == 'median':
                    new_weights = self._median_aggregation(contributions_to_aggregate)
                else:
                    raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
                
                # Apply the new weights to the global model
                for i, w in enumerate(new_weights):
                    self.global_model.weights[i].assign(w)
                
                # Update model version and round
                self.current_round += 1
                if self.current_round % self.config['aggregation']['rounds_per_epoch'] == 0:
                    # Increment version for every N rounds
                    major, minor, patch = self.model_version.split('.')
                    if int(minor) < 9:
                        minor = str(int(minor) + 1)
                    else:
                        major = str(int(major) + 1)
                        minor = '0'
                    self.model_version = f"{major}.{minor}.{patch}"
                
                # Mark contributions as included
                for c in contributions_to_aggregate:
                    c['included_in_aggregation'] = True
                
                # Save the updated global model
                save_path = os.path.join(
                    self.config['model']['save_path'], 
                    f"global_model_{self.model_version}"
                )
                self.global_model.save(save_path)
                
                # Record aggregation on blockchain if enabled
                if self.config['blockchain']['enabled']:
                    aggregation_tx = self._record_aggregation_on_blockchain(
                        [c['contribution_id'] for c in contributions_to_aggregate]
                    )
                    logger.info(f"Recorded aggregation on blockchain: {aggregation_tx}")
                
                logger.info(f"Aggregation complete. New model version: {self.model_version}, Round: {self.current_round}")
                
                return {
                    'status': 'success',
                    'model_version': self.model_version,
                    'round': self.current_round,
                    'contributions_included': len(contributions_to_aggregate)
                }
                
            except Exception as e:
                logger.error(f"Aggregation failed: {e}")
                return {'status': 'failed', 'reason': str(e)}
    
    def _federated_averaging(self, contributions: List[Dict]) -> List[np.ndarray]:
        """
        Perform federated averaging on the model updates.
        
        Args:
            contributions: List of contribution dictionaries
            
        Returns:
            List of averaged weight arrays
        """
        # Get the current global weights
        global_weights = [w.numpy() for w in self.global_model.weights]
        
        # Initialize accumulators for the updates
        weight_updates = [np.zeros_like(w) for w in global_weights]
        total_samples = 0
        
        # Accumulate weighted updates
        for contrib in contributions:
            # Get the update and number of samples
            update = contrib['model_update']['weights']
            n_samples = contrib['metrics'].get('dataset_size', 1)
            total_samples += n_samples
            
            # Add weighted update to accumulators
            for i, layer_update in enumerate(update):
                weight_updates[i] += layer_update * n_samples
        
        # Compute the weighted average
        avg_updates = [w / total_samples for w in weight_updates]
        
        # Apply the average updates to the global weights
        new_weights = [global_weights[i] + avg_updates[i] for i in range(len(global_weights))]
        
        return new_weights
    
    def _weighted_averaging(self, contributions: List[Dict]) -> List[np.ndarray]:
        """
        Perform weighted averaging based on metrics like accuracy.
        
        Args:
            contributions: List of contribution dictionaries
            
        Returns:
            List of weighted averaged weight arrays
        """
        # Get the current global weights
        global_weights = [w.numpy() for w in self.global_model.weights]
        
        # Calculate weights based on accuracy
        accuracies = [contrib['metrics'].get('accuracy', 0.5) for contrib in contributions]
        total_accuracy = sum(accuracies)
        
        if total_accuracy == 0:
            # If all accuracies are 0, use equal weights
            weights = [1.0 / len(contributions)] * len(contributions)
        else:
            # Normalize weights by accuracy
            weights = [acc / total_accuracy for acc in accuracies]
        
        # Initialize new weights with zeros
        new_weights = [np.zeros_like(w) for w in global_weights]
        
        # Accumulate weighted updates
        for i, contrib in enumerate(contributions):
            update = contrib['model_update']['weights']
            for j, layer_update in enumerate(update):
                # Add the update to the global weights to get the client's version
                client_weights = global_weights[j] + layer_update
                # Add weighted contribution to new weights
                new_weights[j] += client_weights * weights[i]
        
        return new_weights
    
    def _median_aggregation(self, contributions: List[Dict]) -> List[np.ndarray]:
        """
        Perform median-based aggregation which is more robust to outliers.
        
        Args:
            contributions: List of contribution dictionaries
            
        Returns:
            List of median aggregated weight arrays
        """
        # Get the current global weights
        global_weights = [w.numpy() for w in self.global_model.weights]
        
        # Collect all client weights
        all_weights = []
        
        for contrib in contributions:
            update = contrib['model_update']['weights']
            # Convert updates to client weights
            client_weights = [global_weights[i] + update[i] for i in range(len(global_weights))]
            all_weights.append(client_weights)
        
        # Compute element-wise median
        new_weights = []
        for i in range(len(global_weights)):
            # Collect i-th layer weights from all clients
            layer_weights = [client[i] for client in all_weights]
            # Stack along a new axis for element-wise median
            stacked = np.stack(layer_weights, axis=0)
            # Compute median along client axis
            median_weights = np.median(stacked, axis=0)
            new_weights.append(median_weights)
        
        return new_weights
    
    def _record_aggregation_on_blockchain(self, contribution_ids: List[str]) -> str:
        """
        Record aggregation metadata on blockchain.
        
        Args:
            contribution_ids: List of contribution IDs included in aggregation
            
        Returns:
            Transaction hash or identifier
        """
        # This is a placeholder for actual blockchain interaction
        # In a real implementation, this would use web3.py or similar
        
        aggregation_data = {
            'model_version': self.model_version,
            'round': self.current_round,
            'timestamp': time.time(),
            'contribution_count': len(contribution_ids),
            'contribution_ids': contribution_ids,
            # We would add a hash of the model weights
            'model_hash': self._hash_model_weights(self.global_model.weights)
        }
        
        # Mock blockchain transaction
        tx_hash = f"0x{os.urandom(32).hex()}"
        
        return tx_hash
    
    def _hash_model_weights(self, weights: List[tf.Variable]) -> str:
        """
        Create a hash of model weights.
        
        Args:
            weights: List of weight tensors
            
        Returns:
            Hash string
        """
        import hashlib
        
        # Concatenate all weights into a single byte array
        all_weights = b''
        for w in weights:
            all_weights += w.numpy().tobytes()
        
        # Create SHA-256 hash
        hasher = hashlib.sha256()
        hasher.update(all_weights)
        
        return hasher.hexdigest()
    
    def get_global_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current global model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_version': self.model_version,
            'current_round': self.current_round,
            'architecture': self.config['model']['architecture'],
            'input_shape': self.config['model']['input_shape'],
            'aggregation_method': self.config['aggregation']['method'],
            'last_updated': time.time()
        }
    
    def get_global_model_weights(self) -> List[np.ndarray]:
        """
        Get the current global model weights.
        
        Returns:
            List of weight arrays
        """
        return [w.numpy() for w in self.global_model.weights]
    
    def get_contribution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about contributions.
        
        Returns:
            Dictionary with contribution statistics
        """
        total_contributions = len(self.contributions)
        verified_contributions = sum(1 for c in self.contributions.values() if c['verified'])
        included_contributions = sum(1 for c in self.contributions.values() if c['included_in_aggregation'])
        
        # Group by round
        by_round = defaultdict(int)
        for c in self.contributions.values():
            by_round[c['round']] += 1
        
        # Group by client
        by_client = defaultdict(int)
        for c in self.contributions.values():
            by_client[c['client_id']] += 1
        
        return {
            'total_contributions': total_contributions,
            'verified_contributions': verified_contributions,
            'included_contributions': included_contributions,
            'by_round': dict(by_round),
            'by_client': dict(by_client),
            'current_round': self.current_round
        }
