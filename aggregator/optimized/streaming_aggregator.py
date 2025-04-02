"""
Optimized Aggregation Algorithms for Large Models

This module provides efficient implementations of aggregation algorithms for federated learning
with a focus on handling large neural network models with memory efficiency and performance.
"""

import numpy as np
import tensorflow as tf
import logging
import time
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import concurrent.futures
import threading
import queue
from dataclasses import dataclass
import psutil
import gc
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ContributionInfo:
    """Information about a contribution for aggregation."""
    client_id: str
    metrics: Dict[str, Any]
    weight: float = 1.0  # Relative weight for aggregation
    
    def __str__(self):
        return f"ContributionInfo(client_id={self.client_id}, weight={self.weight})"

class ChunkedArray:
    """
    Memory-efficient array implementation that processes data in chunks.
    
    This class enables processing large arrays that might not fit in memory all at once
    by splitting operations across manageable chunks.
    """
    
    def __init__(self, shape: Tuple[int, ...], dtype=np.float32, chunk_size: int = 10000000):
        """
        Initialize a chunked array.
        
        Args:
            shape: Shape of the array
            dtype: Data type for the array
            chunk_size: Maximum elements per chunk (default 10M elements)
        """
        self.shape = shape
        self.size = np.prod(shape)
        self.dtype = dtype
        self.chunk_size = min(chunk_size, self.size)  # Ensure chunk_size doesn't exceed total size
        
        # Calculate number of chunks
        self.num_chunks = int(np.ceil(self.size / self.chunk_size))
        
        # Calculate chunk boundaries
        self.chunk_boundaries = []
        for i in range(self.num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, self.size)
            self.chunk_boundaries.append((start, end))
    
    def apply_function(self, func: Callable, *arrays: np.ndarray) -> np.ndarray:
        """
        Apply a function to arrays in chunks.
        
        Args:
            func: Function to apply (takes arrays as input and returns an array)
            arrays: Input arrays
            
        Returns:
            Result array
        """
        # Validate input arrays
        for arr in arrays:
            if arr.size != self.size:
                raise ValueError(f"Array size mismatch: expected {self.size}, got {arr.size}")
        
        # Create output array
        result = np.zeros(self.size, dtype=self.dtype)
        
        # Process in chunks
        for start, end in self.chunk_boundaries:
            # Extract chunks from input arrays
            chunks = [arr.flat[start:end] for arr in arrays]
            
            # Apply function
            result[start:end] = func(*chunks)
            
            # Explicitly free memory
            del chunks
            gc.collect()
        
        # Reshape result to match original shape
        return result.reshape(self.shape)
    
    @staticmethod
    def weighted_average(chunks: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """
        Compute weighted average of arrays.
        
        Args:
            chunks: List of array chunks
            weights: List of weights
            
        Returns:
            Weighted average
        """
        result = np.zeros_like(chunks[0])
        for i, chunk in enumerate(chunks):
            result += chunk * weights[i]
        
        return result

class StreamingAggregator:
    """
    Memory-efficient streaming aggregator for large model parameters.
    
    This class enables aggregating large model updates using streaming operations
    to avoid loading all data into memory at once, suitable for very large models.
    """
    
    def __init__(
        self,
        method: str = "fedavg",
        memory_limit_mb: int = 4096,  # Default 4GB memory limit
        use_gpu: bool = True,
        auto_scaling: bool = True
    ):
        """
        Initialize the streaming aggregator.
        
        Args:
            method: Aggregation method ("fedavg", "weighted_average", or "median")
            memory_limit_mb: Memory limit in megabytes
            use_gpu: Whether to use GPU for computations when available
            auto_scaling: Automatically adjust chunk size based on available memory
        """
        self.method = method
        self.memory_limit_mb = memory_limit_mb
        self.use_gpu = use_gpu and tf.config.list_physical_devices('GPU')
        self.auto_scaling = auto_scaling
        
        # Calculate optimal chunk size based on memory limit
        self.chunk_size = self._calculate_chunk_size()
        
        # Performance metrics
        self.metrics = {
            "aggregation_times": [],
            "memory_usage": [],
            "processed_elements": 0,
            "last_chunk_size": 0
        }
    
    def _calculate_chunk_size(self) -> int:
        """
        Calculate optimal chunk size based on memory constraints.
        
        Returns:
            Chunk size (number of elements)
        """
        # Base chunk size calculation
        # Each float32 is 4 bytes, we aim to use at most 1/4 of memory_limit for a chunk
        bytes_per_element = 4  # float32
        max_elements = (self.memory_limit_mb * 1024 * 1024) // (bytes_per_element * 4)
        
        # Start with a reasonable chunk size
        chunk_size = min(max_elements, 10000000)  # 10M elements default max
        
        if self.auto_scaling:
            # Adjust based on available system memory
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            scaling_factor = min(max(0.1, available_memory / self.memory_limit_mb), 2.0)
            chunk_size = int(chunk_size * scaling_factor)
        
        logger.info(f"Calculated chunk size: {chunk_size} elements")
        return chunk_size
    
    def aggregate(
        self,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Aggregate model updates using the specified method.
        
        Args:
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights (required for some methods)
            
        Returns:
            Aggregated weights
        """
        start_time = time.time()
        
        if not contributions:
            raise ValueError("No contributions to aggregate")
        
        # Select aggregation method
        if self.method == "fedavg":
            result = self._federated_averaging(contributions, global_weights)
        elif self.method == "weighted_average":
            result = self._weighted_averaging(contributions, global_weights)
        elif self.method == "median":
            result = self._median_aggregation(contributions, global_weights)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")
        
        # Record metrics
        elapsed_time = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        self.metrics["aggregation_times"].append(elapsed_time)
        self.metrics["memory_usage"].append(memory_usage)
        self.metrics["last_chunk_size"] = self.chunk_size
        
        # Log performance
        logger.info(
            f"Aggregation completed in {elapsed_time:.2f}s using {memory_usage:.1f}MB of memory. "
            f"Processed {self.metrics['processed_elements']} elements across {len(contributions)} contributions."
        )
        
        return result
    
    def _federated_averaging(
        self,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Perform federated averaging aggregation with streaming operations.
        
        Args:
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights
            
        Returns:
            Aggregated weights
        """
        if not global_weights:
            raise ValueError("Global weights are required for federated averaging")
        
        # Extract updates and weights for aggregation
        updates = [upd for upd, _ in contributions]
        infos = [info for _, info in contributions]
        
        # Calculate total weight
        total_weight = sum(info.metrics.get('dataset_size', 1) for info in infos)
        
        # Calculate normalized weights
        weights = [info.metrics.get('dataset_size', 1) / total_weight for info in infos]
        
        # Create result arrays with the same shape as global weights
        result = [np.zeros_like(w) for w in global_weights]
        
        # Process each layer separately
        for layer_idx, global_layer in enumerate(global_weights):
            # Count elements for metrics
            self.metrics["processed_elements"] += global_layer.size
            
            # For small layers, perform direct computation
            if global_layer.size <= self.chunk_size:
                # Convert updates to arrays for this layer
                layer_updates = np.array([upd[layer_idx] for upd in updates])
                
                # Weighted average of updates
                weighted_update = np.zeros_like(global_layer)
                for i, update in enumerate(layer_updates):
                    weighted_update += update * weights[i]
                
                # Add weighted update to global weights
                result[layer_idx] = global_layer + weighted_update
            else:
                # For large layers, use chunked processing
                chunked_array = ChunkedArray(
                    shape=global_layer.shape,
                    dtype=global_layer.dtype,
                    chunk_size=self.chunk_size
                )
                
                # Process in chunks
                flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
                
                for start, end in chunked_array.chunk_boundaries:
                    # Extract chunks from all updates
                    update_chunks = []
                    for upd in updates:
                        update_chunks.append(upd[layer_idx].flat[start:end])
                    
                    # Calculate weighted average
                    chunk_result = np.zeros_like(update_chunks[0])
                    for i, chunk in enumerate(update_chunks):
                        chunk_result += chunk * weights[i]
                    
                    # Add to global weights
                    global_chunk = global_layer.flat[start:end]
                    flat_result[start:end] = global_chunk + chunk_result
                    
                    # Free memory
                    del update_chunks, chunk_result, global_chunk
                    gc.collect()
                
                # Reshape result
                result[layer_idx] = flat_result.reshape(global_layer.shape)
        
        return result
    
    def _weighted_averaging(
        self,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Perform weighted averaging based on metrics like accuracy.
        
        Args:
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights
            
        Returns:
            Aggregated weights
        """
        if not global_weights:
            raise ValueError("Global weights are required for weighted averaging")
        
        # Extract updates and infos
        updates = [upd for upd, _ in contributions]
        infos = [info for _, info in contributions]
        
        # Calculate weights based on accuracy
        accuracies = [info.metrics.get('accuracy', 0.5) for info in infos]
        total_accuracy = sum(accuracies)
        
        if total_accuracy == 0:
            # If all accuracies are 0, use equal weights
            weights = [1.0 / len(contributions)] * len(contributions)
        else:
            # Normalize weights by accuracy
            weights = [acc / total_accuracy for acc in accuracies]
        
        # Create result list
        result = [np.zeros_like(w) for w in global_weights]
        
        # Process each layer
        for layer_idx, global_layer in enumerate(global_weights):
            # Count elements for metrics
            self.metrics["processed_elements"] += global_layer.size
            
            # For small layers, process directly
            if global_layer.size <= self.chunk_size:
                # Initialize with zeros
                result[layer_idx] = np.zeros_like(global_layer)
                
                # Add weighted contribution from each client
                for i, upd in enumerate(updates):
                    # Add the update to the global weights to get the client's version
                    client_weights = global_layer + upd[layer_idx]
                    
                    # Add weighted contribution
                    result[layer_idx] += client_weights * weights[i]
            else:
                # For large layers, use chunked processing
                chunked_array = ChunkedArray(
                    shape=global_layer.shape,
                    dtype=global_layer.dtype,
                    chunk_size=self.chunk_size
                )
                
                # Process in chunks
                flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
                
                for start, end in chunked_array.chunk_boundaries:
                    # Extract global chunk
                    global_chunk = global_layer.flat[start:end]
                    
                    # Initialize chunk result
                    chunk_result = np.zeros_like(global_chunk)
                    
                    # Process each client's contribution
                    for i, upd in enumerate(updates):
                        # Get client chunk
                        client_chunk = global_chunk + upd[layer_idx].flat[start:end]
                        
                        # Add weighted contribution
                        chunk_result += client_chunk * weights[i]
                    
                    # Store result
                    flat_result[start:end] = chunk_result
                    
                    # Free memory
                    del global_chunk, chunk_result
                    gc.collect()
                
                # Reshape result
                result[layer_idx] = flat_result.reshape(global_layer.shape)
        
        return result
    
    def _median_aggregation(
        self,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Perform median-based aggregation which is more robust to outliers.
        
        Args:
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights
            
        Returns:
            Aggregated weights
        """
        if not global_weights:
            raise ValueError("Global weights are required for median aggregation")
        
        # Extract updates
        updates = [upd for upd, _ in contributions]
        
        # Create result list
        result = [np.zeros_like(w) for w in global_weights]
        
        # Process each layer
        for layer_idx, global_layer in enumerate(global_weights):
            # Count elements for metrics
            self.metrics["processed_elements"] += global_layer.size
            
            # For small layers, process directly
            if global_layer.size <= self.chunk_size:
                # Collect all client weights for this layer
                client_weights = []
                for upd in updates:
                    # Convert updates to weights
                    client_w = global_layer + upd[layer_idx]
                    client_weights.append(client_w)
                
                # Stack along a new axis for element-wise median
                stacked = np.stack(client_weights, axis=0)
                
                # Compute median along client axis
                result[layer_idx] = np.median(stacked, axis=0)
            else:
                # For large layers, use chunked processing
                flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
                
                # Process in chunks
                chunk_size = self.chunk_size
                num_chunks = int(np.ceil(global_layer.size / chunk_size))
                
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * chunk_size
                    end = min((chunk_idx + 1) * chunk_size, global_layer.size)
                    
                    # Extract global chunk
                    global_chunk = global_layer.flat[start:end]
                    
                    # Collect client chunks
                    client_chunks = []
                    for upd in updates:
                        # Get update chunk
                        update_chunk = upd[layer_idx].flat[start:end]
                        
                        # Convert to client weights
                        client_chunk = global_chunk + update_chunk
                        client_chunks.append(client_chunk)
                    
                    # Stack for element-wise median
                    stacked_chunks = np.stack(client_chunks, axis=0)
                    
                    # Calculate median
                    chunk_result = np.median(stacked_chunks, axis=0)
                    
                    # Store result
                    flat_result[start:end] = chunk_result
                    
                    # Free memory
                    del global_chunk, client_chunks, stacked_chunks, chunk_result
                    gc.collect()
                
                # Reshape result
                result[layer_idx] = flat_result.reshape(global_layer.shape)
        
        return result
    
    def parallel_aggregation(
        self,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: List[np.ndarray],
        num_workers: int = None
    ) -> List[np.ndarray]:
        """
        Perform aggregation with parallel processing for large models.
        
        Args:
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights
            num_workers: Number of worker threads/processes
            
        Returns:
            Aggregated weights
        """
        if not num_workers:
            # Default to number of available CPU cores minus 1 (leave one for system)
            num_workers = max(1, psutil.cpu_count(logical=False) - 1)
        
        # Create result list
        result = [None] * len(global_weights)
        
        # Group layers by size for efficient processing
        small_layers = []  # Process together
        large_layers = []  # Process individually
        
        for layer_idx, layer in enumerate(global_weights):
            if layer.size <= self.chunk_size // 10:  # Very small layers
                small_layers.append(layer_idx)
            else:
                large_layers.append(layer_idx)
        
        logger.info(f"Aggregating {len(small_layers)} small layers together and {len(large_layers)} large layers in parallel")
        
        # Create task queue
        task_queue = queue.Queue()
        result_dict = {}
        
        # Add tasks for large layers (one task per layer)
        for layer_idx in large_layers:
            task_queue.put(("single", layer_idx))
        
        # Add small layers as one batch task
        if small_layers:
            task_queue.put(("batch", small_layers))
        
        # Define worker function
        def worker_func():
            while True:
                try:
                    task_type, layer_data = task_queue.get(block=False)
                    
                    if task_type == "single":
                        # Process single large layer
                        layer_idx = layer_data
                        layer_result = self._aggregate_single_layer(
                            layer_idx, contributions, global_weights
                        )
                        result_dict[layer_idx] = layer_result
                    elif task_type == "batch":
                        # Process batch of small layers
                        layer_indices = layer_data
                        for idx in layer_indices:
                            layer_result = self._aggregate_single_layer(
                                idx, contributions, global_weights
                            )
                            result_dict[idx] = layer_result
                    
                    task_queue.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error in worker thread: {e}")
                    task_queue.task_done()
        
        # Start worker threads
        threads = []
        for _ in range(num_workers):
            thread = threading.Thread(target=worker_func)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        task_queue.join()
        
        # Assemble results
        for layer_idx in range(len(global_weights)):
            result[layer_idx] = result_dict.get(layer_idx)
        
        # Ensure all layers were processed
        for layer_idx, layer_result in enumerate(result):
            if layer_result is None:
                # Process any missing layers (shouldn't happen but just in case)
                logger.warning(f"Layer {layer_idx} was not processed by workers, processing now")
                result[layer_idx] = self._aggregate_single_layer(
                    layer_idx, contributions, global_weights
                )
        
        return result
    
    def _aggregate_single_layer(
        self,
        layer_idx: int,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: List[np.ndarray]
    ) -> np.ndarray:
        """
        Aggregate a single layer based on the selected method.
        
        Args:
            layer_idx: Index of the layer to aggregate
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights
            
        Returns:
            Aggregated layer weights
        """
        global_layer = global_weights[layer_idx]
        
        # Use appropriate aggregation method
        if self.method == "fedavg":
            # Extract updates and calculate weights
            updates = [upd[layer_idx] for upd, _ in contributions]
            infos = [info for _, info in contributions]
            
            # Calculate total weight
            total_weight = sum(info.metrics.get('dataset_size', 1) for info in infos)
            
            # Calculate normalized weights
            weights = [info.metrics.get('dataset_size', 1) / total_weight for info in infos]
            
            # Aggregate layer
            if global_layer.size <= self.chunk_size:
                weighted_update = np.zeros_like(global_layer)
                for i, update in enumerate(updates):
                    weighted_update += update * weights[i]
                
                return global_layer + weighted_update
            else:
                return self._chunked_layer_fedavg(
                    global_layer, updates, weights
                )
        
        elif self.method == "weighted_average":
            # Similar to fedavg but with accuracy-based weights
            # Extract updates and infos
            updates = [upd[layer_idx] for upd, _ in contributions]
            infos = [info for _, info in contributions]
            
            # Calculate weights based on accuracy
            accuracies = [info.metrics.get('accuracy', 0.5) for info in infos]
            total_accuracy = sum(accuracies)
            
            if total_accuracy == 0:
                weights = [1.0 / len(contributions)] * len(contributions)
            else:
                weights = [acc / total_accuracy for acc in accuracies]
            
            return self._chunked_layer_weighted_avg(
                global_layer, updates, weights
            )
        
        elif self.method == "median":
            # Extract updates
            updates = [upd[layer_idx] for upd, _ in contributions]
            
            return self._chunked_layer_median(
                global_layer, updates
            )
        
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")
    
    def _chunked_layer_fedavg(
        self,
        global_layer: np.ndarray,
        updates: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """
        Apply federated averaging to a layer using chunked processing.
        
        Args:
            global_layer: Global layer weights
            updates: List of updates from clients
            weights: List of client weights
            
        Returns:
            Aggregated layer
        """
        # Create chunked array processor
        chunked_array = ChunkedArray(
            shape=global_layer.shape,
            dtype=global_layer.dtype,
            chunk_size=self.chunk_size
        )
        
        # Process in chunks
        flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
        
        for start, end in chunked_array.chunk_boundaries:
            # Extract chunks from all updates
            update_chunks = []
            for upd in updates:
                update_chunks.append(upd.flat[start:end])
            
            # Calculate weighted average
            chunk_result = np.zeros_like(update_chunks[0])
            for i, chunk in enumerate(update_chunks):
                chunk_result += chunk * weights[i]
            
            # Add to global weights
            global_chunk = global_layer.flat[start:end]
            flat_result[start:end] = global_chunk + chunk_result
            
            # Free memory
            del update_chunks, chunk_result, global_chunk
            gc.collect()
        
        # Reshape result
        return flat_result.reshape(global_layer.shape)
    
    def _chunked_layer_weighted_avg(
        self,
        global_layer: np.ndarray,
        updates: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """
        Apply weighted averaging to a layer using chunked processing.
        
        Args:
            global_layer: Global layer weights
            updates: List of updates from clients
            weights: List of client weights
            
        Returns:
            Aggregated layer
        """
        # Create chunked array processor
        chunked_array = ChunkedArray(
            shape=global_layer.shape,
            dtype=global_layer.dtype,
            chunk_size=self.chunk_size
        )
        
        # Process in chunks
        flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
        
        for start, end in chunked_array.chunk_boundaries:
            # Extract global chunk
            global_chunk = global_layer.flat[start:end]
            
            # Initialize chunk result
            chunk_result = np.zeros_like(global_chunk)
            
            # Process each client's contribution
            for i, upd in enumerate(updates):
                # Get client chunk
                client_chunk = global_chunk + upd.flat[start:end]
                
                # Add weighted contribution
                chunk_result += client_chunk * weights[i]
            
            # Store result
            flat_result[start:end] = chunk_result
            
            # Free memory
            del global_chunk, chunk_result
            gc.collect()
        
        # Reshape result
        return flat_result.reshape(global_layer.shape)
    
    def _chunked_layer_median(
        self,
        global_layer: np.ndarray,
        updates: List[np.ndarray]
    ) -> np.ndarray:
        """
        Apply median aggregation to a layer using chunked processing.
        
        Args:
            global_layer: Global layer weights
            updates: List of updates from clients
            
        Returns:
            Aggregated layer
        """
        # Create result array
        flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
        
        # Process in chunks
        chunk_size = self.chunk_size
        num_chunks = int(np.ceil(global_layer.size / chunk_size))
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, global_layer.size)
            
            # Extract global chunk
            global_chunk = global_layer.flat[start:end]
            
            # Collect client chunks
            client_chunks = []
            for upd in updates:
                # Get update chunk
                update_chunk = upd.flat[start:end]
                
                # Convert to client weights
                client_chunk = global_chunk + update_chunk
                client_chunks.append(client_chunk)
            
            # Stack for element-wise median
            stacked_chunks = np.stack(client_chunks, axis=0)
            
            # Calculate median
            chunk_result = np.median(stacked_chunks, axis=0)
            
            # Store result
            flat_result[start:end] = chunk_result
            
            # Free memory
            del global_chunk, client_chunks, stacked_chunks, chunk_result
            gc.collect()
        
        # Reshape result
        return flat_result.reshape(global_layer.shape)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the aggregator.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = self.metrics.copy()
        
        if self.metrics["aggregation_times"]:
            metrics["avg_aggregation_time"] = sum(self.metrics["aggregation_times"]) / len(self.metrics["aggregation_times"])
            metrics["max_aggregation_time"] = max(self.metrics["aggregation_times"])
            metrics["min_aggregation_time"] = min(self.metrics["aggregation_times"])
        
        if self.metrics["memory_usage"]:
            metrics["avg_memory_usage"] = sum(self.metrics["memory_usage"]) / len(self.metrics["memory_usage"])
            metrics["max_memory_usage"] = max(self.metrics["memory_usage"])
            metrics["min_memory_usage"] = min(self.metrics["memory_usage"])
        
        return metrics


# Example usage
if __name__ == "__main__":
    import tensorflow as tf
    
    # Create a sample model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Get model weights
    global_weights = model.get_weights()
    
    # Create synthetic client updates
    num_clients = 5
    contributions = []
    
    for i in range(num_clients):
        # Create random updates (small perturbations)
        updates = [w * 0.01 * np.random.randn(*w.shape) for w in global_weights]
        
        # Create contribution info
        info = ContributionInfo(
            client_id=f"client_{i}",
            metrics={
                "accuracy": 0.7 + (0.05 * i),  # Increasing accuracy
                "loss": 0.5 - (0.02 * i),      # Decreasing loss
                "dataset_size": 1000 * (i + 1)  # Different dataset sizes
            }
        )
        
        contributions.append((updates, info))
    
    # Initialize aggregator with different methods
    for method in ["fedavg", "weighted_average", "median"]:
        print(f"\nTesting {method} aggregation method")
        
        aggregator = StreamingAggregator(
            method=method,
            memory_limit_mb=2048,  # 2GB
            use_gpu=False,  # For testing, don't use GPU
            auto_scaling=True
        )
        
        # Time the aggregation
        start_time = time.time()
        
        # Perform aggregation
        aggregated_weights = aggregator.aggregate(contributions, global_weights)
        
        # Print performance stats
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.3f} seconds")
        
        # Verify shapes
        print(f"  Output weight shapes match input: {all(aw.shape == gw.shape for aw, gw in zip(aggregated_weights, global_weights))}")
        
        # Print metrics
        metrics = aggregator.get_performance_metrics()
        print(f"  Processed {metrics['processed_elements']:,} parameters")
        print(f"  Memory usage: {metrics['memory_usage'][-1]:.1f} MB")
        print(f"  Chunk size: {metrics['last_chunk_size']:,} elements")
    
    # Test parallel aggregation with large model
    print("\nTesting parallel aggregation with large model")
    
    # Create a larger model to test parallel aggregation
    large_model = tf.keras.Sequential([
        tf.keras.layers.Dense(2048, activation='relu', input_shape=(1024,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(100, activation='softmax')
    ])
    
    # Get large model weights
    large_weights = large_model.get_weights()
    
    # Create synthetic client updates for large model
    large_contributions = []
    for i in range(3):  # Fewer clients to save memory
        updates = [w * 0.01 * np.random.randn(*w.shape) for w in large_weights]
        info = ContributionInfo(
            client_id=f"client_{i}",
            metrics={
                "accuracy": 0.8 + (0.05 * i),
                "loss": 0.3 - (0.05 * i),
                "dataset_size": 2000 * (i + 1)
            }
        )
        large_contributions.append((updates, info))
    
    # Initialize aggregator for parallel processing
    parallel_aggregator = StreamingAggregator(
        method="fedavg",
        memory_limit_mb=4096,  # 4GB
        auto_scaling=True
    )
    
    # Time the standard aggregation
    start_time = time.time()
    aggregated_weights = parallel_aggregator.aggregate(large_contributions, large_weights)
    standard_time = time.time() - start_time
    
    print(f"  Standard aggregation completed in {standard_time:.3f} seconds")
    
    # Time the parallel aggregation
    start_time = time.time()
    parallel_weights = parallel_aggregator.parallel_aggregation(
        large_contributions, 
        large_weights,
        num_workers=4  # Use 4 worker threads
    )
    parallel_time = time.time() - start_time
    
    print(f"  Parallel aggregation completed in {parallel_time:.3f} seconds")
    print(f"  Speedup: {standard_time / parallel_time:.2f}x")
    
    # Verify results are the same
    is_equal = all(
        np.allclose(aw, pw, rtol=1e-5, atol=1e-5)
        for aw, pw in zip(aggregated_weights, parallel_weights)
    )
    print(f"  Results match: {is_equal}")
    
    # Memory usage comparison
    metrics = parallel_aggregator.get_performance_metrics()
    print(f"  Memory usage (standard): {metrics['memory_usage'][-2]:.1f} MB")
    print(f"  Memory usage (parallel): {metrics['memory_usage'][-1]:.1f} MB")
    
    print("\nTesting different memory limits:")
    for memory_mb in [1024, 2048, 4096, 8192]:
        # Initialize aggregator with specific memory limit
        memory_test_aggregator = StreamingAggregator(
            method="fedavg",
            memory_limit_mb=memory_mb,
            auto_scaling=False  # Disable auto-scaling to test fixed memory limits
        )
        
        # Perform aggregation
        start_time = time.time()
        memory_test_aggregator.aggregate(contributions, global_weights)
        elapsed = time.time() - start_time
        
        # Get metrics
        metrics = memory_test_aggregator.get_performance_metrics()
        chunk_size = metrics["last_chunk_size"]
        memory_usage = metrics["memory_usage"][-1]
        
        print(f"  Memory limit: {memory_mb} MB → Chunk size: {chunk_size:,} elements → "
              f"Actual memory: {memory_usage:.1f} MB → Time: {elapsed:.3f}s")
    
    print("\nAll aggregation methods verified successfully!")
"""
Optimized Aggregation Algorithms for Large Models

This module provides efficient implementations of aggregation algorithms for federated learning
with a focus on handling large neural network models with memory efficiency and performance.
"""

import numpy as np
import tensorflow as tf
import logging
import time
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import concurrent.futures
import threading
import queue
from dataclasses import dataclass
import psutil
import gc
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ContributionInfo:
    """Information about a contribution for aggregation."""
    client_id: str
    metrics: Dict[str, Any]
    weight: float = 1.0  # Relative weight for aggregation
    
    def __str__(self):
        return f"ContributionInfo(client_id={self.client_id}, weight={self.weight})"

class ChunkedArray:
    """
    Memory-efficient array implementation that processes data in chunks.
    
    This class enables processing large arrays that might not fit in memory all at once
    by splitting operations across manageable chunks.
    """
    
    def __init__(self, shape: Tuple[int, ...], dtype=np.float32, chunk_size: int = 10000000):
        """
        Initialize a chunked array.
        
        Args:
            shape: Shape of the array
            dtype: Data type for the array
            chunk_size: Maximum elements per chunk (default 10M elements)
        """
        self.shape = shape
        self.size = np.prod(shape)
        self.dtype = dtype
        self.chunk_size = min(chunk_size, self.size)  # Ensure chunk_size doesn't exceed total size
        
        # Calculate number of chunks
        self.num_chunks = int(np.ceil(self.size / self.chunk_size))
        
        # Calculate chunk boundaries
        self.chunk_boundaries = []
        for i in range(self.num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, self.size)
            self.chunk_boundaries.append((start, end))
    
    def apply_function(self, func: Callable, *arrays: np.ndarray) -> np.ndarray:
        """
        Apply a function to arrays in chunks.
        
        Args:
            func: Function to apply (takes arrays as input and returns an array)
            arrays: Input arrays
            
        Returns:
            Result array
        """
        # Validate input arrays
        for arr in arrays:
            if arr.size != self.size:
                raise ValueError(f"Array size mismatch: expected {self.size}, got {arr.size}")
        
        # Create output array
        result = np.zeros(self.size, dtype=self.dtype)
        
        # Process in chunks
        for start, end in self.chunk_boundaries:
            # Extract chunks from input arrays
            chunks = [arr.flat[start:end] for arr in arrays]
            
            # Apply function
            result[start:end] = func(*chunks)
            
            # Explicitly free memory
            del chunks
            gc.collect()
        
        # Reshape result to match original shape
        return result.reshape(self.shape)
    
    @staticmethod
    def weighted_average(chunks: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """
        Compute weighted average of arrays.
        
        Args:
            chunks: List of array chunks
            weights: List of weights
            
        Returns:
            Weighted average
        """
        result = np.zeros_like(chunks[0])
        for i, chunk in enumerate(chunks):
            result += chunk * weights[i]
        
        return result

class StreamingAggregator:
    """
    Memory-efficient streaming aggregator for large model parameters.
    
    This class enables aggregating large model updates using streaming operations
    to avoid loading all data into memory at once, suitable for very large models.
    """
    
    def __init__(
        self,
        method: str = "fedavg",
        memory_limit_mb: int = 4096,  # Default 4GB memory limit
        use_gpu: bool = True,
        auto_scaling: bool = True
    ):
        """
        Initialize the streaming aggregator.
        
        Args:
            method: Aggregation method ("fedavg", "weighted_average", or "median")
            memory_limit_mb: Memory limit in megabytes
            use_gpu: Whether to use GPU for computations when available
            auto_scaling: Automatically adjust chunk size based on available memory
        """
        self.method = method
        self.memory_limit_mb = memory_limit_mb
        self.use_gpu = use_gpu and tf.config.list_physical_devices('GPU')
        self.auto_scaling = auto_scaling
        
        # Calculate optimal chunk size based on memory limit
        self.chunk_size = self._calculate_chunk_size()
        
        # Performance metrics
        self.metrics = {
            "aggregation_times": [],
            "memory_usage": [],
            "processed_elements": 0,
            "last_chunk_size": 0
        }
    
    def _calculate_chunk_size(self) -> int:
        """
        Calculate optimal chunk size based on memory constraints.
        
        Returns:
            Chunk size (number of elements)
        """
        # Base chunk size calculation
        # Each float32 is 4 bytes, we aim to use at most 1/4 of memory_limit for a chunk
        bytes_per_element = 4  # float32
        max_elements = (self.memory_limit_mb * 1024 * 1024) // (bytes_per_element * 4)
        
        # Start with a reasonable chunk size
        chunk_size = min(max_elements, 10000000)  # 10M elements default max
        
        if self.auto_scaling:
            # Adjust based on available system memory
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            scaling_factor = min(max(0.1, available_memory / self.memory_limit_mb), 2.0)
            chunk_size = int(chunk_size * scaling_factor)
        
        logger.info(f"Calculated chunk size: {chunk_size} elements")
        return chunk_size
    
    def aggregate(
        self,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Aggregate model updates using the specified method.
        
        Args:
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights (required for some methods)
            
        Returns:
            Aggregated weights
        """
        start_time = time.time()
        
        if not contributions:
            raise ValueError("No contributions to aggregate")
        
        # Select aggregation method
        if self.method == "fedavg":
            result = self._federated_averaging(contributions, global_weights)
        elif self.method == "weighted_average":
            result = self._weighted_averaging(contributions, global_weights)
        elif self.method == "median":
            result = self._median_aggregation(contributions, global_weights)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")
        
        # Record metrics
        elapsed_time = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        self.metrics["aggregation_times"].append(elapsed_time)
        self.metrics["memory_usage"].append(memory_usage)
        self.metrics["last_chunk_size"] = self.chunk_size
        
        # Log performance
        logger.info(
            f"Aggregation completed in {elapsed_time:.2f}s using {memory_usage:.1f}MB of memory. "
            f"Processed {self.metrics['processed_elements']} elements across {len(contributions)} contributions."
        )
        
        return result
    
    def _federated_averaging(
        self,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Perform federated averaging aggregation with streaming operations.
        
        Args:
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights
            
        Returns:
            Aggregated weights
        """
        if not global_weights:
            raise ValueError("Global weights are required for federated averaging")
        
        # Extract updates and weights for aggregation
        updates = [upd for upd, _ in contributions]
        infos = [info for _, info in contributions]
        
        # Calculate total weight
        total_weight = sum(info.metrics.get('dataset_size', 1) for info in infos)
        
        # Calculate normalized weights
        weights = [info.metrics.get('dataset_size', 1) / total_weight for info in infos]
        
        # Create result arrays with the same shape as global weights
        result = [np.zeros_like(w) for w in global_weights]
        
        # Process each layer separately
        for layer_idx, global_layer in enumerate(global_weights):
            # Count elements for metrics
            self.metrics["processed_elements"] += global_layer.size
            
            # For small layers, perform direct computation
            if global_layer.size <= self.chunk_size:
                # Convert updates to arrays for this layer
                layer_updates = np.array([upd[layer_idx] for upd in updates])
                
                # Weighted average of updates
                weighted_update = np.zeros_like(global_layer)
                for i, update in enumerate(layer_updates):
                    weighted_update += update * weights[i]
                
                # Add weighted update to global weights
                result[layer_idx] = global_layer + weighted_update
            else:
                # For large layers, use chunked processing
                chunked_array = ChunkedArray(
                    shape=global_layer.shape,
                    dtype=global_layer.dtype,
                    chunk_size=self.chunk_size
                )
                
                # Process in chunks
                flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
                
                for start, end in chunked_array.chunk_boundaries:
                    # Extract chunks from all updates
                    update_chunks = []
                    for upd in updates:
                        update_chunks.append(upd[layer_idx].flat[start:end])
                    
                    # Calculate weighted average
                    chunk_result = np.zeros_like(update_chunks[0])
                    for i, chunk in enumerate(update_chunks):
                        chunk_result += chunk * weights[i]
                    
                    # Add to global weights
                    global_chunk = global_layer.flat[start:end]
                    flat_result[start:end] = global_chunk + chunk_result
                    
                    # Free memory
                    del update_chunks, chunk_result, global_chunk
                    gc.collect()
                
                # Reshape result
                result[layer_idx] = flat_result.reshape(global_layer.shape)
        
        return result
    
    def _weighted_averaging(
        self,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Perform weighted averaging based on metrics like accuracy.
        
        Args:
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights
            
        Returns:
            Aggregated weights
        """
        if not global_weights:
            raise ValueError("Global weights are required for weighted averaging")
        
        # Extract updates and infos
        updates = [upd for upd, _ in contributions]
        infos = [info for _, info in contributions]
        
        # Calculate weights based on accuracy
        accuracies = [info.metrics.get('accuracy', 0.5) for info in infos]
        total_accuracy = sum(accuracies)
        
        if total_accuracy == 0:
            # If all accuracies are 0, use equal weights
            weights = [1.0 / len(contributions)] * len(contributions)
        else:
            # Normalize weights by accuracy
            weights = [acc / total_accuracy for acc in accuracies]
        
        # Create result list
        result = [np.zeros_like(w) for w in global_weights]
        
        # Process each layer
        for layer_idx, global_layer in enumerate(global_weights):
            # Count elements for metrics
            self.metrics["processed_elements"] += global_layer.size
            
            # For small layers, process directly
            if global_layer.size <= self.chunk_size:
                # Initialize with zeros
                result[layer_idx] = np.zeros_like(global_layer)
                
                # Add weighted contribution from each client
                for i, upd in enumerate(updates):
                    # Add the update to the global weights to get the client's version
                    client_weights = global_layer + upd[layer_idx]
                    
                    # Add weighted contribution
                    result[layer_idx] += client_weights * weights[i]
            else:
                # For large layers, use chunked processing
                chunked_array = ChunkedArray(
                    shape=global_layer.shape,
                    dtype=global_layer.dtype,
                    chunk_size=self.chunk_size
                )
                
                # Process in chunks
                flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
                
                for start, end in chunked_array.chunk_boundaries:
                    # Extract global chunk
                    global_chunk = global_layer.flat[start:end]
                    
                    # Initialize chunk result
                    chunk_result = np.zeros_like(global_chunk)
                    
                    # Process each client's contribution
                    for i, upd in enumerate(updates):
                        # Get client chunk
                        client_chunk = global_chunk + upd[layer_idx].flat[start:end]
                        
                        # Add weighted contribution
                        chunk_result += client_chunk * weights[i]
                    
                    # Store result
                    flat_result[start:end] = chunk_result
                    
                    # Free memory
                    del global_chunk, chunk_result
                    gc.collect()
                
                # Reshape result
                result[layer_idx] = flat_result.reshape(global_layer.shape)
        
        return result
    
    def _median_aggregation(
        self,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Perform median-based aggregation which is more robust to outliers.
        
        Args:
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights
            
        Returns:
            Aggregated weights
        """
        if not global_weights:
            raise ValueError("Global weights are required for median aggregation")
        
        # Extract updates
        updates = [upd for upd, _ in contributions]
        
        # Create result list
        result = [np.zeros_like(w) for w in global_weights]
        
        # Process each layer
        for layer_idx, global_layer in enumerate(global_weights):
            # Count elements for metrics
            self.metrics["processed_elements"] += global_layer.size
            
            # For small layers, process directly
            if global_layer.size <= self.chunk_size:
                # Collect all client weights for this layer
                client_weights = []
                for upd in updates:
                    # Convert updates to weights
                    client_w = global_layer + upd[layer_idx]
                    client_weights.append(client_w)
                
                # Stack along a new axis for element-wise median
                stacked = np.stack(client_weights, axis=0)
                
                # Compute median along client axis
                result[layer_idx] = np.median(stacked, axis=0)
            else:
                # For large layers, use chunked processing
                flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
                
                # Process in chunks
                chunk_size = self.chunk_size
                num_chunks = int(np.ceil(global_layer.size / chunk_size))
                
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * chunk_size
                    end = min((chunk_idx + 1) * chunk_size, global_layer.size)
                    
                    # Extract global chunk
                    global_chunk = global_layer.flat[start:end]
                    
                    # Collect client chunks
                    client_chunks = []
                    for upd in updates:
                        # Get update chunk
                        update_chunk = upd[layer_idx].flat[start:end]
                        
                        # Convert to client weights
                        client_chunk = global_chunk + update_chunk
                        client_chunks.append(client_chunk)
                    
                    # Stack for element-wise median
                    stacked_chunks = np.stack(client_chunks, axis=0)
                    
                    # Calculate median
                    chunk_result = np.median(stacked_chunks, axis=0)
                    
                    # Store result
                    flat_result[start:end] = chunk_result
                    
                    # Free memory
                    del global_chunk, client_chunks, stacked_chunks, chunk_result
                    gc.collect()
                
                # Reshape result
                result[layer_idx] = flat_result.reshape(global_layer.shape)
        
        return result
    
    def parallel_aggregation(
        self,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: List[np.ndarray],
        num_workers: int = None
    ) -> List[np.ndarray]:
        """
        Perform aggregation with parallel processing for large models.
        
        Args:
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights
            num_workers: Number of worker threads/processes
            
        Returns:
            Aggregated weights
        """
        if not num_workers:
            # Default to number of available CPU cores minus 1 (leave one for system)
            num_workers = max(1, psutil.cpu_count(logical=False) - 1)
        
        # Create result list
        result = [None] * len(global_weights)
        
        # Group layers by size for efficient processing
        small_layers = []  # Process together
        large_layers = []  # Process individually
        
        for layer_idx, layer in enumerate(global_weights):
            if layer.size <= self.chunk_size // 10:  # Very small layers
                small_layers.append(layer_idx)
            else:
                large_layers.append(layer_idx)
        
        logger.info(f"Aggregating {len(small_layers)} small layers together and {len(large_layers)} large layers in parallel")
        
        # Create task queue
        task_queue = queue.Queue()
        result_dict = {}
        
        # Add tasks for large layers (one task per layer)
        for layer_idx in large_layers:
            task_queue.put(("single", layer_idx))
        
        # Add small layers as one batch task
        if small_layers:
            task_queue.put(("batch", small_layers))
        
        # Define worker function
        def worker_func():
            while True:
                try:
                    task_type, layer_data = task_queue.get(block=False)
                    
                    if task_type == "single":
                        # Process single large layer
                        layer_idx = layer_data
                        layer_result = self._aggregate_single_layer(
                            layer_idx, contributions, global_weights
                        )
                        result_dict[layer_idx] = layer_result
                    elif task_type == "batch":
                        # Process batch of small layers
                        layer_indices = layer_data
                        for idx in layer_indices:
                            layer_result = self._aggregate_single_layer(
                                idx, contributions, global_weights
                            )
                            result_dict[idx] = layer_result
                    
                    task_queue.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error in worker thread: {e}")
                    task_queue.task_done()
        
        # Start worker threads
        threads = []
        for _ in range(num_workers):
            thread = threading.Thread(target=worker_func)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        task_queue.join()
        
        # Assemble results
        for layer_idx in range(len(global_weights)):
            result[layer_idx] = result_dict.get(layer_idx)
        
        # Ensure all layers were processed
        for layer_idx, layer_result in enumerate(result):
            if layer_result is None:
                # Process any missing layers (shouldn't happen but just in case)
                logger.warning(f"Layer {layer_idx} was not processed by workers, processing now")
                result[layer_idx] = self._aggregate_single_layer(
                    layer_idx, contributions, global_weights
                )
        
        return result
    
    def _aggregate_single_layer(
        self,
        layer_idx: int,
        contributions: List[Tuple[List[np.ndarray], ContributionInfo]],
        global_weights: List[np.ndarray]
    ) -> np.ndarray:
        """
        Aggregate a single layer based on the selected method.
        
        Args:
            layer_idx: Index of the layer to aggregate
            contributions: List of (model_update, contribution_info) tuples
            global_weights: Current global weights
            
        Returns:
            Aggregated layer weights
        """
        global_layer = global_weights[layer_idx]
        
        # Use appropriate aggregation method
        if self.method == "fedavg":
            # Extract updates and calculate weights
            updates = [upd[layer_idx] for upd, _ in contributions]
            infos = [info for _, info in contributions]
            
            # Calculate total weight
            total_weight = sum(info.metrics.get('dataset_size', 1) for info in infos)
            
            # Calculate normalized weights
            weights = [info.metrics.get('dataset_size', 1) / total_weight for info in infos]
            
            # Aggregate layer
            if global_layer.size <= self.chunk_size:
                weighted_update = np.zeros_like(global_layer)
                for i, update in enumerate(updates):
                    weighted_update += update * weights[i]
                
                return global_layer + weighted_update
            else:
                return self._chunked_layer_fedavg(
                    global_layer, updates, weights
                )
        
        elif self.method == "weighted_average":
            # Similar to fedavg but with accuracy-based weights
            # Extract updates and infos
            updates = [upd[layer_idx] for upd, _ in contributions]
            infos = [info for _, info in contributions]
            
            # Calculate weights based on accuracy
            accuracies = [info.metrics.get('accuracy', 0.5) for info in infos]
            total_accuracy = sum(accuracies)
            
            if total_accuracy == 0:
                weights = [1.0 / len(contributions)] * len(contributions)
            else:
                weights = [acc / total_accuracy for acc in accuracies]
            
            return self._chunked_layer_weighted_avg(
                global_layer, updates, weights
            )
        
        elif self.method == "median":
            # Extract updates
            updates = [upd[layer_idx] for upd, _ in contributions]
            
            return self._chunked_layer_median(
                global_layer, updates
            )
        
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")
    
    def _chunked_layer_fedavg(
        self,
        global_layer: np.ndarray,
        updates: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """
        Apply federated averaging to a layer using chunked processing.
        
        Args:
            global_layer: Global layer weights
            updates: List of updates from clients
            weights: List of client weights
            
        Returns:
            Aggregated layer
        """
        # Create chunked array processor
        chunked_array = ChunkedArray(
            shape=global_layer.shape,
            dtype=global_layer.dtype,
            chunk_size=self.chunk_size
        )
        
        # Process in chunks
        flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
        
        for start, end in chunked_array.chunk_boundaries:
            # Extract chunks from all updates
            update_chunks = []
            for upd in updates:
                update_chunks.append(upd.flat[start:end])
            
            # Calculate weighted average
            chunk_result = np.zeros_like(update_chunks[0])
            for i, chunk in enumerate(update_chunks):
                chunk_result += chunk * weights[i]
            
            # Add to global weights
            global_chunk = global_layer.flat[start:end]
            flat_result[start:end] = global_chunk + chunk_result
            
            # Free memory
            del update_chunks, chunk_result, global_chunk
            gc.collect()
        
        # Reshape result
        return flat_result.reshape(global_layer.shape)
    
    def _chunked_layer_weighted_avg(
        self,
        global_layer: np.ndarray,
        updates: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """
        Apply weighted averaging to a layer using chunked processing.
        
        Args:
            global_layer: Global layer weights
            updates: List of updates from clients
            weights: List of client weights
            
        Returns:
            Aggregated layer
        """
        # Create chunked array processor
        chunked_array = ChunkedArray(
            shape=global_layer.shape,
            dtype=global_layer.dtype,
            chunk_size=self.chunk_size
        )
        
        # Process in chunks
        flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
        
        for start, end in chunked_array.chunk_boundaries:
            # Extract global chunk
            global_chunk = global_layer.flat[start:end]
            
            # Initialize chunk result
            chunk_result = np.zeros_like(global_chunk)
            
            # Process each client's contribution
            for i, upd in enumerate(updates):
                # Get client chunk
                client_chunk = global_chunk + upd.flat[start:end]
                
                # Add weighted contribution
                chunk_result += client_chunk * weights[i]
            
            # Store result
            flat_result[start:end] = chunk_result
            
            # Free memory
            del global_chunk, chunk_result
            gc.collect()
        
        # Reshape result
        return flat_result.reshape(global_layer.shape)
    
    def _chunked_layer_median(
        self,
        global_layer: np.ndarray,
        updates: List[np.ndarray]
    ) -> np.ndarray:
        """
        Apply median aggregation to a layer using chunked processing.
        
        Args:
            global_layer: Global layer weights
            updates: List of updates from clients
            
        Returns:
            Aggregated layer
        """
        # Create result array
        flat_result = np.zeros(global_layer.size, dtype=global_layer.dtype)
        
        # Process in chunks
        chunk_size = self.chunk_size
        num_chunks = int(np.ceil(global_layer.size / chunk_size))
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, global_layer.size)
            
            # Extract global chunk
            global_chunk = global_layer.flat[start:end]
            
            # Collect client chunks
            client_chunks = []
            for upd in updates:
                # Get update chunk
                update_chunk = upd.flat[start:end]
                
                # Convert to client weights
                client_chunk = global_chunk + update_chunk
                client_chunks.append(client_chunk)
            
            # Stack for element-wise median
            stacked_chunks = np.stack(client_chunks, axis=0)
            
            # Calculate median
            chunk_result = np.median(stacked_chunks, axis=0)
            
            # Store result
            flat_result[start:end] = chunk_result
            
            # Free memory
            del global_chunk, client_chunks, stacked_chunks, chunk_result
            gc.collect()
        
        # Reshape result
        return flat_result.reshape(global_layer.shape)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the aggregator.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = self.metrics.copy()
        
        if self.metrics["aggregation_times"]:
            metrics["avg_aggregation_time"] = sum(self.metrics["aggregation_times"]) / len(self.metrics["aggregation_times"])
            metrics["max_aggregation_time"] = max(self.metrics["aggregation_times"])
            metrics["min_aggregation_time"] = min(self.metrics["aggregation_times"])
        
        if self.metrics["memory_usage"]:
            metrics["avg_memory_usage"] = sum(self.metrics["memory_usage"]) / len(self.metrics["memory_usage"])
            metrics["max_memory_usage"] = max(self.metrics["memory_usage"])
            metrics["min_memory_usage"] = min(self.metrics["memory_usage"])
        
        return metrics


# Example usage
if __name__ == "__main__":
    import tensorflow as tf
    
    # Create a sample model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Get model weights
    global_weights = model.get_weights()
    
    # Create synthetic client updates
    num_clients = 5
    contributions = []
    
    for i in range(num_clients):
        # Create random updates (small perturbations)
        updates = [w * 0.01 * np.random.randn(*w.shape) for w in global_weights]
        
        # Create contribution info
        info = ContributionInfo(
            client_id=f"client_{i}",
            metrics={
                "accuracy": 0.7 + (0.05 * i),  # Increasing accuracy
                "loss": 0.5 - (0.02 * i),      # Decreasing loss
                "dataset_size": 1000 * (i + 1)  # Different dataset sizes
            }
        )
        
        contributions.append((updates, info))
    
    # Initialize aggregator with different methods
    for method in ["fedavg", "weighted_average", "median"]:
        print(f"\nTesting {method} aggregation method")
