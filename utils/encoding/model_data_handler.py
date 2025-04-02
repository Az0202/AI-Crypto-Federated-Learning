"""
Optimized Data Encoding Module

This module provides efficient in-memory encoding and decoding for model weights
and updates, avoiding file I/O operations for improved performance.
"""

import io
import base64
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class ModelDataHandler:
    """Efficient handler for encoding and decoding model data."""
    
    def __init__(self, compression_level: int = 1):
        """
        Initialize the data handler.
        
        Args:
            compression_level: Level of compression (0-9, where 0 is no compression)
                              Higher values increase CPU usage but reduce data size
        """
        self.compression_level = compression_level
        
        # Performance metrics
        self.encode_times = []
        self.decode_times = []
        self.encoded_sizes = []
        
        # Max entries to track for metrics
        self.max_metric_entries = 100
    
    def encode_weights(self, weights: List[np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Encode model weights to base64 string using in-memory approach.
        
        Args:
            weights: List of numpy arrays containing model weights
            metadata: Optional dictionary of metadata to include with weights
            
        Returns:
            Base64 encoded string
        """
        start_time = time.time()
        
        try:
            # Use BytesIO instead of temporary file
            buffer = io.BytesIO()
            
            # Save with compression if enabled
            if self.compression_level > 0:
                np.savez_compressed(buffer, *weights)
            else:
                np.savez(buffer, *weights)
            
            # Add metadata if provided
            if metadata:
                # We need to reset the buffer position to add metadata
                buffer.seek(0, io.SEEK_END)
                metadata_str = str(metadata).encode('utf-8')
                buffer.write(b'METADATA:')
                buffer.write(metadata_str)
            
            # Reset buffer position for reading
            buffer.seek(0)
            
            # Encode to base64
            base64_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Track metrics
            encode_time = time.time() - start_time
            self._update_metrics(encode_time, len(base64_data))
            
            return base64_data
            
        except Exception as e:
            logger.error(f"Failed to encode model weights: {e}")
            raise ValueError(f"Failed to encode model weights: {e}")
    
    def decode_weights(self, base64_data: str) -> tuple[List[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Decode base64 model weights back to numpy arrays using in-memory approach.
        
        Args:
            base64_data: Base64 encoded model weights
            
        Returns:
            Tuple of (weights, metadata)
        """
        start_time = time.time()
        
        try:
            # Decode base64 to bytes
            binary_data = base64.b64decode(base64_data)
            
            # Check for metadata
            metadata_marker = b'METADATA:'
            metadata_pos = binary_data.find(metadata_marker)
            
            metadata = None
            data_for_numpy = binary_data
            
            if metadata_pos > 0:
                # Extract metadata
                metadata_bytes = binary_data[metadata_pos + len(metadata_marker):]
                try:
                    metadata_str = metadata_bytes.decode('utf-8')
                    # Convert string representation of dict back to dict
                    # Note: This is a simple approach and not secure for untrusted data
                    metadata = eval(metadata_str)
                except Exception as e:
                    logger.warning(f"Failed to decode metadata: {e}")
                
                # Use only the numpy data portion
                data_for_numpy = binary_data[:metadata_pos]
            
            # Load from bytes
            buffer = io.BytesIO(data_for_numpy)
            with np.load(buffer) as data:
                # Convert file-like object to list of arrays
                arrays = [data[key] for key in data.files]
            
            # Track metrics
            decode_time = time.time() - start_time
            self._track_decode_time(decode_time)
            
            return arrays, metadata
            
        except Exception as e:
            logger.error(f"Failed to decode model weights: {e}")
            raise ValueError(f"Failed to decode model weights: {e}")
    
    def encode_model_update(self, model_update: List[np.ndarray], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Encode model update to base64 string (alias for encode_weights for clarity).
        
        Args:
            model_update: List of numpy arrays representing model update
            metadata: Optional dictionary of metadata to include
            
        Returns:
            Base64 encoded model update
        """
        return self.encode_weights(model_update, metadata)
    
    def decode_model_update(self, base64_data: str) -> tuple[List[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Decode base64 model update (alias for decode_weights for clarity).
        
        Args:
            base64_data: Base64 encoded model update
            
        Returns:
            Tuple of (model_update, metadata)
        """
        return self.decode_weights(base64_data)
    
    def _update_metrics(self, encode_time: float, encoded_size: int):
        """Update performance metrics with new data."""
        self.encode_times.append(encode_time)
        self.encoded_sizes.append(encoded_size)
        
        # Keep lists at max size
        if len(self.encode_times) > self.max_metric_entries:
            self.encode_times.pop(0)
        if len(self.encoded_sizes) > self.max_metric_entries:
            self.encoded_sizes.pop(0)
    
    def _track_decode_time(self, decode_time: float):
        """Track decode time performance."""
        self.decode_times.append(decode_time)
        if len(self.decode_times) > self.max_metric_entries:
            self.decode_times.pop(0)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the encoding/decoding process.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.encode_times:
            return {"status": "No metrics available"}
        
        return {
            "encode": {
                "avg_time_ms": sum(self.encode_times) / len(self.encode_times) * 1000,
                "min_time_ms": min(self.encode_times) * 1000,
                "max_time_ms": max(self.encode_times) * 1000
            },
            "decode": {
                "avg_time_ms": sum(self.decode_times) / len(self.decode_times) * 1000 if self.decode_times else 0,
                "min_time_ms": min(self.decode_times) * 1000 if self.decode_times else 0,
                "max_time_ms": max(self.decode_times) * 1000 if self.decode_times else 0
            },
            "size": {
                "avg_bytes": sum(self.encoded_sizes) / len(self.encoded_sizes),
                "min_bytes": min(self.encoded_sizes),
                "max_bytes": max(self.encoded_sizes)
            },
            "compression_level": self.compression_level
        }
    
    def optimize_compression(self, sample_weights: List[np.ndarray]) -> int:
        """
        Find optimal compression level for specific model weights.
        
        Args:
            sample_weights: Sample weights to test compression levels on
            
        Returns:
            Optimal compression level based on time/size tradeoff
        """
        results = []
        
        for level in range(10):  # Test all compression levels 0-9
            self.compression_level = level
            
            # Test encoding speed and size
            start_time = time.time()
            encoded = self.encode_weights(sample_weights)
            encode_time = time.time() - start_time
            
            # Test decoding speed
            start_time = time.time()
            self.decode_weights(encoded)
            decode_time = time.time() - start_time
            
            # Calculate score (lower is better)
            # This formula prioritizes decode speed (3x) over encode speed (1x) and size (2x)
            # Adjust weights based on your specific needs
            score = (decode_time * 3) + encode_time + (len(encoded) / 1000000 * 2)
            
            results.append({
                "level": level,
                "encode_time": encode_time,
                "decode_time": decode_time,
                "size": len(encoded),
                "score": score
            })
        
        # Find level with lowest score
        optimal_level = min(results, key=lambda x: x["score"])["level"]
        self.compression_level = optimal_level
        
        return optimal_level


# Example usage
if __name__ == "__main__":
    # Create sample weights
    sample_weights = [
        np.random.rand(100, 100).astype(np.float32),
        np.random.rand(100).astype(np.float32),
        np.random.rand(100, 50).astype(np.float32),
        np.random.rand(50).astype(np.float32)
    ]
    
    # Create data handler
    handler = ModelDataHandler(compression_level=1)
    
    # Find optimal compression level
    optimal_level = handler.optimize_compression(sample_weights)
    print(f"Optimal compression level: {optimal_level}")
    
    # Encode and decode with optimal level
    encoded = handler.encode_weights(sample_weights, {"version": "1.0", "author": "test"})
    decoded, metadata = handler.decode_weights(encoded)
    
    # Verify results
    for i, (original, decoded_array) in enumerate(zip(sample_weights, decoded)):
        assert np.allclose(original, decoded_array), f"Array {i} doesn't match"
    
    assert metadata["version"] == "1.0", "Metadata mismatch"
    
    # Print performance metrics
    metrics = handler.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Test with larger arrays to simulate real models
    large_weights = [
        np.random.rand(1000, 1000).astype(np.float32),
        np.random.rand(1000).astype(np.float32),
        np.random.rand(1000, 500).astype(np.float32),
        np.random.rand(500).astype(np.float32)
    ]
    
    encoded_large = handler.encode_weights(large_weights)
    decoded_large, _ = handler.decode_weights(encoded_large)
    
    print(f"Updated performance metrics: {handler.get_performance_metrics()}")
