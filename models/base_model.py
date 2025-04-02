import abc
import json
import numpy as np
import uuid
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime


class BaseModel(abc.ABC):
    """
    Abstract base class for all federated learning models.
    
    This class defines the common interface that all federated learning models
    must implement to be compatible with the platform.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new base model.
        
        Args:
            model_id: Unique identifier for the model. If None, a new UUID is generated.
            version: Model version string following semantic versioning.
            metadata: Additional metadata associated with the model.
        """
        self.model_id = model_id if model_id else str(uuid.uuid4())
        self.version = version
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.metadata = metadata or {}
        
        # Track training history
        self.training_rounds = 0
        self.metrics_history = []
        
    @abc.abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """
        Get the current weights of the model.
        
        Returns:
            List of numpy arrays containing the model weights.
        """
        pass
    
    @abc.abstractmethod
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Set the weights of the model.
        
        Args:
            weights: List of numpy arrays containing the model weights.
        """
        pass
    
    @abc.abstractmethod
    def train(self, data: Any, **kwargs) -> Dict[str, float]:
        """
        Train the model on local data.
        
        Args:
            data: The data to train on. Format depends on the specific model implementation.
            **kwargs: Additional training arguments.
            
        Returns:
            Dictionary of metrics (e.g., loss, accuracy) from training.
        """
        pass
    
    @abc.abstractmethod
    def evaluate(self, data: Any, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on data.
        
        Args:
            data: The data to evaluate on. Format depends on the specific model implementation.
            **kwargs: Additional evaluation arguments.
            
        Returns:
            Dictionary of metrics (e.g., loss, accuracy) from evaluation.
        """
        pass
    
    @abc.abstractmethod
    def predict(self, data: Any, **kwargs) -> Any:
        """
        Generate predictions using the model.
        
        Args:
            data: Input data for prediction.
            **kwargs: Additional prediction arguments.
            
        Returns:
            Model predictions.
        """
        pass
    
    def serialize_weights(self) -> str:
        """
        Serialize model weights to a string representation.
        
        Returns:
            JSON string representation of the weights.
        """
        weights = self.get_weights()
        serialized = []
        
        for arr in weights:
            serialized.append({
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "data": arr.flatten().tolist()
            })
            
        return json.dumps(serialized)
    
    def deserialize_weights(self, serialized_weights: str) -> None:
        """
        Deserialize weights from a string representation and set model weights.
        
        Args:
            serialized_weights: JSON string representation of weights.
        """
        serialized = json.loads(serialized_weights)
        weights = []
        
        for arr_dict in serialized:
            shape = tuple(arr_dict["shape"])
            dtype = np.dtype(arr_dict["dtype"])
            data = np.array(arr_dict["data"], dtype=dtype).reshape(shape)
            weights.append(data)
            
        self.set_weights(weights)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the model.
        
        Returns:
            Dictionary containing model configuration.
        """
        return {
            "model_id": self.model_id,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "training_rounds": self.training_rounds,
            "metadata": self.metadata
        }
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log metrics from training or evaluation.
        
        Args:
            metrics: Dictionary of metric values.
        """
        timestamp = datetime.utcnow().isoformat()
        metrics_entry = {
            "timestamp": timestamp,
            "round": self.training_rounds,
            **metrics
        }
        self.metrics_history.append(metrics_entry)
        self.updated_at = timestamp
    
    def increment_training_round(self) -> None:
        """Increment the training round counter."""
        self.training_rounds += 1
        self.updated_at = datetime.utcnow().isoformat()
        
    def compute_weight_diff(self, original_weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute the difference between current weights and original weights.
        
        This is useful for federated learning to submit only the weight updates.
        
        Args:
            original_weights: List of numpy arrays containing the original weights.
            
        Returns:
            List of numpy arrays containing the weight differences.
        """
        current_weights = self.get_weights()
        
        if len(current_weights) != len(original_weights):
            raise ValueError("Weight arrays must have the same length")
            
        weight_diff = []
        for curr, orig in zip(current_weights, original_weights):
            if curr.shape != orig.shape:
                raise ValueError(f"Weight shapes don't match: {curr.shape} vs {orig.shape}")
            weight_diff.append(curr - orig)
            
        return weight_diff 