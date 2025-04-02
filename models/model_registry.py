import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Any, Type, Union, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import hashlib

from models.base_model import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing and versioning federated learning models.
    
    This class provides functionality for:
    - Registering new models
    - Tracking model versions
    - Storing and loading model weights
    - Managing model metadata
    """
    
    def __init__(
        self,
        storage_path: str = "./model_storage",
        metadata_file: str = "registry_metadata.json"
    ):
        """
        Initialize the model registry.
        
        Args:
            storage_path: Directory path for storing models and metadata
            metadata_file: Filename for the registry metadata
        """
        self.storage_path = Path(storage_path)
        self.metadata_file = metadata_file
        self.metadata_path = self.storage_path / self.metadata_file
        self.models: Dict[str, Dict[str, Any]] = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Load existing metadata if available
        self._load_metadata()
        
    def _load_metadata(self) -> None:
        """Load metadata from disk if it exists."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.models = json.load(f)
                logger.info(f"Loaded metadata for {len(self.models)} models from registry")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load metadata: {e}")
                self.models = {}
        else:
            logger.info("No existing metadata found, starting with empty registry")
            self.models = {}
            
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(self.models, f, indent=2)
            logger.info(f"Saved metadata for {len(self.models)} models to registry")
        except IOError as e:
            logger.error(f"Failed to save metadata: {e}")
            
    def register_model(
        self,
        model: BaseModel,
        description: str = "",
        tags: List[str] = None,
        extra_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Register a new model or a new version of an existing model.
        
        Args:
            model: The model to register
            description: Human-readable description of the model
            tags: List of tags for categorizing the model
            extra_metadata: Additional metadata to store with the model
            
        Returns:
            The model ID
        """
        model_id = model.model_id
        model_config = model.get_config()
        
        # Calculate hash of model weights for versioning
        weights = model.get_weights()
        weights_hash = self._hash_weights(weights)
        
        # Check if this model ID already exists
        if model_id in self.models:
            # Check if this exact weight configuration already exists
            for version_info in self.models[model_id]["versions"]:
                if version_info["weights_hash"] == weights_hash:
                    logger.warning(f"Model with identical weights already exists with version {version_info['version']}")
                    return model_id
                    
            # Increment version number for existing model
            current_version = model.version
            model.version = self._increment_version(current_version)
            model_config["version"] = model.version
            logger.info(f"Registering new version {model.version} for existing model {model_id}")
        else:
            # Create new model entry
            self.models[model_id] = {
                "model_id": model_id,
                "created_at": datetime.utcnow().isoformat(),
                "versions": []
            }
            logger.info(f"Registering new model with ID {model_id}")
        
        # Prepare version metadata
        version_metadata = {
            "version": model.version,
            "created_at": datetime.utcnow().isoformat(),
            "description": description,
            "tags": tags or [],
            "weights_hash": weights_hash,
            "extra_metadata": extra_metadata or {},
            **model_config
        }
        
        # Add new version
        self.models[model_id]["versions"].append(version_metadata)
        self.models[model_id]["updated_at"] = datetime.utcnow().isoformat()
        self.models[model_id]["latest_version"] = model.version
        
        # Save model weights
        self._save_model_weights(model, model_id, model.version, weights)
        
        # Update metadata file
        self._save_metadata()
        
        return model_id
        
    def _hash_weights(self, weights: List[np.ndarray]) -> str:
        """
        Create a hash of model weights for versioning.
        
        Args:
            weights: List of numpy arrays containing model weights
            
        Returns:
            Hash string of weights
        """
        hasher = hashlib.sha256()
        for weight_array in weights:
            hasher.update(weight_array.tobytes())
        return hasher.hexdigest()
        
    def _increment_version(self, version: str) -> str:
        """
        Increment the version string following semantic versioning.
        
        Args:
            version: Current version string (e.g., "1.0.0")
            
        Returns:
            Incremented version string
        """
        parts = version.split(".")
        if len(parts) != 3:
            # Invalid version format, start from "1.0.0"
            return "1.0.0"
            
        try:
            major, minor, patch = map(int, parts)
            patch += 1
            return f"{major}.{minor}.{patch}"
        except ValueError:
            # Invalid version format, start from "1.0.0"
            return "1.0.0"
    
    def _get_weights_path(self, model_id: str, version: str) -> Path:
        """Get the file path for storing model weights."""
        return self.storage_path / f"{model_id}_{version}_weights.pkl"
    
    def _save_model_weights(
        self, 
        model: BaseModel, 
        model_id: str, 
        version: str,
        weights: Optional[List[np.ndarray]] = None
    ) -> None:
        """
        Save model weights to disk.
        
        Args:
            model: The model whose weights to save
            model_id: ID of the model
            version: Version string
            weights: Optional pre-extracted weights
        """
        weights_to_save = weights if weights is not None else model.get_weights()
        weights_path = self._get_weights_path(model_id, version)
        
        try:
            with open(weights_path, "wb") as f:
                pickle.dump(weights_to_save, f)
            logger.info(f"Saved weights for model {model_id} version {version}")
        except IOError as e:
            logger.error(f"Failed to save model weights: {e}")
    
    def load_model(
        self, 
        model_id: str, 
        version: Optional[str] = None, 
        model_class: Type[BaseModel] = None
    ) -> Tuple[Optional[BaseModel], Optional[List[np.ndarray]]]:
        """
        Load a model from the registry.
        
        Args:
            model_id: ID of the model to load
            version: Specific version to load, or None for latest
            model_class: Class to instantiate for the model
            
        Returns:
            Tuple of (model instance if model_class provided, model weights)
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in registry")
            return None, None
            
        model_info = self.models[model_id]
        
        # Determine version to load
        if version is None:
            version = model_info.get("latest_version", "1.0.0")
            
        # Find the requested version
        version_info = None
        for v in model_info["versions"]:
            if v["version"] == version:
                version_info = v
                break
                
        if version_info is None:
            logger.error(f"Version {version} not found for model {model_id}")
            return None, None
            
        # Load weights
        weights_path = self._get_weights_path(model_id, version)
        
        if not os.path.exists(weights_path):
            logger.error(f"Weights file not found for model {model_id} version {version}")
            return None, None
            
        try:
            with open(weights_path, "rb") as f:
                weights = pickle.load(f)
        except (pickle.PickleError, IOError) as e:
            logger.error(f"Failed to load model weights: {e}")
            return None, None
            
        # Create model instance if class provided
        model_instance = None
        if model_class is not None:
            try:
                model_instance = model_class(
                    model_id=model_id,
                    version=version,
                    metadata=version_info.get("extra_metadata", {})
                )
                model_instance.set_weights(weights)
            except Exception as e:
                logger.error(f"Failed to instantiate model: {e}")
                return None, weights
                
        return model_instance, weights
        
    def get_model_info(
        self, 
        model_id: str, 
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific model and version.
        
        Args:
            model_id: ID of the model
            version: Specific version to get info for, or None for latest
            
        Returns:
            Dictionary of model metadata or None if not found
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in registry")
            return None
            
        model_info = self.models[model_id]
        
        # Determine version to get info for
        if version is None:
            version = model_info.get("latest_version", "1.0.0")
            
        # Find the requested version
        for v in model_info["versions"]:
            if v["version"] == version:
                return v
                
        logger.error(f"Version {version} not found for model {model_id}")
        return None
        
    def list_models(
        self, 
        tag: Optional[str] = None, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List models in the registry with pagination support.
        
        Args:
            tag: Optional tag to filter by
            limit: Maximum number of models to return
            offset: Number of models to skip
            
        Returns:
            List of model metadata dictionaries
        """
        result = []
        
        for model_id, model_info in self.models.items():
            # Apply tag filter if specified
            if tag is not None:
                latest_version = model_info.get("latest_version", "1.0.0")
                version_info = None
                
                for v in model_info["versions"]:
                    if v["version"] == latest_version:
                        version_info = v
                        break
                        
                if version_info is None or tag not in version_info.get("tags", []):
                    continue
                    
            # Create summary info
            summary = {
                "model_id": model_id,
                "created_at": model_info.get("created_at"),
                "updated_at": model_info.get("updated_at"),
                "latest_version": model_info.get("latest_version"),
                "version_count": len(model_info.get("versions", []))
            }
            result.append(summary)
            
        # Apply pagination
        return result[offset:offset+limit]
        
    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """
        Delete a model or specific version from the registry.
        
        Args:
            model_id: ID of the model to delete
            version: Specific version to delete, or None for all versions
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in registry")
            return False
            
        if version is None:
            # Delete all versions
            model_info = self.models[model_id]
            for v in model_info["versions"]:
                weights_path = self._get_weights_path(model_id, v["version"])
                if os.path.exists(weights_path):
                    try:
                        os.remove(weights_path)
                    except OSError as e:
                        logger.error(f"Failed to delete weights file: {e}")
                        
            # Remove from registry
            del self.models[model_id]
            self._save_metadata()
            logger.info(f"Deleted model {model_id} with all versions")
            return True
        else:
            # Delete specific version
            model_info = self.models[model_id]
            updated_versions = []
            version_found = False
            
            for v in model_info["versions"]:
                if v["version"] == version:
                    version_found = True
                    # Delete weights file
                    weights_path = self._get_weights_path(model_id, version)
                    if os.path.exists(weights_path):
                        try:
                            os.remove(weights_path)
                        except OSError as e:
                            logger.error(f"Failed to delete weights file: {e}")
                else:
                    updated_versions.append(v)
                    
            if not version_found:
                logger.error(f"Version {version} not found for model {model_id}")
                return False
                
            if not updated_versions:
                # No versions left, delete the entire model
                del self.models[model_id]
            else:
                # Update model info with remaining versions
                self.models[model_id]["versions"] = updated_versions
                
                # Update latest version if needed
                if model_info.get("latest_version") == version:
                    # Set latest version to the most recent remaining version
                    latest_version = max(updated_versions, key=lambda v: v["created_at"])["version"]
                    self.models[model_id]["latest_version"] = latest_version
                    
            self._save_metadata()
            logger.info(f"Deleted version {version} of model {model_id}")
            return True 