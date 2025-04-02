"""
API Server for Decentralized Federated Learning Platform

This module implements a FastAPI server that exposes the federated learning
system to clients, handling contribution submissions, model distribution,
and platform metrics.
"""

import os
import json
import logging
import time
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Union
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import jwt
from datetime import datetime, timedelta
import hashlib
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules - in a real implementation, these would be proper imports
# For demonstration, we'll create stub imports and mock implementations
class GlobalAggregator:
    """Stub for GlobalAggregator"""
    def __init__(self): pass
    def receive_contribution(self, client_id, metrics, model_update): pass
    def get_global_model_weights(self): return [np.zeros((10, 10))]
    def get_global_model_info(self): return {"model_version": "0.1.0", "current_round": 1}
    def get_contribution_stats(self): return {"total_contributions": 0}

class BlockchainMiddleware:
    """Stub for BlockchainMiddleware"""
    def __init__(self): pass
    async def log_contribution(self, contribution_id, client_id, round_num, metrics_json, model_version, update_hash): return "0x123"
    async def verify_contribution_quality(self, contribution_id, passed): return "0x123"
    async def issue_reward(self, contribution_id, recipient, accuracy, dataset_size, round_num): return "0x123"
    async def get_token_balance(self, address=None): return 1000 * 10**18

# Create FastAPI app
app = FastAPI(
    title="Decentralized Federated Learning API",
    description="API for the decentralized federated learning platform with tokenized incentives",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize components
# In a real implementation, these would be initialized with proper configuration
global_aggregator = GlobalAggregator()
blockchain_middleware = BlockchainMiddleware()

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "temp_secret_key_replace_in_production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# Request models
class ContributionRequest(BaseModel):
    """Model for submitting a training contribution"""
    client_id: str = Field(..., description="Unique identifier for the client")
    metrics: Dict[str, Union[float, int, str]] = Field(..., description="Training metrics")
    model_update_base64: str = Field(..., description="Base64 encoded model update")
    
    @validator('metrics')
    def validate_metrics(cls, v):
        required_metrics = ['loss', 'accuracy', 'dataset_size']
        for metric in required_metrics:
            if metric not in v:
                raise ValueError(f"Missing required metric: {metric}")
        return v

class LoginRequest(BaseModel):
    """Model for login request"""
    wallet_address: str = Field(..., description="Ethereum wallet address")
    signed_message: str = Field(..., description="Signed message proving wallet ownership")
    
    @validator('wallet_address')
    def validate_wallet_address(cls, v):
        # Basic validation for Ethereum address format
        if not v.startswith('0x') or len(v) != 42:
            raise ValueError("Invalid Ethereum wallet address format")
        return v

class VoteRequest(BaseModel):
    """Model for voting on a governance proposal"""
    proposal_id: int = Field(..., description="ID of the proposal to vote on")
    in_support: bool = Field(..., description="Whether the vote is in support")

class ProposalRequest(BaseModel):
    """Model for creating a governance proposal"""
    title: str = Field(..., description="Short title of the proposal")
    description: str = Field(..., description="Detailed description of the proposal")
    target_contract: str = Field(..., description="Address of the contract to call if proposal passes")
    call_data: str = Field(..., description="Hex-encoded function call data for execution")

# Response models
class ContributionResponse(BaseModel):
    """Model for contribution submission response"""
    status: str
    contribution_id: str
    verified: bool
    current_round: int
    model_version: str
    tx_hash: Optional[str] = None

class ModelResponse(BaseModel):
    """Model for global model response"""
    model_version: str
    current_round: int
    weights_base64: str
    last_updated: str

class StatsResponse(BaseModel):
    """Model for platform statistics response"""
    total_contributions: int
    active_clients: int
    current_round: int
    model_version: str
    total_rewards_issued: str  # In token units

class TokenResponse(BaseModel):
    """Model for token balance response"""
    balance: str  # In token units
    wallet_address: str

class LoginResponse(BaseModel):
    """Model for login response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds until expiration

class GovernanceResponse(BaseModel):
    """Model for governance action response"""
    status: str
    tx_hash: Optional[str] = None
    proposal_id: Optional[int] = None

# Authentication functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_ethereum_signature(wallet_address: str, signed_message: str) -> bool:
    """
    Verify an Ethereum signed message.
    
    In a real implementation, this would use eth_account or similar to
    recover the signer address from the signature and verify it matches.
    
    For this example, we'll just simulate verification.
    """
    # Simplified for demonstration - in production use proper verification
    return True

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current user from JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        wallet_address = payload.get("sub")
        if wallet_address is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return wallet_address
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Helper functions
def decode_model_update(base64_data: str) -> List[np.ndarray]:
    """Decode base64 model update back to numpy arrays"""
    try:
        # Decode base64 to bytes
        binary_data = base64.b64decode(base64_data)
        
        # Load from bytes (assuming numpy .npz format)
        with np.load(binary_data) as data:
            # Convert file-like object to list of arrays
            arrays = [data[key] for key in data.files]
        
        return arrays
    except Exception as e:
        logger.error(f"Failed to decode model update: {e}")
        raise ValueError(f"Failed to decode model update: {e}")

def encode_model_weights(weights: List[np.ndarray]) -> str:
    """Encode model weights to base64 string"""
    try:
        # Save numpy arrays to in-memory file
        with np.memmap('.temp.npz', dtype='float32', mode='w+', shape=(1,)) as temp:
            np.savez(temp, *weights)
        
        # Read the file and encode to base64
        with open('.temp.npz', 'rb') as f:
            binary_data = f.read()
        
        # Remove temporary file
        os.remove('.temp.npz')
        
        # Encode to base64
        base64_data = base64.b64encode(binary_data).decode('utf-8')
        
        return base64_data
    except Exception as e:
        logger.error(f"Failed to encode model weights: {e}")
        raise ValueError(f"Failed to encode model weights: {e}")

def hash_model_update(model_update: List[np.ndarray]) -> str:
    """Create a hash of model update weights"""
    try:
        # Concatenate all weights into a single byte array
        all_weights = b''
        for layer_weights in model_update:
            all_weights += layer_weights.tobytes()
        
        # Create SHA-256 hash
        hasher = hashlib.sha256()
        hasher.update(all_weights)
        
        return "0x" + hasher.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash model update: {e}")
        raise ValueError(f"Failed to hash model update: {e}")

# API routes
@app.post("/api/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate with wallet signature and get JWT token"""
    try:
        # Verify Ethereum signature
        if not verify_ethereum_signature(request.wallet_address, request.signed_message):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature",
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": request.wallet_address},
            expires_delta=access_token_expires
        )
        
        return LoginResponse(
            access_token=access_token,
            expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.post("/api/contributions", response_model=ContributionResponse)
async def submit_contribution(
    request: ContributionRequest,
    background_tasks: BackgroundTasks,
    wallet_address: str = Depends(get_current_user)
):
    """Submit a model update contribution"""
    try:
        # Decode model update
        try:
            model_update_arrays = decode_model_update(request.model_update_base64)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model update format: {str(e)}"
            )
        
        # Create unique contribution ID
        contribution_id = f"{request.client_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Calculate hash of model update for blockchain
        update_hash = hash_model_update(model_update_arrays)
        
        # Submit to aggregator
        model_update = {
            'weights': model_update_arrays,
            'metadata': {
                'client_id': request.client_id,
                'timestamp': int(time.time()),
                'wallet_address': wallet_address
            }
        }
        
        result = global_aggregator.receive_contribution(
            client_id=request.client_id,
            metrics=request.metrics,
            model_update=model_update
        )
        
        # Log contribution to blockchain in background
        # This prevents blocking the API response while waiting for blockchain
        background_tasks.add_task(
            log_contribution_to_blockchain,
            contribution_id=contribution_id,
            client_id=request.client_id,
            metrics=request.metrics,
            model_version=result['model_version'],
            update_hash=update_hash,
            wallet_address=wallet_address,
            round_num=result['current_round']
        )
        
        return ContributionResponse(
            status="success",
            contribution_id=contribution_id,
            verified=result.get('verified', False),
            current_round=result['current_round'],
            model_version=result['model_version']
        )
    except Exception as e:
        logger.error(f"Contribution submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

async def log_contribution_to_blockchain(
    contribution_id: str,
    client_id: str,
    metrics: Dict[str, Any],
    model_version: str,
    update_hash: str,
    wallet_address: str,
    round_num: int
):
    """Background task to log contribution to blockchain"""
    try:
        # Convert metrics to JSON string
        metrics_json = json.dumps(metrics)
        
        # Log contribution on blockchain
        tx_hash = await blockchain_middleware.log_contribution(
            contribution_id=contribution_id,
            client_id=client_id,
            round_num=round_num,
            metrics_json=metrics_json,
            model_version=model_version,
            update_hash=update_hash
        )
        
        # Verify quality (usually this would be done after proper evaluation)
        # For demo purposes, we'll assume it passes
        await blockchain_middleware.verify_contribution_quality(
            contribution_id=contribution_id,
            passed=True
        )
        
        # Issue reward
        accuracy = int(metrics.get('accuracy', 0.7) * 1000)  # Scale to integer (e.g., 0.85 -> 850)
        dataset_size = int(metrics.get('dataset_size', 100))
        
        await blockchain_middleware.issue_reward(
            contribution_id=contribution_id,
            recipient=wallet_address,
            accuracy=accuracy,
            dataset_size=dataset_size,
            round_num=round_num
        )
        
        logger.info(f"Blockchain processing completed for contribution {contribution_id}")
    except Exception as e:
        logger.error(f"Blockchain processing failed for contribution {contribution_id}: {e}")

@app.get("/api/model", response_model=ModelResponse)
async def get_global_model(wallet_address: str = Depends(get_current_user)):
    """Get the current global model weights"""
    try:
        # Get global model info
        model_info = global_aggregator.get_global_model_info()
        
        # Get model weights
        weights = global_aggregator.get_global_model_weights()
        
        # Encode weights to base64
        weights_base64 = encode_model_weights(weights)
        
        return ModelResponse(
            model_version=model_info['model_version'],
            current_round=model_info['current_round'],
            weights_base64=weights_base64,
            last_updated=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Failed to get global model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/stats", response_model=StatsResponse)
async def get_platform_stats():
    """Get platform statistics"""
    try:
        # Get stats from aggregator
        aggregator_stats = global_aggregator.get_contribution_stats()
        
        # Get model info
        model_info = global_aggregator.get_global_model_info()
        
        # In a real implementation, get more detailed stats from blockchain
        # For now, we'll use placeholder values
        return StatsResponse(
            total_contributions=aggregator_stats['total_contributions'],
            active_clients=len(aggregator_stats.get('by_client', {})),
            current_round=model_info['current_round'],
            model_version=model_info['model_version'],
            total_rewards_issued="1000000"  # Placeholder
        )
    except Exception as e:
        logger.error(f"Failed to get platform stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/token/balance", response_model=TokenResponse)
async def get_token_balance(wallet_address: str = Depends(get_current_user)):
    """Get token balance for the authenticated wallet"""
    try:
        # Get token balance from blockchain
        balance = await blockchain_middleware.get_token_balance(wallet_address)
        
        # Convert to string (to avoid precision issues with large integers in JSON)
        balance_str = str(balance)
        
        return TokenResponse(
            balance=balance_str,
            wallet_address=wallet_address
        )
    except Exception as e:
        logger.error(f"Failed to get token balance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/governance/vote", response_model=GovernanceResponse)
async def vote_on_proposal(
    request: VoteRequest,
    background_tasks: BackgroundTasks,
    wallet_address: str = Depends(get_current_user)
):
    """Vote on a governance proposal"""
    try:
        # In a real implementation, this would call the blockchain_middleware
        # to cast a vote on the governance contract
        background_tasks.add_task(
            cast_vote_on_blockchain,
            proposal_id=request.proposal_id,
            in_support=request.in_support
        )
        
        return GovernanceResponse(
            status="success",
            proposal_id=request.proposal_id
        )
    except Exception as e:
        logger.error(f"Failed to vote on proposal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

async def cast_vote_on_blockchain(proposal_id: int, in_support: bool):
    """Background task to cast vote on blockchain"""
    # Mock implementation - in a real system, this would call blockchain_middleware
    await asyncio.sleep(1)  # Simulate blockchain interaction
    logger.info(f"Vote cast for proposal {proposal_id}, in support: {in_support}")

@app.post("/api/governance/propose", response_model=GovernanceResponse)
async def create_proposal(
    request: ProposalRequest,
    background_tasks: BackgroundTasks,
    wallet_address: str = Depends(get_current_user)
):
    """Create a new governance proposal"""
    try:
        # In a real implementation, this would call the blockchain_middleware
        # to create a proposal on the governance contract
        # For now, we'll simulate it
        background_tasks.add_task(
            create_proposal_on_blockchain,
            title=request.title,
            description=request.description,
            target_contract=request.target_contract,
            call_data=request.call_data
        )
        
        return GovernanceResponse(
            status="success",
            proposal_id=123  # Mock ID
        )
    except Exception as e:
        logger.error(f"Failed to create proposal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

async def create_proposal_on_blockchain(
    title: str,
    description: str,
    target_contract: str,
    call_data: str
):
    """Background task to create proposal on blockchain"""
    # Mock implementation - in a real system, this would call blockchain_middleware
    await asyncio.sleep(1)  # Simulate blockchain interaction
    logger.info(f"Proposal created: {title}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
