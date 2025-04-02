"""
Enhanced Ethereum Signature Verification Module

This module provides secure verification of Ethereum signatures for wallet authentication.
"""

import logging
from eth_account.messages import encode_defunct
from eth_account import Account
from web3 import Web3
import time
import hashlib

logger = logging.getLogger(__name__)

class SignatureVerifier:
    """Handles secure verification of Ethereum signatures."""
    
    def __init__(self, nonce_timeout=300):
        """
        Initialize the signature verifier.
        
        Args:
            nonce_timeout: Timeout in seconds for nonce validity (default: 5 minutes)
        """
        self.nonce_timeout = nonce_timeout
        self._used_nonces = {}  # Store used nonces with timestamps
        
        # Clean up expired nonces periodically
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # Clean up every hour
    
    def verify_signature(self, wallet_address: str, signature: str, message: str) -> bool:
        """
        Verify an Ethereum signature against the provided message and wallet address.
        
        Args:
            wallet_address: The claimed Ethereum address (0x prefixed)
            signature: The signature to verify (0x prefixed)
            message: The original message that was signed
            
        Returns:
            Boolean indicating if the signature is valid
        """
        try:
            # Clean up expired nonces if needed
            self._maybe_cleanup_nonces()
            
            # Check if message contains a nonce and timestamp
            if not self._validate_message_format(message):
                logger.warning(f"Invalid message format: {message}")
                return False
            
            # Extract nonce to prevent replay attacks
            nonce = self._extract_nonce(message)
            if nonce in self._used_nonces:
                logger.warning(f"Nonce already used: {nonce}")
                return False
            
            # Check timestamp to prevent delayed replay attacks
            timestamp = self._extract_timestamp(message)
            current_time = int(time.time())
            if abs(current_time - timestamp) > self.nonce_timeout:
                logger.warning(f"Message timestamp expired: {timestamp} vs {current_time}")
                return False
            
            # Normalize addresses to checksum format
            wallet_address = Web3.to_checksum_address(wallet_address)
            
            # Create the EIP-191 compliant message hash
            message_hash = encode_defunct(text=message)
            
            # Recover the address from the signature
            recovered_address = Account.recover_message(message_hash, signature=signature)
            
            # Compare recovered address with claimed address
            is_valid = recovered_address.lower() == wallet_address.lower()
            
            # Record the nonce as used if verification was successful
            if is_valid:
                self._used_nonces[nonce] = current_time
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Signature verification error: {str(e)}")
            return False
    
    def generate_auth_message(self, client_id: str) -> str:
        """
        Generate a standardized authentication message with nonce and timestamp.
        
        Args:
            client_id: Client identifier
            
        Returns:
            A formatted message for the user to sign
        """
        timestamp = int(time.time())
        nonce = self._generate_nonce(client_id, timestamp)
        
        return (
            f"Authenticate with Federated Learning Platform\n"
            f"Client ID: {client_id}\n"
            f"Timestamp: {timestamp}\n"
            f"Nonce: {nonce}"
        )
    
    def _generate_nonce(self, client_id: str, timestamp: int) -> str:
        """
        Generate a unique nonce based on client_id and timestamp.
        
        Args:
            client_id: Client identifier
            timestamp: Current timestamp
            
        Returns:
            A unique nonce string
        """
        # Create a unique nonce by hashing client_id and timestamp
        data = f"{client_id}:{timestamp}:{os.urandom(8).hex()}"
        nonce = hashlib.sha256(data.encode()).hexdigest()[:16]
        return nonce
    
    def _validate_message_format(self, message: str) -> bool:
        """
        Validate that the message contains required elements.
        
        Args:
            message: The message to validate
            
        Returns:
            Boolean indicating if message format is valid
        """
        # Check for presence of timestamp and nonce
        return "Timestamp:" in message and "Nonce:" in message
    
    def _extract_nonce(self, message: str) -> str:
        """
        Extract nonce from the message.
        
        Args:
            message: The message containing nonce
            
        Returns:
            Extracted nonce string
        """
        try:
            # Extract nonce from message
            for line in message.split('\n'):
                if line.startswith("Nonce:"):
                    return line.split(':', 1)[1].strip()
            return ""
        except Exception:
            return ""
    
    def _extract_timestamp(self, message: str) -> int:
        """
        Extract timestamp from the message.
        
        Args:
            message: The message containing timestamp
            
        Returns:
            Extracted timestamp as integer
        """
        try:
            # Extract timestamp from message
            for line in message.split('\n'):
                if line.startswith("Timestamp:"):
                    return int(line.split(':', 1)[1].strip())
            return 0
        except Exception:
            return 0
    
    def _maybe_cleanup_nonces(self):
        """Clean up expired nonces to prevent memory growth."""
        current_time = time.time()
        
        # Only clean up if enough time has passed
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        # Remove expired nonces
        expiration_time = current_time - self.nonce_timeout
        self._used_nonces = {
            nonce: timestamp 
            for nonce, timestamp in self._used_nonces.items()
            if timestamp > expiration_time
        }
        
        self._last_cleanup = current_time
        logger.info(f"Cleaned up nonces. Remaining count: {len(self._used_nonces)}")


# Example usage
if __name__ == "__main__":
    import os
    
    # Create signature verifier
    verifier = SignatureVerifier()
    
    # Generate message for a client to sign
    message = verifier.generate_auth_message("client_123")
    print(f"Message to sign:\n{message}")
    
    # In a real scenario, the client would sign this message with their private key
    # Here's how it might work with a test wallet:
    
    # Generate a test private key (don't use this in production!)
    test_private_key = "0x" + "1" * 64
    test_account = Account.from_key(test_private_key)
    
    # Sign the message
    message_hash = encode_defunct(text=message)
    signed_message = test_account.sign_message(message_hash)
    signature = signed_message.signature.hex()
    
    # Verify the signature
    is_valid = verifier.verify_signature(
        test_account.address,
        signature,
        message
    )
    
    print(f"Signature verification result: {is_valid}")
