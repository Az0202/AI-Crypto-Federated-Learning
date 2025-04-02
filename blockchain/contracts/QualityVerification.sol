// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title QualityVerification
 * @dev Smart contract for verifying the quality of model updates in federated learning
 */
contract QualityVerification is AccessControl, ReentrancyGuard {
    // Define roles
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    
    // Reference to ContributionLogging contract
    address public contributionLoggingContract;
    
    // Quality metrics thresholds
    struct QualityThresholds {
        uint256 minAccuracy;        // Minimum accuracy (scaled by 1000, i.e., 70.5% = 705)
        uint256 maxLoss;            // Maximum loss (scaled by 1000)
        bool outlierDetection;      // Whether to use outlier detection
        uint256 similarityThreshold; // Minimum similarity threshold (scaled by 1000)
        uint256 minDatasetSize;     // Minimum dataset size
    }
    
    // Current quality thresholds
    QualityThresholds public thresholds;
    
    // Verification result struct
    struct VerificationResult {
        string contributionId;     // ID of the contribution verified
        address verifier;          // Address that performed verification
        bool passed;               // Whether the contribution passed verification
        string reason;             // Reason for verification result
        uint256 timestamp;         // When verification was performed
    }
    
    // Store verification results by contribution ID
    mapping(string => VerificationResult) public verificationResults;
    
    // History of verifications by contribution ID
    mapping(string => VerificationResult[]) public verificationHistory;
    
    // Events
    event ThresholdsUpdated(
        uint256 minAccuracy,
        uint256 maxLoss,
        bool outlierDetection,
        uint256 similarityThreshold,
        uint256 minDatasetSize
    );
    event ContributionVerified(
        string contributionId,
        bool passed,
        string reason,
        address verifier
    );
    event ContributionLoggingContractUpdated(address newContractAddress);
    
    /**
     * @dev Constructor to initialize the contract
     * @param _contributionLoggingContract Address of the ContributionLogging contract
     */
    constructor(address _contributionLoggingContract) {
        contributionLoggingContract = _contributionLoggingContract;
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(VERIFIER_ROLE, msg.sender);
        
        // Set initial quality thresholds
        thresholds = QualityThresholds({
            minAccuracy: 600,         // 60.0%
            maxLoss: 2000,            // 2.0
            outlierDetection: true,
            similarityThreshold: 700, // 70.0%
            minDatasetSize: 10        // Minimum 10 samples
        });
    }
    
    /**
     * @dev Update quality thresholds
     * @param _minAccuracy New minimum accuracy threshold (scaled by 1000)
     * @param _maxLoss New maximum loss threshold (scaled by 1000)
     * @param _outlierDetection Whether to use outlier detection
     * @param _similarityThreshold New similarity threshold (scaled by 1000)
     * @param _minDatasetSize New minimum dataset size
     */
    function updateThresholds(
        uint256 _minAccuracy,
        uint256 _maxLoss,
        bool _outlierDetection,
        uint256 _similarityThreshold,
        uint256 _minDatasetSize
    )
        external
        onlyRole(ADMIN_ROLE)
    {
        thresholds.minAccuracy = _minAccuracy;
        thresholds.maxLoss = _maxLoss;
        thresholds.outlierDetection = _outlierDetection;
        thresholds.similarityThreshold = _similarityThreshold;
        thresholds.minDatasetSize = _minDatasetSize;
        
        emit ThresholdsUpdated(
            _minAccuracy,
            _maxLoss,
            _outlierDetection,
            _similarityThreshold,
            _minDatasetSize
        );
    }
    
    /**
     * @dev Submit verification result for a contribution
     * @param contributionId ID of the contribution to verify
     * @param metrics JSON string with metrics used for verification
     * @param passed Whether the contribution passed verification
     * @param reason Reason for the verification result
     */
    function verifyContribution(
        string memory contributionId,
        string memory metrics,
        bool passed,
        string memory reason
    )
        external
        nonReentrant
        onlyRole(VERIFIER_ROLE)
    {
        // Create verification result
        VerificationResult memory result = VerificationResult({
            contributionId: contributionId,
            verifier: msg.sender,
            passed: passed,
            reason: reason,
            timestamp: block.timestamp
        });
        
        // Store verification result
        verificationResults[contributionId] = result;
        
        // Add to verification history
        verificationHistory[contributionId].push(result);
        
        // Update the contribution logging contract if it exists
        if (contributionLoggingContract != address(0)) {
            // Interface for ContributionLogging contract
            bytes memory callData = abi.encodeWithSignature(
                "verifyContributionQuality(string,bool)",
                contributionId,
                passed
            );
            
            (bool success, ) = contributionLoggingContract.call(callData);
            require(success, "Failed to update ContributionLogging contract");
        }
        
        // Emit event
        emit ContributionVerified(contributionId, passed, reason, msg.sender);
    }
    
    /**
     * @dev Verify contribution against current thresholds
     * @param accuracy Accuracy value (scaled by 1000)
     * @param loss Loss value (scaled by 1000)
     * @param similarity Similarity value (scaled by 1000)
     * @param datasetSize Size of the dataset used for training
     * @return passed Whether the contribution passed verification
     * @return reason Reason for the verification result
     */
    function checkQualityThresholds(
        uint256 accuracy,
        uint256 loss,
        uint256 similarity,
        uint256 datasetSize
    )
        external
        view
        returns (bool passed, string memory reason)
    {
        // Check accuracy
        if (accuracy < thresholds.minAccuracy) {
            return (false, "Accuracy below threshold");
        }
        
        // Check loss
        if (loss > thresholds.maxLoss) {
            return (false, "Loss above threshold");
        }
        
        // Check dataset size
        if (datasetSize < thresholds.minDatasetSize) {
            return (false, "Dataset size too small");
        }
        
        // Check similarity if outlier detection is enabled
        if (thresholds.outlierDetection && similarity < thresholds.similarityThreshold) {
            return (false, "Model update is an outlier");
        }
        
        // All checks passed
        return (true, "All quality checks passed");
    }
    
    /**
     * @dev Get verification history for a contribution
     * @param contributionId ID of the contribution
     * @return Array of verification results
     */
    function getVerificationHistory(string memory contributionId)
        external
        view
        returns (VerificationResult[] memory)
    {
        return verificationHistory[contributionId];
    }
    
    /**
     * @dev Update the ContributionLogging contract address
     * @param _newContractAddress New contract address
     */
    function updateContributionLoggingContract(address _newContractAddress)
        external
        onlyRole(ADMIN_ROLE)
    {
        contributionLoggingContract = _newContractAddress;
        emit ContributionLoggingContractUpdated(_newContractAddress);
    }
    
    /**
     * @dev Grant verifier role to an address
     * @param account Address to grant the verifier role
     */
    function grantVerifierRole(address account)
        external
        onlyRole(ADMIN_ROLE)
    {
        _grantRole(VERIFIER_ROLE, account);
    }
    
    /**
     * @dev Revoke verifier role from an address
     * @param account Address to revoke the verifier role from
     */
    function revokeVerifierRole(address account)
        external
        onlyRole(ADMIN_ROLE)
    {
        _revokeRole(VERIFIER_ROLE, account);
    }
}
