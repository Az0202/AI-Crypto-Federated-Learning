// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title ContributionLogging
 * @dev Smart contract for logging federated learning contributions
 */
contract ContributionLogging is AccessControl, ReentrancyGuard {
    // Define roles
    bytes32 public constant AGGREGATOR_ROLE = keccak256("AGGREGATOR_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    
    // ERC20 token for rewards
    IERC20 public rewardToken;
    
    // Model and round information
    uint256 public currentRound;
    string public currentModelVersion;
    
    // Contribution struct to store contribution metadata
    struct Contribution {
        address contributor;       // Client wallet address
        string clientId;           // Client identifier
        uint256 round;             // Training round
        uint256 timestamp;         // Timestamp when contribution was logged
        string metricsJson;        // JSON string with training metrics
        string modelVersion;       // Model version
        string updateHash;         // Hash of model update
        bool qualityVerified;      // Whether quality has been verified
        bool rewardIssued;         // Whether reward has been issued
        uint256 rewardAmount;      // Amount of tokens rewarded
        string aggregationId;      // ID of aggregation that included this contribution (if any)
    }
    
    // Aggregation struct to store aggregation metadata
    struct Aggregation {
        string aggregationId;      // Unique aggregation identifier
        uint256 round;             // Training round
        uint256 timestamp;         // Timestamp when aggregation was logged
        string modelVersion;       // Resulting model version
        string modelHash;          // Hash of aggregated model
        uint256 contributionCount; // Number of contributions included
        string[] contributionIds;  // IDs of included contributions
    }
    
    // Store all contributions by their IDs
    mapping(string => Contribution) public contributions;
    
    // Store all aggregations by their IDs
    mapping(string => Aggregation) public aggregations;
    
    // Track contribution IDs by client
    mapping(address => string[]) public clientContributions;
    
    // Track contribution IDs by round
    mapping(uint256 => string[]) public roundContributions;
    
    // Events
    event ContributionLogged(string contributionId, address contributor, uint256 round, string modelVersion);
    event QualityVerified(string contributionId, bool passed);
    event RewardIssued(string contributionId, address contributor, uint256 amount);
    event AggregationLogged(string aggregationId, uint256 round, string modelVersion, uint256 contributionCount);
    
    /**
     * @dev Constructor to initialize the contract
     * @param _rewardTokenAddress Address of the ERC20 token used for rewards
     */
    constructor(address _rewardTokenAddress) {
        rewardToken = IERC20(_rewardTokenAddress);
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(AGGREGATOR_ROLE, msg.sender);
        
        currentRound = 0;
        currentModelVersion = "0.0.1";
    }
    
    /**
     * @dev Log a new contribution
     * @param contributionId Unique identifier for the contribution
     * @param clientId Client identifier
     * @param round Training round
     * @param metricsJson JSON string with training metrics
     * @param modelVersion Model version used for training
     * @param updateHash Hash of model update
     */
    function logContribution(
        string memory contributionId,
        string memory clientId,
        uint256 round,
        string memory metricsJson,
        string memory modelVersion,
        string memory updateHash
    ) 
        external 
        nonReentrant 
    {
        // Validate round
        require(round <= currentRound, "Invalid round");
        
        // Ensure contribution ID is unique
        require(contributions[contributionId].timestamp == 0, "Contribution ID already exists");
        
        // Create and store the contribution
        Contribution memory newContribution = Contribution({
            contributor: msg.sender,
            clientId: clientId,
            round: round,
            timestamp: block.timestamp,
            metricsJson: metricsJson,
            modelVersion: modelVersion,
            updateHash: updateHash,
            qualityVerified: false,
            rewardIssued: false,
            rewardAmount: 0,
            aggregationId: ""
        });
        
        contributions[contributionId] = newContribution;
        
        // Track contribution ID by client and round
        clientContributions[msg.sender].push(contributionId);
        roundContributions[round].push(contributionId);
        
        // Emit event
        emit ContributionLogged(contributionId, msg.sender, round, modelVersion);
    }
    
    /**
     * @dev Verify the quality of a contribution
     * @param contributionId Unique identifier for the contribution
     * @param passed Whether the contribution passed quality checks
     */
    function verifyContributionQuality(
        string memory contributionId,
        bool passed
    )
        external
        nonReentrant
        onlyRole(AGGREGATOR_ROLE)
    {
        // Ensure contribution exists
        require(contributions[contributionId].timestamp > 0, "Contribution not found");
        
        // Update quality verification status
        contributions[contributionId].qualityVerified = passed;
        
        // Emit event
        emit QualityVerified(contributionId, passed);
    }
    
    /**
     * @dev Issue rewards for a contribution
     * @param contributionId Unique identifier for the contribution
     * @param amount Amount of tokens to reward
     */
    function issueReward(
        string memory contributionId,
        uint256 amount
    )
        external
        nonReentrant
        onlyRole(AGGREGATOR_ROLE)
    {
        // Ensure contribution exists and quality is verified
        Contribution storage contribution = contributions[contributionId];
        require(contribution.timestamp > 0, "Contribution not found");
        require(contribution.qualityVerified, "Quality not verified");
        require(!contribution.rewardIssued, "Reward already issued");
        
        // Update reward information
        contribution.rewardIssued = true;
        contribution.rewardAmount = amount;
        
        // Transfer tokens to contributor
        require(rewardToken.transfer(contribution.contributor, amount), "Token transfer failed");
        
        // Emit event
        emit RewardIssued(contributionId, contribution.contributor, amount);
    }
    
    /**
     * @dev Log a new model aggregation
     * @param aggregationId Unique identifier for the aggregation
     * @param round Training round
     * @param modelVersion Resulting model version
     * @param modelHash Hash of aggregated model
     * @param contributionIds Array of contribution IDs included in aggregation
     */
    function logAggregation(
        string memory aggregationId,
        uint256 round,
        string memory modelVersion,
        string memory modelHash,
        string[] memory contributionIds
    )
        external
        nonReentrant
        onlyRole(AGGREGATOR_ROLE)
    {
        // Ensure aggregation ID is unique
        require(aggregations[aggregationId].timestamp == 0, "Aggregation ID already exists");
        
        // Create and store the aggregation
        Aggregation memory newAggregation = Aggregation({
            aggregationId: aggregationId,
            round: round,
            timestamp: block.timestamp,
            modelVersion: modelVersion,
            modelHash: modelHash,
            contributionCount: contributionIds.length,
            contributionIds: contributionIds
        });
        
        aggregations[aggregationId] = newAggregation;
        
        // Update contributions with aggregation ID
        for (uint i = 0; i < contributionIds.length; i++) {
            string memory contribId = contributionIds[i];
            // Skip if contribution doesn't exist
            if (contributions[contribId].timestamp > 0) {
                contributions[contribId].aggregationId = aggregationId;
            }
        }
        
        // Update current round and model version if this is a new round
        if (round > currentRound) {
            currentRound = round;
            currentModelVersion = modelVersion;
        }
        
        // Emit event
        emit AggregationLogged(aggregationId, round, modelVersion, contributionIds.length);
    }
    
    /**
     * @dev Get contribution IDs for a specific client
     * @param client Address of the client
     * @return Array of contribution IDs
     */
    function getClientContributions(address client) 
        external 
        view 
        returns (string[] memory) 
    {
        return clientContributions[client];
    }
    
    /**
     * @dev Get contribution IDs for a specific round
     * @param round Training round
     * @return Array of contribution IDs
     */
    function getRoundContributions(uint256 round) 
        external 
        view 
        returns (string[] memory) 
    {
        return roundContributions[round];
    }
    
    /**
     * @dev Set the reward token address
     * @param _rewardTokenAddress Address of the ERC20 token used for rewards
     */
    function setRewardToken(address _rewardTokenAddress) 
        external 
        onlyRole(ADMIN_ROLE) 
    {
        rewardToken = IERC20(_rewardTokenAddress);
    }
}
