// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title RewardDistribution
 * @dev Smart contract for distributing rewards to federated learning contributors
 */
contract RewardDistribution is AccessControl, ReentrancyGuard {
    // Define roles
    bytes32 public constant DISTRIBUTOR_ROLE = keccak256("DISTRIBUTOR_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    
    // Reference to token contract
    IERC20 public rewardToken;
    
    // Reference to ContributionLogging contract
    address public contributionLoggingContract;
    
    // Reward policy parameters
    struct RewardPolicy {
        uint256 baseReward;            // Base reward for each contribution (in tokens * 10^18)
        uint256 accuracyMultiplier;    // Multiplier for accuracy (scaled by 1000)
        uint256 datasetSizeMultiplier; // Multiplier for dataset size (scaled by 1000)
        uint256 maxReward;             // Maximum reward per contribution (in tokens * 10^18)
        uint256 roundRewardBudget;     // Reward budget per round (in tokens * 10^18)
    }
    
    // Current reward policy
    RewardPolicy public policy;
    
    // Reward record struct
    struct RewardRecord {
        string contributionId;     // ID of the rewarded contribution
        address recipient;         // Address that received the reward
        uint256 amount;            // Amount of tokens rewarded
        uint256 timestamp;         // When reward was issued
        uint256 round;             // Training round
    }
    
    // Track rewards by contribution ID
    mapping(string => RewardRecord) public rewardsByContribution;
    
    // Track rewards by round
    mapping(uint256 => uint256) public rewardsByRound;
    
    // Track total rewards issued
    uint256 public totalRewardsIssued;
    
    // Track rewards by recipient
    mapping(address => RewardRecord[]) public rewardsByRecipient;
    
    // Events
    event RewardPolicyUpdated(
        uint256 baseReward,
        uint256 accuracyMultiplier,
        uint256 datasetSizeMultiplier,
        uint256 maxReward,
        uint256 roundRewardBudget
    );
    event RewardIssued(
        string contributionId,
        address recipient,
        uint256 amount,
        uint256 round
    );
    event ContributionLoggingContractUpdated(address newContractAddress);
    event RewardTokenUpdated(address newTokenAddress);
    
    /**
     * @dev Constructor to initialize the contract
     * @param _rewardTokenAddress Address of the reward token contract
     * @param _contributionLoggingContract Address of the ContributionLogging contract
     */
    constructor(
        address _rewardTokenAddress,
        address _contributionLoggingContract
    ) {
        rewardToken = IERC20(_rewardTokenAddress);
        contributionLoggingContract = _contributionLoggingContract;
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(DISTRIBUTOR_ROLE, msg.sender);
        
        // Set initial reward policy
        policy = RewardPolicy({
            baseReward: 10 * 10**18,          // 10 tokens base reward
            accuracyMultiplier: 1500,          // 1.5x multiplier at 100% accuracy
            datasetSizeMultiplier: 2000,       // 2.0x multiplier for large datasets
            maxReward: 50 * 10**18,           // 50 tokens maximum per contribution
            roundRewardBudget: 1000 * 10**18  // 1000 tokens budget per round
        });
    }
    
    /**
     * @dev Update reward policy parameters
     * @param _baseReward New base reward
     * @param _accuracyMultiplier New accuracy multiplier
     * @param _datasetSizeMultiplier New dataset size multiplier
     * @param _maxReward New maximum reward
     * @param _roundRewardBudget New round reward budget
     */
    function updateRewardPolicy(
        uint256 _baseReward,
        uint256 _accuracyMultiplier,
        uint256 _datasetSizeMultiplier,
        uint256 _maxReward,
        uint256 _roundRewardBudget
    )
        external
        onlyRole(ADMIN_ROLE)
    {
        policy.baseReward = _baseReward;
        policy.accuracyMultiplier = _accuracyMultiplier;
        policy.datasetSizeMultiplier = _datasetSizeMultiplier;
        policy.maxReward = _maxReward;
        policy.roundRewardBudget = _roundRewardBudget;
        
        emit RewardPolicyUpdated(
            _baseReward,
            _accuracyMultiplier,
            _datasetSizeMultiplier,
            _maxReward,
            _roundRewardBudget
        );
    }
    
    /**
     * @dev Issue reward for a contribution
     * @param contributionId ID of the contribution to reward
     * @param recipient Address to receive the reward
     * @param accuracy Accuracy of the contribution (scaled by 1000)
     * @param datasetSize Size of the dataset used for training
     * @param round Training round
     */
    function issueReward(
        string memory contributionId,
        address recipient,
        uint256 accuracy,
        uint256 datasetSize,
        uint256 round
    )
        external
        nonReentrant
        onlyRole(DISTRIBUTOR_ROLE)
    {
        // Ensure contribution hasn't been rewarded already
        require(
            rewardsByContribution[contributionId].timestamp == 0,
            "Contribution already rewarded"
        );
        
        // Calculate reward amount based on policy
        uint256 rewardAmount = calculateReward(accuracy, datasetSize);
        
        // Ensure we don't exceed round budget
        uint256 roundRewardsSoFar = rewardsByRound[round];
        require(
            roundRewardsSoFar + rewardAmount <= policy.roundRewardBudget,
            "Round reward budget exceeded"
        );
        
        // Transfer tokens to recipient
        require(
            rewardToken.transfer(recipient, rewardAmount),
            "Token transfer failed"
        );
        
        // Record the reward
        RewardRecord memory record = RewardRecord({
            contributionId: contributionId,
            recipient: recipient,
            amount: rewardAmount,
            timestamp: block.timestamp,
            round: round
        });
        
        rewardsByContribution[contributionId] = record;
        rewardsByRecipient[recipient].push(record);
        rewardsByRound[round] += rewardAmount;
        totalRewardsIssued += rewardAmount;
        
        // Update the contribution logging contract if it exists
        if (contributionLoggingContract != address(0)) {
            // Interface for ContributionLogging contract
            bytes memory callData = abi.encodeWithSignature(
                "issueReward(string,uint256)",
                contributionId,
                rewardAmount
            );
            
            (bool success, ) = contributionLoggingContract.call(callData);
            // We don't revert here to avoid blocking rewards if the logging contract has issues
            if (!success) {
                // Perhaps we could emit an event instead
                emit ContributionLoggingContractUpdated(contributionLoggingContract);
            }
        }
        
        // Emit event
        emit RewardIssued(contributionId, recipient, rewardAmount, round);
    }
    
    /**
     * @dev Calculate reward based on contribution metrics
     * @param accuracy Accuracy of the contribution (scaled by 1000)
     * @param datasetSize Size of the dataset used for training
     * @return Calculated reward amount
     */
    function calculateReward(
        uint256 accuracy,
        uint256 datasetSize
    )
        public
        view
        returns (uint256)
    {
        // Base reward
        uint256 reward = policy.baseReward;
        
        // Apply accuracy multiplier (linear from 0 to accuracyMultiplier)
        reward = reward * (1000 + (accuracy * (policy.accuracyMultiplier - 1000)) / 1000) / 1000;
        
        // Apply dataset size multiplier (logarithmic scale)
        // The formula scales from 1.0 to datasetSizeMultiplier based on dataset size
        // For simplicity, we use some threshold values
        uint256 sizeMultiplier;
        if (datasetSize < 10) {
            sizeMultiplier = 1000; // 1.0x for small datasets
        } else if (datasetSize < 100) {
            sizeMultiplier = 1000 + (policy.datasetSizeMultiplier - 1000) / 3; // ~1.33x
        } else if (datasetSize < 1000) {
            sizeMultiplier = 1000 + (policy.datasetSizeMultiplier - 1000) * 2 / 3; // ~1.67x
        } else {
            sizeMultiplier = policy.datasetSizeMultiplier; // Full multiplier for large datasets
        }
        
        reward = reward * sizeMultiplier / 1000;
        
        // Cap at maximum reward
        if (reward > policy.maxReward) {
            reward = policy.maxReward;
        }
        
        return reward;
    }
    
    /**
     * @dev Get rewards received by an address
     * @param recipient Address to query
     * @return Array of reward records
     */
    function getRewardsByRecipient(address recipient)
        external
        view
        returns (RewardRecord[] memory)
    {
        return rewardsByRecipient[recipient];
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
     * @dev Update the reward token address
     * @param _newTokenAddress New token address
     */
    function updateRewardToken(address _newTokenAddress)
        external
        onlyRole(ADMIN_ROLE)
    {
        rewardToken = IERC20(_newTokenAddress);
        emit RewardTokenUpdated(_newTokenAddress);
    }
    
    /**
     * @dev Grant distributor role to an address
     * @param account Address to grant the distributor role
     */
    function grantDistributorRole(address account)
        external
        onlyRole(ADMIN_ROLE)
    {
        _grantRole(DISTRIBUTOR_ROLE, account);
    }
    
    /**
     * @dev Revoke distributor role from an address
     * @param account Address to revoke the distributor role from
     */
    function revokeDistributorRole(address account)
        external
        onlyRole(ADMIN_ROLE)
    {
        _revokeRole(DISTRIBUTOR_ROLE, account);
    }
    
    /**
     * @dev Withdraw any excess tokens (only admin)
     * @param amount Amount of tokens to withdraw
     * @param recipient Address to receive the tokens
     */
    function withdrawTokens(uint256 amount, address recipient)
        external
        onlyRole(ADMIN_ROLE)
    {
        require(amount > 0, "Amount must be greater than zero");
        require(recipient != address(0), "Cannot withdraw to zero address");
        
        require(
            rewardToken.transfer(recipient, amount),
            "Token transfer failed"
        );
    }
}
