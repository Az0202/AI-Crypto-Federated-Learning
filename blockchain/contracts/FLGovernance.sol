// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title FLGovernance
 * @dev Smart contract for decentralized governance of the federated learning platform
 */
contract FLGovernance is AccessControl, ReentrancyGuard {
    // Define roles
    bytes32 public constant MEMBER_ROLE = keccak256("MEMBER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    
    // Reference to token contract for voting power
    IERC20 public governanceToken;
    
    // Proposal struct
    struct Proposal {
        uint256 id;                  // Unique identifier
        address proposer;            // Address that created the proposal
        string title;                // Short title
        string description;          // Detailed description
        uint256 createdAt;           // Creation timestamp
        uint256 votingEndsAt;        // End of voting period
        ProposalStatus status;       // Current status
        uint256 yesVotes;            // Total yes votes
        uint256 noVotes;             // Total no votes
        bytes callData;              // Function call data for execution
        address targetContract;      // Contract to call if proposal passes
        mapping(address => Vote) votes; // Votes by address
        bool executed;               // Whether the proposal has been executed
    }
    
    // Vote struct
    struct Vote {
        bool hasVoted;               // Whether the address has voted
        bool inSupport;              // Whether the vote is in support
        uint256 votingPower;         // Voting power used
    }
    
    // Proposal status enum
    enum ProposalStatus {
        Active,
        Passed,
        Rejected,
        Executed,
        Canceled
    }
    
    // Voting parameters
    struct VotingParams {
        uint256 votingPeriod;        // Voting period in seconds
        uint256 proposalThreshold;   // Minimum tokens required to create proposal
        uint256 quorumThreshold;     // Percentage of total supply required for quorum (scaled by 100)
        uint256 executionDelay;      // Delay between passing and execution in seconds
    }
    
    // Current voting parameters
    VotingParams public votingParams;
    
    // Proposal counter
    uint256 public proposalCount;
    
    // Proposals by ID
    mapping(uint256 => Proposal) public proposals;
    
    // Active proposals
    uint256[] public activeProposals;
    
    // Events
    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed proposer,
        string title,
        uint256 votingEndsAt
    );
    event VoteCast(
        uint256 indexed proposalId,
        address indexed voter,
        bool inSupport,
        uint256 votingPower
    );
    event ProposalStatusChanged(
        uint256 indexed proposalId,
        ProposalStatus status
    );
    event ProposalExecuted(
        uint256 indexed proposalId,
        address executor,
        bool success
    );
    event VotingParamsUpdated(
        uint256 votingPeriod,
        uint256 proposalThreshold,
        uint256 quorumThreshold,
        uint256 executionDelay
    );
    event GovernanceTokenUpdated(address newTokenAddress);
    
    /**
     * @dev Constructor to initialize the contract
     * @param _governanceTokenAddress Address of the governance token contract
     */
    constructor(address _governanceTokenAddress) {
        governanceToken = IERC20(_governanceTokenAddress);
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(MEMBER_ROLE, msg.sender);
        
        // Set initial voting parameters
        votingParams = VotingParams({
            votingPeriod: 7 days,         // 7-day voting period
            proposalThreshold: 1000 * 10**18, // 1000 tokens to propose
            quorumThreshold: 10,          // 10% quorum
            executionDelay: 2 days        // 2-day delay
        });
    }
    
    /**
     * @dev Create a new governance proposal
     * @param title Short title of the proposal
     * @param description Detailed description of the proposal
     * @param targetContract Address of the contract to call if proposal passes
     * @param callData Function call data for execution
     * @return proposalId ID of the created proposal
     */
    function createProposal(
        string memory title,
        string memory description,
        address targetContract,
        bytes memory callData
    )
        external
        nonReentrant
        returns (uint256 proposalId)
    {
        // Check if sender has enough tokens to create proposal
        uint256 balance = governanceToken.balanceOf(msg.sender);
        require(
            balance >= votingParams.proposalThreshold,
            "Insufficient tokens to create proposal"
        );
        
        // Increment proposal counter
        proposalCount++;
        proposalId = proposalCount;
        
        // Create new proposal
        Proposal storage newProposal = proposals[proposalId];
        newProposal.id = proposalId;
        newProposal.proposer = msg.sender;
        newProposal.title = title;
        newProposal.description = description;
        newProposal.createdAt = block.timestamp;
        newProposal.votingEndsAt = block.timestamp + votingParams.votingPeriod;
        newProposal.status = ProposalStatus.Active;
        newProposal.targetContract = targetContract;
        newProposal.callData = callData;
        newProposal.executed = false;
        
        // Add to active proposals
        activeProposals.push(proposalId);
        
        // Grant member role if not already granted
        if (!hasRole(MEMBER_ROLE, msg.sender)) {
            _grantRole(MEMBER_ROLE, msg.sender);
        }
        
        // Emit event
        emit ProposalCreated(
            proposalId,
            msg.sender,
            title,
            newProposal.votingEndsAt
        );
        
        return proposalId;
    }
    
    /**
     * @dev Cast a vote on a proposal
     * @param proposalId ID of the proposal to vote on
     * @param inSupport Whether the vote is in support
     */
    function castVote(uint256 proposalId, bool inSupport)
        external
        nonReentrant
    {
        Proposal storage proposal = proposals[proposalId];
        
        // Check if proposal exists and is active
        require(proposal.createdAt > 0, "Proposal does not exist");
        require(proposal.status == ProposalStatus.Active, "Proposal not active");
        require(block.timestamp < proposal.votingEndsAt, "Voting period ended");
        require(!proposal.votes[msg.sender].hasVoted, "Already voted");
        
        // Calculate voting power based on token balance
        uint256 votingPower = governanceToken.balanceOf(msg.sender);
        require(votingPower > 0, "No voting power");
        
        // Record vote
        proposal.votes[msg.sender] = Vote({
            hasVoted: true,
            inSupport: inSupport,
            votingPower: votingPower
        });
        
        // Update vote tallies
        if (inSupport) {
            proposal.yesVotes += votingPower;
        } else {
            proposal.noVotes += votingPower;
        }
        
        // Grant member role if not already granted
        if (!hasRole(MEMBER_ROLE, msg.sender)) {
            _grantRole(MEMBER_ROLE, msg.sender);
        }
        
        // Emit event
        emit VoteCast(proposalId, msg.sender, inSupport, votingPower);
    }
    
    /**
     * @dev Check if a proposal has reached quorum
     * @param proposalId ID of the proposal to check
     * @return hasQuorum Whether the proposal has reached quorum
     */
    function hasReachedQuorum(uint256 proposalId)
        public
        view
        returns (bool)
    {
        Proposal storage proposal = proposals[proposalId];
        
        // Total votes
        uint256 totalVotes = proposal.yesVotes + proposal.noVotes;
        
        // Total token supply
        uint256 totalSupply = governanceToken.totalSupply();
        
        // Calculate quorum threshold
        uint256 requiredVotes = (totalSupply * votingParams.quorumThreshold) / 100;
        
        return totalVotes >= requiredVotes;
    }
    
    /**
     * @dev Process a proposal after voting period ends
     * @param proposalId ID of the proposal to process
     */
    function processProposal(uint256 proposalId)
        external
        nonReentrant
    {
        Proposal storage proposal = proposals[proposalId];
        
        // Check if proposal exists and is active
        require(proposal.createdAt > 0, "Proposal does not exist");
        require(proposal.status == ProposalStatus.Active, "Proposal not active");
        require(block.timestamp >= proposal.votingEndsAt, "Voting period not ended");
        
        // Check quorum
        bool quorumReached = hasReachedQuorum(proposalId);
        
        // Determine outcome
        if (quorumReached && proposal.yesVotes > proposal.noVotes) {
            proposal.status = ProposalStatus.Passed;
        } else {
            proposal.status = ProposalStatus.Rejected;
        }
        
        // Remove from active proposals
        removeFromActiveProposals(proposalId);
        
        // Emit event
        emit ProposalStatusChanged(proposalId, proposal.status);
    }
    
    /**
     * @dev Execute a passed proposal after the execution delay
     * @param proposalId ID of the proposal to execute
     */
    function executeProposal(uint256 proposalId)
        external
        nonReentrant
    {
        Proposal storage proposal = proposals[proposalId];
        
        // Check if proposal exists and is passed
        require(proposal.createdAt > 0, "Proposal does not exist");
        require(proposal.status == ProposalStatus.Passed, "Proposal not passed");
        require(!proposal.executed, "Proposal already executed");
        
        // Check execution delay
        require(
            block.timestamp >= proposal.votingEndsAt + votingParams.executionDelay,
            "Execution delay not elapsed"
        );
        
        // Mark as executed
        proposal.executed = true;
        proposal.status = ProposalStatus.Executed;
        
        // Execute the proposal
        (bool success, ) = proposal.targetContract.call(proposal.callData);
        
        // Emit events
        emit ProposalStatusChanged(proposalId, ProposalStatus.Executed);
        emit ProposalExecuted(proposalId, msg.sender, success);
        
        // Require successful execution
        require(success, "Proposal execution failed");
    }
    
    /**
     * @dev Cancel a proposal (only proposer or admin)
     * @param proposalId ID of the proposal to cancel
     */
    function cancelProposal(uint256 proposalId)
        external
        nonReentrant
    {
        Proposal storage proposal = proposals[proposalId];
        
        // Check if proposal exists and is active
        require(proposal.createdAt > 0, "Proposal does not exist");
        require(proposal.status == ProposalStatus.Active, "Proposal not active");
        
        // Only proposer or admin can cancel
        require(
            proposal.proposer == msg.sender || hasRole(ADMIN_ROLE, msg.sender),
            "Not authorized to cancel"
        );
        
        // Update status
        proposal.status = ProposalStatus.Canceled;
        
        // Remove from active proposals
        removeFromActiveProposals(proposalId);
        
        // Emit event
        emit ProposalStatusChanged(proposalId, ProposalStatus.Canceled);
    }
    
    /**
     * @dev Remove a proposal from the active proposals array
     * @param proposalId ID of the proposal to remove
     */
    function removeFromActiveProposals(uint256 proposalId) 
        internal 
    {
        for (uint256 i = 0; i < activeProposals.length; i++) {
            if (activeProposals[i] == proposalId) {
                // Replace with the last element and pop
                activeProposals[i] = activeProposals[activeProposals.length - 1];
                activeProposals.pop();
                break;
            }
        }
    }
    
    /**
     * @dev Update voting parameters (only admin)
     * @param _votingPeriod New voting period
     * @param _proposalThreshold New proposal threshold
     * @param _quorumThreshold New quorum threshold
     * @param _executionDelay New execution delay
     */
    function updateVotingParams(
        uint256 _votingPeriod,
        uint256 _proposalThreshold,
        uint256 _quorumThreshold,
        uint256 _executionDelay
    )
        external
        onlyRole(ADMIN_ROLE)
    {
        // Validate parameters
        require(_votingPeriod > 0, "Voting period must be positive");
        require(_proposalThreshold > 0, "Proposal threshold must be positive");
        require(_quorumThreshold > 0 && _quorumThreshold <= 100, "Quorum threshold must be between 1 and 100");
        
        votingParams.votingPeriod = _votingPeriod;
        votingParams.proposalThreshold = _proposalThreshold;
        votingParams.quorumThreshold = _quorumThreshold;
        votingParams.executionDelay = _executionDelay;
        
        emit VotingParamsUpdated(
            _votingPeriod,
            _proposalThreshold,
            _quorumThreshold,
            _executionDelay
        );
    }
    
    /**
     * @dev Update the governance token address (only admin)
     * @param _newTokenAddress New token address
     */
    function updateGovernanceToken(address _newTokenAddress)
        external
        onlyRole(ADMIN_ROLE)
    {
        require(_newTokenAddress != address(0), "Invalid token address");
        governanceToken = IERC20(_newTokenAddress);
        emit GovernanceTokenUpdated(_newTokenAddress);
    }
    
    /**
     * @dev Grant member role to an address (only admin)
     * @param account Address to grant the member role
     */
    function grantMemberRole(address account)
        external
        onlyRole(ADMIN_ROLE)
    {
        _grantRole(MEMBER_ROLE, account);
    }
    
    /**
     * @dev Revoke member role from an address (only admin)
     * @param account Address to revoke the member role from
     */
    function revokeMemberRole(address account)
        external
        onlyRole(ADMIN_ROLE)
    {
        _revokeRole(MEMBER_ROLE, account);
    }
    
    /**
     * @dev Get proposal details
     * @param proposalId ID of the proposal
     * @return title Proposal title
     * @return description Proposal description
     * @return proposer Address that created the proposal
     * @return createdAt Creation timestamp
     * @return votingEndsAt End of voting period
     * @return status Current status
     * @return yesVotes Total yes votes
     * @return noVotes Total no votes
     * @return executed Whether the proposal has been executed
     */
    function getProposalDetails(uint256 proposalId)
        external
        view
        returns (
            string memory title,
            string memory description,
            address proposer,
            uint256 createdAt,
            uint256 votingEndsAt,
            ProposalStatus status,
            uint256 yesVotes,
            uint256 noVotes,
            bool executed
        )
    {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.createdAt > 0, "Proposal does not exist");
        
        return (
            proposal.title,
            proposal.description,
            proposal.proposer,
            proposal.createdAt,
            proposal.votingEndsAt,
            proposal.status,
            proposal.yesVotes,
            proposal.noVotes,
            proposal.executed
        );
    }
    
    /**
     * @dev Get all active proposals
     * @return Array of active proposal IDs
     */
    function getActiveProposals()
        external
        view
        returns (uint256[] memory)
    {
        return activeProposals;
    }
    
    /**
     * @dev Check if an address has voted on a proposal
     * @param proposalId ID of the proposal
     * @param voter Address to check
     * @return hasVoted Whether the address has voted
     * @return inSupport Whether the vote was in support
     * @return votingPower Voting power used
     */
    function getVote(uint256 proposalId, address voter)
        external
        view
        returns (bool hasVoted, bool inSupport, uint256 votingPower)
    {
        Vote storage vote = proposals[proposalId].votes[voter];
        return (vote.hasVoted, vote.inSupport, vote.votingPower);
    }
}
