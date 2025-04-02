// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Capped.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title FedLearningToken
 * @dev ERC20 token for the decentralized federated learning platform
 */
contract FedLearningToken is ERC20, ERC20Burnable, AccessControl {
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");
    
    // Maximum token supply (100 million tokens with 18 decimals)
    uint256 public immutable _cap = 100000000 * 10**18;
    
    // Token distribution parameters
    uint256 public contributorAllocation;    // Allocated to model contributors
    uint256 public ecosystemAllocation;      // Allocated to ecosystem development
    uint256 public teamAllocation;           // Allocated to founding team
    uint256 public communityAllocation;      // Allocated to community rewards and incentives
    
    // Token minting rate control
    uint256 public lastMintTimestamp;
    uint256 public mintCooldownPeriod = 1 days;  // Minimum time between mints
    uint256 public maxMintPerPeriod;             // Maximum tokens mintable per period
    
    // Events
    event TokensMinted(address to, uint256 amount, string reason);
    event MintingParametersUpdated(uint256 cooldownPeriod, uint256 maxPerPeriod);
    event AllocationUpdated(
        uint256 contributorAllocation,
        uint256 ecosystemAllocation,
        uint256 teamAllocation,
        uint256 communityAllocation
    );
    
    /**
     * @dev Constructor to initialize the contract
     * @param name Token name
     * @param symbol Token symbol
     * @param initialSupply Initial token supply to mint
     */
    constructor(
        string memory name,
        string memory symbol,
        uint256 initialSupply
    ) 
        ERC20(name, symbol) 
    {
        require(initialSupply <= _cap, "ERC20Capped: cap exceeded");
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(GOVERNANCE_ROLE, msg.sender);
        
        // Set initial allocations (percentages)
        contributorAllocation = 60;  // 60% for model contributors
        ecosystemAllocation = 20;    // 20% for ecosystem development
        teamAllocation = 10;         // 10% for founding team
        communityAllocation = 10;    // 10% for community incentives
        
        // Set minting parameters
        lastMintTimestamp = block.timestamp;
        maxMintPerPeriod = 100000 * 10**18;  // 100,000 tokens per day initially
        
        // Mint initial supply to deployer
        _mint(msg.sender, initialSupply);
    }
    
    /**
     * @dev Mint new tokens (respecting the cap)
     * @param to Address to receive the minted tokens
     * @param amount Amount of tokens to mint
     * @param reason Reason for minting (for event logging)
     */
    function mint(address to, uint256 amount, string memory reason) 
        external 
        onlyRole(MINTER_ROLE) 
    {
        // Check if cap would be exceeded
        require(ERC20.totalSupply() + amount <= _cap, "ERC20Capped: cap exceeded");
        
        // Check if minting cooldown has passed
        require(
            block.timestamp >= lastMintTimestamp + mintCooldownPeriod,
            "Minting cooldown period not elapsed"
        );
        
        // Check if amount exceeds max mint per period
        require(amount <= maxMintPerPeriod, "Exceeds maximum mint per period");
        
        // Update last mint timestamp
        lastMintTimestamp = block.timestamp;
        
        // Mint tokens
        _mint(to, amount);
        
        // Emit event
        emit TokensMinted(to, amount, reason);
    }
    
    /**
     * @dev Update minting parameters
     * @param newCooldownPeriod New cooldown period between mints
     * @param newMaxPerPeriod New maximum tokens mintable per period
     */
    function updateMintingParameters(
        uint256 newCooldownPeriod,
        uint256 newMaxPerPeriod
    ) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
    {
        mintCooldownPeriod = newCooldownPeriod;
        maxMintPerPeriod = newMaxPerPeriod;
        
        emit MintingParametersUpdated(newCooldownPeriod, newMaxPerPeriod);
    }
    
    /**
     * @dev Update token allocation percentages
     * @param newContributorAllocation New allocation for model contributors
     * @param newEcosystemAllocation New allocation for ecosystem development
     * @param newTeamAllocation New allocation for founding team
     * @param newCommunityAllocation New allocation for community incentives
     */
    function updateAllocation(
        uint256 newContributorAllocation,
        uint256 newEcosystemAllocation,
        uint256 newTeamAllocation,
        uint256 newCommunityAllocation
    ) 
        external 
        onlyRole(GOVERNANCE_ROLE) 
    {
        // Ensure percentages add up to 100
        require(
            newContributorAllocation + newEcosystemAllocation + 
            newTeamAllocation + newCommunityAllocation == 100,
            "Allocations must add up to 100%"
        );
        
        contributorAllocation = newContributorAllocation;
        ecosystemAllocation = newEcosystemAllocation;
        teamAllocation = newTeamAllocation;
        communityAllocation = newCommunityAllocation;
        
        emit AllocationUpdated(
            newContributorAllocation,
            newEcosystemAllocation,
            newTeamAllocation,
            newCommunityAllocation
        );
    }
    
    /**
     * @dev Grant minter role to an address
     * @param account Address to grant the minter role
     */
    function grantMinterRole(address account) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        _grantRole(MINTER_ROLE, account);
    }
    
    /**
     * @dev Revoke minter role from an address
     * @param account Address to revoke the minter role from
     */
    function revokeMinterRole(address account)
        external
        onlyRole(DEFAULT_ADMIN_ROLE)
    {
        _revokeRole(MINTER_ROLE, account);
    }
    
    /**
     * @dev Grant governance role to an address
     * @param account Address to grant the governance role
     */
    function grantGovernanceRole(address account)
        external
        onlyRole(DEFAULT_ADMIN_ROLE)
    {
        _grantRole(GOVERNANCE_ROLE, account);
    }
    
    /**
     * @dev Revoke governance role from an address
     * @param account Address to revoke the governance role from
     */
    function revokeGovernanceRole(address account)
        external
        onlyRole(DEFAULT_ADMIN_ROLE)
    {
        _revokeRole(GOVERNANCE_ROLE, account);
    }
    
    /**
     * @dev Return the cap on the token's total supply
     */
    function cap() public view virtual returns (uint256) {
        return _cap;
    }
    
    /**
     * @dev See {IERC20-_beforeTokenTransfer}.
     *
     * Requirements:
     *
     * - minted tokens must not cause the total supply to go over the cap.
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal virtual override {
        super._beforeTokenTransfer(from, to, amount);

        if (from == address(0)) {
            // When minting tokens
            require(
                ERC20.totalSupply() + amount <= cap(),
                "ERC20Capped: cap exceeded"
            );
        }
    }
