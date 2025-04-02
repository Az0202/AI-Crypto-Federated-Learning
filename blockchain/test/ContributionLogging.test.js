/**
 * Tests for the ContributionLogging contract
 */
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ContributionLogging Contract", function () {
  let ContributionLogging;
  let contributionLogger;
  let owner;
  let aggregator;
  let contributor1;
  let contributor2;
  let mockToken;

  // Test data
  const contributionId1 = "contrib_123";
  const clientId1 = "client_123";
  const round = 1;
  const metricsJson = '{"accuracy":0.85,"loss":0.15,"dataset_size":1000}';
  const modelVersion = "0.1.0";
  const updateHash = "0x123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

  beforeEach(async function () {
    // Get signers
    [owner, aggregator, contributor1, contributor2] = await ethers.getSigners();

    // Deploy mock token contract first
    const MockToken = await ethers.getContractFactory("FedLearningToken");
    mockToken = await MockToken.deploy("Federated Learning Token", "FLT", ethers.utils.parseEther("1000000"));
    await mockToken.deployed();

    // Deploy ContributionLogging contract
    ContributionLogging = await ethers.getContractFactory("ContributionLogging");
    contributionLogger = await ContributionLogging.deploy(mockToken.address);
    await contributionLogger.deployed();

    // Grant AGGREGATOR_ROLE to the aggregator account
    const AGGREGATOR_ROLE = await contributionLogger.AGGREGATOR_ROLE();
    await contributionLogger.grantRole(AGGREGATOR_ROLE, aggregator.address);
  });

  describe("Deployment", function () {
    it("Should set the right owner (admin)", async function () {
      const DEFAULT_ADMIN_ROLE = await contributionLogger.DEFAULT_ADMIN_ROLE();
      expect(await contributionLogger.hasRole(DEFAULT_ADMIN_ROLE, owner.address)).to.equal(true);
    });

    it("Should set the right reward token", async function () {
      expect(await contributionLogger.rewardToken()).to.equal(mockToken.address);
    });

    it("Should initialize with round 0", async function () {
      expect(await contributionLogger.currentRound()).to.equal(0);
    });
  });

  describe("Contribution Logging", function () {
    it("Should log a contribution", async function () {
      await contributionLogger.connect(contributor1).logContribution(
        contributionId1,
        clientId1,
        round,
        metricsJson,
        modelVersion,
        updateHash
      );

      // Get the logged contribution
      const contribution = await contributionLogger.contributions(contributionId1);
      
      expect(contribution.contributor).to.equal(contributor1.address);
      expect(contribution.clientId).to.equal(clientId1);
      expect(contribution.round).to.equal(round);
      expect(contribution.metricsJson).to.equal(metricsJson);
      expect(contribution.modelVersion).to.equal(modelVersion);
      expect(contribution.updateHash).to.equal(updateHash);
      expect(contribution.qualityVerified).to.equal(false);
      expect(contribution.rewardIssued).to.equal(false);
    });

    it("Should emit ContributionLogged event", async function () {
      await expect(
        contributionLogger.connect(contributor1).logContribution(
          contributionId1,
          clientId1,
          round,
          metricsJson,
          modelVersion,
          updateHash
        )
      )
        .to.emit(contributionLogger, "ContributionLogged")
        .withArgs(contributionId1, contributor1.address, round, modelVersion);
    });

    it("Should reject duplicate contribution IDs", async function () {
      await contributionLogger.connect(contributor1).logContribution(
        contributionId1,
        clientId1,
        round,
        metricsJson,
        modelVersion,
        updateHash
      );

      await expect(
        contributionLogger.connect(contributor1).logContribution(
          contributionId1,
          clientId1,
          round,
          metricsJson,
          modelVersion,
          updateHash
        )
      ).to.be.revertedWith("Contribution ID already exists");
    });

    it("Should track contributions by client", async function () {
      await contributionLogger.connect(contributor1).logContribution(
        contributionId1,
        clientId1,
        round,
        metricsJson,
        modelVersion,
        updateHash
      );

      const clientContributions = await contributionLogger.getClientContributions(contributor1.address);
      expect(clientContributions.length).to.equal(1);
      expect(clientContributions[0]).to.equal(contributionId1);
    });

    it("Should track contributions by round", async function () {
      await contributionLogger.connect(contributor1).logContribution(
        contributionId1,
        clientId1,
        round,
        metricsJson,
        modelVersion,
        updateHash
      );

      const roundContributions = await contributionLogger.getRoundContributions(round);
      expect(roundContributions.length).to.equal(1);
      expect(roundContributions[0]).to.equal(contributionId1);
    });
  });

  describe("Quality Verification", function () {
    beforeEach(async function () {
      // Log a contribution first
      await contributionLogger.connect(contributor1).logContribution(
        contributionId1,
        clientId1,
        round,
        metricsJson,
        modelVersion,
        updateHash
      );
    });

    it("Should allow aggregator to verify quality", async function () {
      await contributionLogger.connect(aggregator).verifyContributionQuality(contributionId1, true);
      
      const contribution = await contributionLogger.contributions(contributionId1);
      expect(contribution.qualityVerified).to.equal(true);
    });

    it("Should emit QualityVerified event", async function () {
      await expect(
        contributionLogger.connect(aggregator).verifyContributionQuality(contributionId1, true)
      )
        .to.emit(contributionLogger, "QualityVerified")
        .withArgs(contributionId1, true);
    });

    it("Should prevent non-aggregators from verifying quality", async function () {
      await expect(
        contributionLogger.connect(contributor2).verifyContributionQuality(contributionId1, true)
      ).to.be.revertedWith(/AccessControl/); // Revert with access control error
    });
  });

  describe("Reward Issuance", function () {
    const rewardAmount = ethers.utils.parseEther("10");

    beforeEach(async function () {
      // Log a contribution first
      await contributionLogger.connect(contributor1).logContribution(
        contributionId1,
        clientId1,
        round,
        metricsJson,
        modelVersion,
        updateHash
      );
      
      // Verify quality
      await contributionLogger.connect(aggregator).verifyContributionQuality(contributionId1, true);
      
      // Transfer tokens to the contribution logger for rewards
      await mockToken.transfer(contributionLogger.address, ethers.utils.parseEther("100"));
    });

    it("Should allow aggregator to issue rewards", async function () {
      await contributionLogger.connect(aggregator).issueReward(contributionId1, rewardAmount);
      
      const contribution = await contributionLogger.contributions(contributionId1);
      expect(contribution.rewardIssued).to.equal(true);
      expect(contribution.rewardAmount).to.equal(rewardAmount);
    });

    it("Should transfer tokens to contributor", async function () {
      const balanceBefore = await mockToken.balanceOf(contributor1.address);
      
      await contributionLogger.connect(aggregator).issueReward(contributionId1, rewardAmount);
      
      const balanceAfter = await mockToken.balanceOf(contributor1.address);
      expect(balanceAfter.sub(balanceBefore)).to.equal(rewardAmount);
    });

    it("Should emit RewardIssued event", async function () {
      await expect(
        contributionLogger.connect(aggregator).issueReward(contributionId1, rewardAmount)
      )
        .to.emit(contributionLogger, "RewardIssued")
        .withArgs(contributionId1, contributor1.address, rewardAmount);
    });

    it("Should prevent rewarding non-verified contributions", async function () {
      // Log another contribution without verification
      const contributionId2 = "contrib_456";
      await contributionLogger.connect(contributor1).logContribution(
        contributionId2,
        clientId1,
        round,
        metricsJson,
        modelVersion,
        updateHash
      );
      
      await expect(
        contributionLogger.connect(aggregator).issueReward(contributionId2, rewardAmount)
      ).to.be.revertedWith("Quality not verified");
    });

    it("Should prevent double rewards", async function () {
      await contributionLogger.connect(aggregator).issueReward(contributionId1, rewardAmount);
      
      await expect(
        contributionLogger.connect(aggregator).issueReward(contributionId1, rewardAmount)
      ).to.be.revertedWith("Reward already issued");
    });
  });

  describe("Aggregation Logging", function () {
    const aggregationId = "agg_123";
    const contributionIds = [contributionId1, "contrib_456", "contrib_789"];
    const modelHash = "0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    beforeEach(async function () {
      // Log contributions first
      for (let i = 0; i < contributionIds.length; i++) {
        await contributionLogger.connect(contributor1).logContribution(
          contributionIds[i],
          clientId1,
          round,
          metricsJson,
          modelVersion,
          updateHash
        );
      }
    });

    it("Should log an aggregation", async function () {
      await contributionLogger.connect(aggregator).logAggregation(
        aggregationId,
        round,
        modelVersion,
        modelHash,
        contributionIds
      );
      
      const aggregation = await contributionLogger.aggregations(aggregationId);
      
      expect(aggregation.aggregationId).to.equal(aggregationId);
      expect(aggregation.round).to.equal(round);
      expect(aggregation.modelVersion).to.equal(modelVersion);
      expect(aggregation.modelHash).to.equal(modelHash);
      expect(aggregation.contributionCount).to.equal(contributionIds.length);
    });

    it("Should update contributions with aggregation ID", async function () {
      await contributionLogger.connect(aggregator).logAggregation(
        aggregationId,
        round,
        modelVersion,
        modelHash,
        contributionIds
      );
      
      // Check all contributions were updated
      for (let i = 0; i < contributionIds.length; i++) {
        const contribution = await contributionLogger.contributions(contributionIds[i]);
        expect(contribution.aggregationId).to.equal(aggregationId);
      }
    });

    it("Should update currentRound and modelVersion for new rounds", async function () {
      const newRound = round + 1;
      const newVersion = "0.2.0";
      
      await contributionLogger.connect(aggregator).logAggregation(
        aggregationId,
        newRound,
        newVersion,
        modelHash,
        contributionIds
      );
      
      expect(await contributionLogger.currentRound()).to.equal(newRound);
      expect(await contributionLogger.currentModelVersion()).to.equal(newVersion);
    });

    it("Should emit AggregationLogged event", async function () {
      await expect(
        contributionLogger.connect(aggregator).logAggregation(
          aggregationId,
          round,
          modelVersion,
          modelHash,
          contributionIds
        )
      )
        .to.emit(contributionLogger, "AggregationLogged")
        .withArgs(aggregationId, round, modelVersion, contributionIds.length);
    });
  });

  describe("Admin Functions", function () {
    it("Should allow admin to set reward token", async function () {
      const newTokenAddress = "0x1234567890123456789012345678901234567890";
      
      await contributionLogger.setRewardToken(newTokenAddress);
      
      expect(await contributionLogger.rewardToken()).to.equal(newTokenAddress);
    });

    it("Should prevent non-admin from setting reward token", async function () {
      const newTokenAddress = "0x1234567890123456789012345678901234567890";
      
      await expect(
        contributionLogger.connect(contributor1).setRewardToken(newTokenAddress)
      ).to.be.revertedWith(/AccessControl/);
    });
  });
});

/**
 * Tests for FedLearningToken contract
 */
describe("FedLearningToken Contract", function () {
  let token;
  let owner;
  let minter;
  let governance;
  let recipient;

  beforeEach(async function () {
    // Get signers
    [owner, minter, governance, recipient] = await ethers.getSigners();

    // Deploy token contract
    const FedLearningToken = await ethers.getContractFactory("FedLearningToken");
    token = await FedLearningToken.deploy(
      "Federated Learning Token",
      "FLT",
      ethers.utils.parseEther("1000000") // Initial supply
    );
    await token.deployed();

    // Grant roles
    await token.grantMinterRole(minter.address);
    await token.grantGovernanceRole(governance.address);
  });

  describe("Deployment", function () {
    it("Should set the right name and symbol", async function () {
      expect(await token.name()).to.equal("Federated Learning Token");
      expect(await token.symbol()).to.equal("FLT");
    });

    it("Should mint initial supply to deployer", async function () {
      expect(await token.balanceOf(owner.address)).to.equal(ethers.utils.parseEther("1000000"));
    });

    it("Should set token cap correctly", async function () {
      expect(await token.cap()).to.equal(ethers.utils.parseEther("100000000")); // 100 million
    });

    it("Should set initial allocations correctly", async function () {
      expect(await token.contributorAllocation()).to.equal(60);
      expect(await token.ecosystemAllocation()).to.equal(20);
      expect(await token.teamAllocation()).to.equal(10);
      expect(await token.communityAllocation()).to.equal(10);
    });
  });

  describe("Minting", function () {
    it("Should allow minter to mint tokens", async function () {
      const mintAmount = ethers.utils.parseEther("1000");
      
      await token.connect(minter).mint(
        recipient.address,
        mintAmount,
        "Test minting"
      );
      
      expect(await token.balanceOf(recipient.address)).to.equal(mintAmount);
    });

    it("Should emit TokensMinted event", async function () {
      const mintAmount = ethers.utils.parseEther("1000");
      
      await expect(
        token.connect(minter).mint(
          recipient.address,
          mintAmount,
          "Test minting"
        )
      )
        .to.emit(token, "TokensMinted")
        .withArgs(recipient.address, mintAmount, "Test minting");
    });

    it("Should prevent non-minters from minting", async function () {
      const mintAmount = ethers.utils.parseEther("1000");
      
      await expect(
        token.connect(recipient).mint(
          recipient.address,
          mintAmount,
          "Test minting"
        )
      ).to.be.revertedWith(/AccessControl/);
    });

    it("Should enforce minting cooldown period", async function () {
      // Mint once
      await token.connect(minter).mint(
        recipient.address,
        ethers.utils.parseEther("1000"),
        "First mint"
      );
      
      // Try to mint again immediately
      await expect(
        token.connect(minter).mint(
          recipient.address,
          ethers.utils.parseEther("1000"),
          "Second mint"
        )
      ).to.be.revertedWith("Minting cooldown period not elapsed");
    });

    it("Should enforce maximum mint per period", async function () {
      // Set a large time in the future to bypass cooldown
      await network.provider.send("evm_increaseTime", [86400]);
      await network.provider.send("evm_mine");
      
      // Try to mint more than the max per period
      await expect(
        token.connect(minter).mint(
          recipient.address,
          ethers.utils.parseEther("200000"), // > default maxMintPerPeriod
          "Large mint"
        )
      ).to.be.revertedWith("Exceeds maximum mint per period");
    });

    it("Should enforce token cap", async function () {
      // Try to mint beyond the cap
      await expect(
        token.connect(minter).mint(
          recipient.address,
          ethers.utils.parseEther("100000000"), // equal to cap
          "Exceeds cap mint"
        )
      ).to.be.revertedWith("ERC20Capped: cap exceeded");
    });
  });

  describe("Governance Functions", function () {
    it("Should allow governance to update minting parameters", async function () {
      const newCooldown = 2 * 86400; // 2 days
      const newMaxPerPeriod = ethers.utils.parseEther("200000");
      
      await token.connect(governance).updateMintingParameters(
        newCooldown,
        newMaxPerPeriod
      );
      
      // Check the new parameters
      expect(await token.mintCooldownPeriod()).to.equal(newCooldown);
      expect(await token.maxMintPerPeriod()).to.equal(newMaxPerPeriod);
    });

    it("Should emit MintingParametersUpdated event", async function () {
      const newCooldown = 2 * 86400; // 2 days
      const newMaxPerPeriod = ethers.utils.parseEther("200000");
      
      await expect(
        token.connect(governance).updateMintingParameters(
          newCooldown,
          newMaxPerPeriod
        )
      )
        .to.emit(token, "MintingParametersUpdated")
        .withArgs(newCooldown, newMaxPerPeriod);
    });

    it("Should allow governance to update allocation", async function () {
      await token.connect(governance).updateAllocation(50, 20, 15, 15);
      
      expect(await token.contributorAllocation()).to.equal(50);
      expect(await token.ecosystemAllocation()).to.equal(20);
      expect(await token.teamAllocation()).to.equal(15);
      expect(await token.communityAllocation()).to.equal(15);
    });

    it("Should prevent non-governance from updating parameters", async function () {
      await expect(
        token.connect(recipient).updateMintingParameters(
          86400,
          ethers.utils.parseEther("100000")
        )
      ).to.be.revertedWith(/AccessControl/);
    });
  });

  describe("Role Management", function () {
    it("Should allow admin to grant minter role", async function () {
      await token.grantMinterRole(recipient.address);
      
      const MINTER_ROLE = await token.MINTER_ROLE();
      expect(await token.hasRole(MINTER_ROLE, recipient.address)).to.equal(true);
    });

    it("Should allow admin to revoke minter role", async function () {
      await token.revokeMinterRole(minter.address);
      
      const MINTER_ROLE = await token.MINTER_ROLE();
      expect(await token.hasRole(MINTER_ROLE, minter.address)).to.equal(false);
    });

    it("Should allow admin to grant governance role", async function () {
      await token.grantGovernanceRole(recipient.address);
      
      const GOVERNANCE_ROLE = await token.GOVERNANCE_ROLE();
      expect(await token.hasRole(GOVERNANCE_ROLE, recipient.address)).to.equal(true);
    });

    it("Should allow admin to revoke governance role", async function () {
      await token.revokeGovernanceRole(governance.address);
      
      const GOVERNANCE_ROLE = await token.GOVERNANCE_ROLE();
      expect(await token.hasRole(GOVERNANCE_ROLE, governance.address)).to.equal(false);
    });
  });
});

/**
 * Tests for QualityVerification contract
 */
describe("QualityVerification Contract", function () {
  let qualityVerification;
  let contributionLogging;
  let owner;
  let verifier;
  let contributor;

  beforeEach(async function () {
    // Get signers
    [owner, verifier, contributor] = await ethers.getSigners();

    // First deploy a mock ContributionLogging contract
    const MockContribLogging = await ethers.getContractFactory("ContributionLogging");
    contributionLogging = await MockContribLogging.deploy(ethers.constants.AddressZero);
    await contributionLogging.deployed();

    // Deploy QualityVerification contract
    const QualityVerification = await ethers.getContractFactory("QualityVerification");
    qualityVerification = await QualityVerification.deploy(contributionLogging.address);
    await qualityVerification.deployed();

    // Grant VERIFIER_ROLE to the verifier account
    const VERIFIER_ROLE = await qualityVerification.VERIFIER_ROLE();
    await qualityVerification.grantVerifierRole(verifier.address);
  });

  describe("Deployment", function () {
    it("Should set the right ContributionLogging contract", async function () {
      expect(await qualityVerification.contributionLoggingContract()).to.equal(contributionLogging.address);
    });

    it("Should initialize quality thresholds correctly", async function () {
      const thresholds = await qualityVerification.thresholds();
      
      expect(thresholds.minAccuracy).to.equal(600);      // 60.0%
      expect(thresholds.maxLoss).to.equal(2000);         // 2.0
      expect(thresholds.outlierDetection).to.equal(true);
      expect(thresholds.similarityThreshold).to.equal(700); // 70.0%
      expect(thresholds.minDatasetSize).to.equal(10);
    });
  });

  describe("Quality Verification", function () {
    const contributionId = "contrib_123";
    const metrics = '{"accuracy": 850, "loss": 150, "similarity": 900, "dataset_size": 1000}';
    const passed = true;
    const reason = "All quality checks passed";

    it("Should allow verifier to verify contributions", async function () {
      await qualityVerification.connect(verifier).verifyContribution(
        contributionId,
        metrics,
        passed,
        reason
      );
      
      const result = await qualityVerification.verificationResults(contributionId);
      
      expect(result.contributionId).to.equal(contributionId);
      expect(result.verifier).to.equal(verifier.address);
      expect(result.passed).to.equal(passed);
      expect(result.reason).to.equal(reason);
    });

    it("Should emit ContributionVerified event", async function () {
      await expect(
        qualityVerification.connect(verifier).verifyContribution(
          contributionId,
          metrics,
          passed,
          reason
        )
      )
        .to.emit(qualityVerification, "ContributionVerified")
        .withArgs(contributionId, passed, reason, verifier.address);
    });

    it("Should store verification history", async function () {
      // First verification
      await qualityVerification.connect(verifier).verifyContribution(
        contributionId,
        metrics,
        passed,
        reason
      );
      
      // Second verification (e.g., after review)
      await qualityVerification.connect(verifier).verifyContribution(
        contributionId,
        metrics,
        !passed, // opposite result
        "Failed on second review"
      );
      
      // Get verification history
      const history = await qualityVerification.getVerificationHistory(contributionId);
      
      expect(history.length).to.equal(2);
      expect(history[0].passed).to.equal(passed);
      expect(history[1].passed).to.equal(!passed);
    });

    it("Should prevent non-verifiers from verifying", async function () {
      await expect(
        qualityVerification.connect(contributor).verifyContribution(
          contributionId,
          metrics,
          passed,
          reason
        )
      ).to.be.revertedWith(/AccessControl/);
    });
  });

  describe("Quality Threshold Checking", function () {
    it("Should pass when all metrics exceed thresholds", async function () {
      const result = await qualityVerification.checkQualityThresholds(
        700,  // accuracy (above 600 threshold)
        1500, // loss (below 2000 threshold)
        800,  // similarity (above 700 threshold)
        20    // dataset size (above 10 threshold)
      );
      
      expect(result.passed).to.equal(true);
      expect(result.reason).to.equal("All quality checks passed");
    });

    it("Should fail when accuracy is below threshold", async function () {
      const result = await qualityVerification.checkQualityThresholds(
        500,  // accuracy (below 600 threshold)
        1500, // loss
        800,  // similarity
        20    // dataset size
      );
      
      expect(result.passed).to.equal(false);
      expect(result.reason).to.equal("Accuracy below threshold");
    });

    it("Should fail when loss is above threshold", async function () {
      const result = await qualityVerification.checkQualityThresholds(
        700,  // accuracy
        2500, // loss (above 2000 threshold)
        800,  // similarity
        20    // dataset size
      );
      
      expect(result.passed).to.equal(false);
      expect(result.reason).to.equal("Loss above threshold");
    });

    it("Should fail when dataset size is too small", async function () {
      const result = await qualityVerification.checkQualityThresholds(
        700,  // accuracy
        1500, // loss
        800,  // similarity
        5     // dataset size (below 10 threshold)
      );
      
      expect(result.passed).to.equal(false);
      expect(result.reason).to.equal("Dataset size too small");
    });

    it("Should fail when similarity is below threshold", async function () {
      const result = await qualityVerification.checkQualityThresholds(
        700,  // accuracy
        1500, // loss
        600,  // similarity (below 700 threshold)
        20    // dataset size
      );
      
      expect(result.passed).to.equal(false);
      expect(result.reason).to.equal("Model update is an outlier");
    });
  });

  describe("Admin Functions", function () {
    it("Should allow admin to update thresholds", async function () {
      await qualityVerification.updateThresholds(
        700,  // minAccuracy
        1500, // maxLoss
        true, // outlierDetection
        800,  // similarityThreshold
        15    // minDatasetSize
      );
      
      const thresholds = await qualityVerification.thresholds();
      
      expect(thresholds.minAccuracy).to.equal(700);
      expect(thresholds.maxLoss).to.equal(1500);
      expect(thresholds.outlierDetection).to.equal(true);
      expect(thresholds.similarityThreshold).to.equal(800);
      expect(thresholds.minDatasetSize).to.equal(15);
    });

    it("Should emit ThresholdsUpdated event", async function () {
      await expect(
        qualityVerification.updateThresholds(
          700,  // minAccuracy
          1500, // maxLoss
          true, // outlierDetection
          800,  // similarityThreshold
          15    // minDatasetSize
        )
      )
        .to.emit(qualityVerification, "ThresholdsUpdated")
        .withArgs(700, 1500, true, 800, 15);
    });

    it("Should allow admin to update ContributionLogging contract", async function () {
      const newAddress = "0x1234567890123456789012345678901234567890";
      
      await qualityVerification.updateContributionLoggingContract(newAddress);
      
      expect(await qualityVerification.contributionLoggingContract()).to.equal(newAddress);
    });

    it("Should prevent non-admin from updating thresholds", async function () {
      await expect(
        qualityVerification.connect(contributor).updateThresholds(
          700, 1500, true, 800, 15
        )
      ).to.be.revertedWith(/AccessControl/);
    });
  });
});
