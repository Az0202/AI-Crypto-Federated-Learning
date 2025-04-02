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
