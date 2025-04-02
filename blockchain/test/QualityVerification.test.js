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
