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
