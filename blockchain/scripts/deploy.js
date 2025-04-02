// Smart Contract Deployment Script
const fs = require('fs');
const path = require('path');
const hre = require('hardhat');
const { ethers } = require('hardhat');

// Helper function to sleep for specified milliseconds
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function main() {
  console.log('Starting deployment...');

  // Get the network
  const network = hre.network.name;
  console.log(`Deploying to ${network} network`);

  // Get accounts
  const [deployer] = await ethers.getSigners();
  console.log(`Deploying contracts with account: ${deployer.address}`);
  console.log(`Account balance: ${(await deployer.getBalance()).toString()}`);

  // Deploy FedLearningToken
  console.log('Deploying FedLearningToken...');
  const FedLearningToken = await ethers.getContractFactory('FedLearningToken');
  const initialSupply = ethers.utils.parseEther('10000000'); // 10 million tokens
  const token = await FedLearningToken.deploy(
    'Federated Learning Token',
    'FLT',
    initialSupply
  );
  await token.deployed();
  console.log(`FedLearningToken deployed to: ${token.address}`);

  // Deploy ContributionLogging
  console.log('Deploying ContributionLogging...');
  const ContributionLogging = await ethers.getContractFactory('ContributionLogging');
  const contributionLogging = await ContributionLogging.deploy(token.address);
  await contributionLogging.deployed();
  console.log(`ContributionLogging deployed to: ${contributionLogging.address}`);

  // Deploy QualityVerification
  console.log('Deploying QualityVerification...');
  const QualityVerification = await ethers.getContractFactory('QualityVerification');
  const qualityVerification = await QualityVerification.deploy(contributionLogging.address);
  await qualityVerification.deployed();
  console.log(`QualityVerification deployed to: ${qualityVerification.address}`);

  // Deploy RewardDistribution
  console.log('Deploying RewardDistribution...');
  const RewardDistribution = await ethers.getContractFactory('RewardDistribution');
  const rewardDistribution = await RewardDistribution.deploy(
    token.address,
    contributionLogging.address
  );
  await rewardDistribution.deployed();
  console.log(`RewardDistribution deployed to: ${rewardDistribution.address}`);

  // Deploy FLGovernance
  console.log('Deploying FLGovernance...');
  const FLGovernance = await ethers.getContractFactory('FLGovernance');
  const flGovernance = await FLGovernance.deploy(token.address);
  await flGovernance.deployed();
  console.log(`FLGovernance deployed to: ${flGovernance.address}`);

  // Setup roles and permissions
  console.log('Setting up roles and permissions...');
  
  try {
    // Grant AGGREGATOR_ROLE in ContributionLogging
    const AGGREGATOR_ROLE = await contributionLogging.AGGREGATOR_ROLE();
    const tx1 = await contributionLogging.grantRole(AGGREGATOR_ROLE, qualityVerification.address);
    await tx1.wait();
    console.log('Granted AGGREGATOR_ROLE to QualityVerification');
    
    await sleep(1000); // Wait 1 second between transactions
    
    const tx2 = await contributionLogging.grantRole(AGGREGATOR_ROLE, rewardDistribution.address);
    await tx2.wait();
    console.log('Granted AGGREGATOR_ROLE to RewardDistribution');
    
    await sleep(1000);
    
    // Grant VERIFIER_ROLE in QualityVerification
    const VERIFIER_ROLE = await qualityVerification.VERIFIER_ROLE();
    const tx3 = await qualityVerification.grantVerifierRole(deployer.address);
    await tx3.wait();
    console.log('Granted VERIFIER_ROLE to deployer');
    
    await sleep(1000);
    
    // Grant DISTRIBUTOR_ROLE in RewardDistribution
    const DISTRIBUTOR_ROLE = await rewardDistribution.DISTRIBUTOR_ROLE();
    const tx4 = await rewardDistribution.grantDistributorRole(deployer.address);
    await tx4.wait();
    console.log('Granted DISTRIBUTOR_ROLE to deployer');
    
    await sleep(1000);
    
    // Grant MINTER_ROLE in FedLearningToken to RewardDistribution
    const MINTER_ROLE = await token.MINTER_ROLE();
    const tx5 = await token.grantMinterRole(rewardDistribution.address);
    await tx5.wait();
    console.log('Granted MINTER_ROLE to RewardDistribution');
    
    await sleep(1000);
    
    // Transfer some tokens to the RewardDistribution contract
    const transferAmount = ethers.utils.parseEther('1000000'); // 1 million tokens
    const tx6 = await token.transfer(rewardDistribution.address, transferAmount);
    await tx6.wait();
    console.log(`Transferred ${ethers.utils.formatEther(transferAmount)} tokens to RewardDistribution`);
  } catch (error) {
    console.error('Error in setting up roles and permissions:', error);
    // Continue with deployment anyway to save the deployed contract addresses
  }

  // Save deployment information
  const deploymentInfo = {
    network,
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      FedLearningToken: {
        address: token.address,
        initialSupply: initialSupply.toString()
      },
      ContributionLogging: {
        address: contributionLogging.address
      },
      QualityVerification: {
        address: qualityVerification.address
      },
      RewardDistribution: {
        address: rewardDistribution.address
      },
      FLGovernance: {
        address: flGovernance.address
      }
    }
  };

  // Create deployed directory if it doesn't exist
  const deployedDir = path.join(__dirname, '../deployed');
  if (!fs.existsSync(deployedDir)) {
    fs.mkdirSync(deployedDir, { recursive: true });
  }

  // Write deployment info to file
  const deploymentPath = path.join(
    deployedDir,
    `deployment-${network}-${Date.now()}.json`
  );
  fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));
  console.log(`Deployment information saved to ${deploymentPath}`);

  // Write current deployment address to a specific file for easy access
  const currentDeploymentPath = path.join(deployedDir, `current-deployment.json`);
  fs.writeFileSync(currentDeploymentPath, JSON.stringify(deploymentInfo, null, 2));
  console.log(`Current deployment info saved to ${currentDeploymentPath}`);

  // Copy ABIs to deployed directory for middleware access
  console.log('Copying ABIs to deployed directory...');
  const artifactsDir = path.join(__dirname, '../artifacts/contracts');
  const contracts = [
    'FedLearningToken',
    'ContributionLogging',
    'QualityVerification',
    'RewardDistribution',
    'FLGovernance'
  ];

  for (const contract of contracts) {
    const artifactPath = path.join(
      artifactsDir,
      `${contract}.sol/${contract}.json`
    );
    const targetPath = path.join(deployedDir, `${contract}.json`);

    if (fs.existsSync(artifactPath)) {
      const artifact = JSON.parse(fs.readFileSync(artifactPath, 'utf8'));
      const abi = {
        abi: artifact.abi,
        address: deploymentInfo.contracts[contract].address,
        network
      };
      fs.writeFileSync(targetPath, JSON.stringify(abi, null, 2));
      console.log(`ABI for ${contract} copied to ${targetPath}`);
    } else {
      console.error(`Artifact for ${contract} not found at ${artifactPath}`);
    }
  }

  console.log('Deployment complete!');
}

// Execute deployment
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
