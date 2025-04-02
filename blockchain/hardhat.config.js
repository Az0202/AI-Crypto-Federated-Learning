require('@nomiclabs/hardhat-waffle');
require('@nomiclabs/hardhat-ethers');
require('@nomiclabs/hardhat-etherscan');
require('hardhat-gas-reporter');
require('solidity-coverage');
require('hardhat-contract-sizer');
require('dotenv').config();

// Private keys for deployment
const DEPLOYER_PRIVATE_KEY = process.env.DEPLOYER_PRIVATE_KEY || '0x0000000000000000000000000000000000000000000000000000000000000000';
const TESTNET_PRIVATE_KEY = process.env.TESTNET_PRIVATE_KEY || DEPLOYER_PRIVATE_KEY;
const MAINNET_PRIVATE_KEY = process.env.MAINNET_PRIVATE_KEY || DEPLOYER_PRIVATE_KEY;

// Provider URLs
const LOCALHOST_URL = process.env.BLOCKCHAIN_URL || 'http://localhost:8545';
const TESTNET_URL = process.env.TESTNET_RPC_URL || 'https://eth-sepolia.g.alchemy.com/v2/your-api-key';
const MAINNET_URL = process.env.MAINNET_RPC_URL || 'https://eth-mainnet.g.alchemy.com/v2/your-api-key';

// Etherscan API key for contract verification
const ETHERSCAN_API_KEY = process.env.ETHERSCAN_API_KEY || '';

/**
 * @type import('hardhat/config').HardhatUserConfig
 */
module.exports = {
  solidity: {
    version: '0.8.17',
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    hardhat: {
      chainId: 1337,
      accounts: {
        mnemonic: "test test test test test test test test test test test junk",
        path: "m/44'/60'/0'/0",
        initialIndex: 0,
        count: 20
      }
    },
    localhost: {
      url: LOCALHOST_URL,
      accounts: [DEPLOYER_PRIVATE_KEY],
      chainId: 1337
    },
    testnet: {
      url: TESTNET_URL,
      accounts: [TESTNET_PRIVATE_KEY],
      chainId: 11155111, // Sepolia testnet
      gasMultiplier: 1.2,
      timeout: 60000
    },
    mainnet: {
      url: MAINNET_URL,
      accounts: [MAINNET_PRIVATE_KEY],
      chainId: 1,
      gasMultiplier: 1.1,
      timeout: 90000,
      gasPrice: 'auto'
    }
  },
  gasReporter: {
    enabled: process.env.REPORT_GAS === 'true',
    currency: 'USD',
    outputFile: 'gas-report.txt',
    noColors: process.env.CI === 'true',
    coinmarketcap: process.env.COINMARKETCAP_API_KEY || '',
    token: 'ETH',
    gasPriceApi: 'https://api.etherscan.io/api?module=proxy&action=eth_gasPrice',
    excludeContracts: [],
    src: './contracts'
  },
  etherscan: {
    apiKey: ETHERSCAN_API_KEY
  },
  contractSizer: {
    alphaSort: true,
    disambiguatePaths: false,
    runOnCompile: true,
    strict: true
  },
  mocha: {
    timeout: 120000
  },
  paths: {
    sources: './contracts',
    tests: './test',
    cache: './cache',
    artifacts: './artifacts'
  },
  // Solidity coverage configuration
  coverage: {
    provider: "hardhat",
    // Files to exclude from coverage report
    skipFiles: [
      "mocks/",
      "interfaces/"
    ],
    // Coverage threshold requirements
    istanbulReporter: ['html', 'lcov', 'text', 'json'],
    // Enforce coverage thresholds
    coverageThreshold: {
      global: {
        statements: 90,
        branches: 85,
        functions: 90,
        lines: 90
      }
    }
  }
};
