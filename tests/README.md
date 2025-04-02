# Testing Framework for Decentralized Federated Learning Platform

This directory contains comprehensive tests for the decentralized federated learning platform. The testing framework is designed to ensure the reliability, security, and correctness of all platform components.

## Test Structure

The testing framework is organized into several categories:

### Unit Tests

Unit tests focus on testing individual components in isolation:

- **Client Layer:**
  - `tests/unit/test_local_trainer.py` - Tests for the local training module
  - `tests/unit/test_privacy_utils.py` - Tests for differential privacy implementations

- **Aggregation Layer:**
  - `tests/unit/test_global_aggregator.py` - Tests for the global aggregation module
  - `tests/unit/test_quality_verification.py` - Tests for contribution quality verification

- **API Layer:**
  - `tests/unit/test_api_routes.py` - Tests for API endpoints
  - `tests/unit/test_client_library.py` - Tests for client library functions

### Integration Tests

Integration tests verify the interactions between multiple components:

- `tests/integration/test_api_middleware.py` - Tests for API and middleware integration
- `tests/integration/test_middleware_blockchain.py` - Tests for middleware and blockchain integration
- `tests/integration/test_client_api.py` - Tests for client and API interactions

### Contract Tests

Smart contract tests ensure the correctness of our blockchain components:

- `blockchain/test/ContributionLogging.test.js` - Tests for the contribution logging contract
- `blockchain/test/FedLearningToken.test.js` - Tests for the platform token contract
- `blockchain/test/QualityVerification.test.js` - Tests for quality verification contract
- `blockchain/test/RewardDistribution.test.js` - Tests for reward distribution contract
- `blockchain/test/FLGovernance.test.js` - Tests for governance mechanisms

### End-to-End Tests

End-to-end tests validate the complete platform workflow:

- `tests/e2e/test_training_workflow.py` - Tests the complete training and contribution cycle
- `tests/e2e/test_governance_workflow.py` - Tests governance proposals and voting

## Setting Up the Testing Environment

### Prerequisites

- Python 3.8+
- Node.js 16+
- Hardhat
- Ganache (for local blockchain testing)

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Install Node.js dependencies:
```bash
cd blockchain
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with appropriate values
```

## Running Tests

### Running Python Unit Tests

```bash
# Run all unit tests
python -m unittest discover -s tests/unit

# Run specific test file
python -m unittest tests/unit/test_local_trainer.py

# Run with coverage
coverage run -m unittest discover -s tests/unit
coverage report
coverage html  # Generates HTML report in htmlcov/
```

### Running Python Integration Tests

```bash
# Run all integration tests
python -m unittest discover -s tests/integration

# Run specific integration test
python -m unittest tests/integration/test_api_middleware.py
```

### Running Smart Contract Tests

```bash
cd blockchain

# Run all contract tests
npx hardhat test

# Run with coverage
npx hardhat coverage

# Run specific test file
npx hardhat test test/ContributionLogging.test.js
```

### Running End-to-End Tests

```bash
# Start local services first (in separate terminals)
# Terminal 1: Start local blockchain
npx hardhat node

# Terminal 2: Deploy contracts
npx hardhat run --network localhost scripts/deploy.js

# Terminal 3: Start API server
python middleware/api_server.py

# Terminal 4: Run tests
python -m unittest discover -s tests/e2e
```

## Continuous Integration

This project uses GitHub Actions for continuous integration. The workflow includes:

1. Running unit and integration tests for Python components
2. Running smart contract tests with coverage reports
3. Generating and uploading test coverage reports
4. Running linting and static analysis

The CI pipeline is configured in `.github/workflows/ci.yml`.

## Test Coverage Requirements

We maintain high test coverage standards:

- Python code: Minimum 80% line coverage
- Smart contracts: Minimum 90% statement coverage, 85% branch coverage

## Adding New Tests

When adding new functionality, corresponding tests should be added to maintain coverage:

1. For new Python modules, add unit tests in the appropriate directory under `tests/unit/`
2. For new smart contracts, add tests in the `blockchain/test/` directory
3. For new integrations between components, add tests in `tests/integration/`

## Troubleshooting Common Issues

- **Port conflicts:** Ensure services are running on the correct ports without conflicts
- **Contract deployment failures:** Check that Hardhat node is running before deploying contracts
- **Database connection issues:** Verify that any required databases are running and accessible
- **API authentication failures:** Check that JWT secret keys are properly configured

## Contributing

When contributing to the testing framework, please follow these guidelines:

1. Write tests before implementing functionality (Test-Driven Development)
2. Ensure all tests pass before submitting a pull request
3. Maintain or improve current test coverage percentages
4. Document any new testing patterns or frameworks added

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Hardhat Testing Guide](https://hardhat.org/hardhat-runner/docs/guides/test-contracts)
- [Solidity Coverage](https://github.com/sc-forks/solidity-coverage)
- [Blockchain Testing Best Practices](https://ethereum.org/en/developers/docs/smart-contracts/testing/)
