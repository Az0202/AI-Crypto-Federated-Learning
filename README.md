# AI-Crypto-Federated-Learning
# Decentralized Federated Learning Platform

A privacy-preserving AI training ecosystem powered by blockchain technology that enables multiple organizations to collaboratively train AI models without sharing raw data.

## Features

- **Privacy-Preserving Training**: Local training with differential privacy
- **Blockchain Integration**: Immutable record of contributions with tokenized incentives
- **Decentralized Governance**: Community-driven platform evolution via DAO
- **Transparent Rewards**: Fair compensation based on contribution quality and impact
- **Memory-Efficient Processing**: Optimized for large neural network models
- **Interactive Dashboard**: Real-time monitoring of system performance and token economics

## System Architecture

The platform consists of five key layers:

1. **Client/Edge Layer**: Local training with privacy measures
2. **Aggregation Layer**: Secure model update aggregation
3. **Blockchain Layer**: Smart contracts for contribution tracking and rewards
4. **Governance Module**: Decentralized decision-making
5. **API/Middleware Layer**: Integration and orchestration services
6. **Dashboard**: Real-time visualization of system metrics and performance

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js 16+ (for local development)
- Python 3.9+ (for local development)
- MetaMask or another Ethereum wallet

### Quick Start with Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/Az0202/AI-Crypto-Federated-Learning.git
   cd AI-Crypto-Federated-Learning
   ```

2. Create environment file:
   ```bash
   cp .env.template .env
   # Edit .env with your secure configuration
   # IMPORTANT: Generate new private keys and strong passwords
   # NEVER commit this .env file to version control
   ```

3. Start the platform:
   ```bash
   docker-compose up -d
   ```

4. Monitor the deployment:
   ```bash
   docker-compose logs -f
   ```

5. Access services:
   - Dashboard: http://localhost:8501
   - API: http://localhost:8000
   - Blockchain node: http://localhost:8545

### Dashboard Features

The dashboard provides a real-time view of:
- Platform Overview: Active clients, total contributions, and current training round
- Model Performance: Accuracy, loss, and other metrics over time
- Contributions: Detailed view of client contributions
- Token Economics: Reward distribution and token balances
- Governance: Proposal viewing and voting (requires authentication)

**Note**: To view your token balance, you need to authenticate by entering your JWT token in the dashboard sidebar.

### Client Setup

To participate as a training client:

1. Install the client library:
   ```bash
   pip install federated-learning-client
   ```

2. Import and use in your Python code:
   ```python
   from fed_learning_client import FederatedLearningClient
   
   # Initialize client
   client = FederatedLearningClient(
       api_url="http://localhost:8000",
       client_id="your_client_id"
   )
   
   # Load Ethereum wallet
   client.load_ethereum_wallet("your_private_key")
   
   # Authenticate
   await client.authenticate()
   
   # Download global model
   global_weights, model_info = await client.get_global_model()
   
   # Local training (implement your training logic)
   # ...
   
   # Submit contribution
   result = await client.submit_contribution(model_update, metrics)
   ```

## Docker Container Architecture

The platform runs as a set of interconnected Docker containers:

- **blockchain**: Local Ethereum node running Ganache for development
- **contract-deployer**: Deploys smart contracts to the blockchain
- **redis**: Handles caching and rate limiting
- **postgres**: Persistent database storage
- **api**: FastAPI server exposing the federated learning endpoints
- **aggregator**: Service that processes and aggregates model updates
- **dashboard**: Streamlit dashboard for monitoring and visualization

The services are designed to work together, with proper health checks and dependencies to ensure orderly startup. Docker Compose manages the deployment and ensures services can communicate through the internal Docker network.

## Deployment Options

### Local Development

1. Start local blockchain:
   ```bash
   cd blockchain
   npm install
   npx hardhat node
   ```

2. Deploy contracts:
   ```bash
   # In a new terminal
   cd blockchain
   npx hardhat run scripts/deploy.js --network localhost
   ```

3. Start API server:
   ```bash
   # In a new terminal
   cd middleware
   poetry install
   python -m uvicorn api.server:app --reload
   ```

4. Start aggregator service:
   ```bash
   # In a new terminal
   cd aggregator
   poetry install
   python -m aggregator.service
   ```

5. Start dashboard:
   ```bash
   # In a new terminal
   cd visualization
   poetry install
   streamlit run dashboard.py
   ```

### Production Deployment

For production deployment, follow these steps:

1. Configure secure environment variables:
   - Generate new private keys for all system accounts
   - Set strong JWT secret and passwords
   - Configure proper rate limiting settings

2. Update Docker Compose configuration:
   ```bash
   # Use production environment
   cp docker-compose.yml docker-compose.prod.yml
   # Edit docker-compose.prod.yml to use production settings
   ```

3. Deploy to your infrastructure:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

4. Set up monitoring and backups:
   ```bash
   # Configure your preferred monitoring solution
   # Set up database backups
   ```

5. Deploy smart contracts to mainnet:
   ```bash
   cd blockchain
   ENVIRONMENT=production npm run deploy:mainnet
   ```

## Use Cases

### Healthcare Diagnostics

Hospitals and research centers can collaboratively train diagnostic models while keeping sensitive patient data local and secure.

### Financial Fraud Detection

Banks and financial institutions improve fraud detection and risk assessment models without exposing proprietary data.

### IoT/Smart Cities

Distributed IoT sensors contribute to predictive maintenance, traffic optimization, or environmental monitoring models while maintaining data privacy.

## Development

### Project Structure

```
decentralized-federated-learning/
├── .env.template                       # Environment variables template
├── docker-compose.yml                  # Main Docker Compose configuration
├── README.md                           # Project documentation
├── blockchain_config.json              # Blockchain configuration
│
├── client/                             # Client/Edge Layer
│   ├── local_trainer.py                # Local model training module
│   ├── fed_learning_client.py          # Client library
│   ├── privacy_utils.py                # Differential privacy implementation
│   └── client_config.yaml              # Client configuration
│
├── aggregator/                         # Aggregation Layer
│   ├── Dockerfile                      # Aggregator container definition
│   ├── pyproject.toml                  # Poetry dependencies for aggregator
│   ├── poetry.lock                     # Poetry lock file
│   ├── global_aggregator.py            # Global model aggregation
│   ├── service.py                      # Main aggregator service
│   ├── optimized/
│   │   └── streaming_aggregator.py     # Optimized aggregation algorithms
│   └── quality_verification.py         # Contribution quality checks
│
├── blockchain/                         # Blockchain Layer
│   ├── Dockerfile                      # Blockchain container definition
│   ├── package.json                    # Node.js dependencies
│   ├── hardhat.config.js               # Hardhat configuration
│   ├── truffle-config.js               # Truffle configuration
│   ├── contracts/                      # Smart contracts
│   │   ├── ContributionLogging.sol     # Contribution logging contract
│   │   ├── FedLearningToken.sol        # Token contract
│   │   ├── QualityVerification.sol     # Quality verification contract
│   │   ├── RewardDistribution.sol      # Reward distribution contract
│   │   └── FLGovernance.sol            # Governance contract
│   ├── scripts/                        # Deployment scripts
│   │   └── deploy.js                   # Main deployment script
│   ├── test/                           # Contract tests
│   │   ├── ContributionLogging.test.js # Contract unit tests
│   │   ├── FedLearningToken.test.js
│   │   ├── QualityVerification.test.js
│   │   ├── RewardDistribution.test.js
│   │   └── FLGovernance.test.js
│   └── deployed/                       # Deployed contract information
│
├── middleware/                         # API/Middleware Layer
│   ├── Dockerfile                      # Middleware container definition
│   ├── pyproject.toml                  # Poetry dependencies for middleware
│   ├── api/                            # API endpoints
│   │   └── api_server.py               # Main FastAPI application
│   ├── auth/                           # Authentication
│   │   └── signature_verifier.py       # Ethereum signature verification
│   ├── blockchain/                     # Blockchain integration
│   │   ├── blockchain_middleware.py    # Blockchain interaction
│   │   └── transaction_manager.py      # Transaction management
│   ├── contracts/                      # Contract integration modules
│   ├── models/                         # Data models
│   └── security/                       # Security features
│       └── rate_limiter.py             # Rate limiting middleware
│
├── visualization/                      # Dashboard
│   ├── Dockerfile                      # Dashboard container definition
│   ├── pyproject.toml                  # Poetry dependencies for dashboard
│   ├── poetry.lock                     # Poetry lock file
│   ├── .env.template                   # Environment template for visualization
│   └── dashboard.py                    # Streamlit dashboard application
│
├── utils/                              # Utility functions
│   ├── crypto.py                       # Cryptographic utilities
│   ├── data_handling.py                # Data processing utilities
│   ├── encoding/
│   │   └── model_data_handler.py       # Data encoding module
│   └── logging.py                      # Logging utilities
│
├── models/                             # Model definitions
│   ├── base_model.py                   # Base model structure
│   └── model_registry.py               # Model versioning
│
├── tests/                              # Tests
│   ├── README.md                       # Testing documentation
│   ├── unit/                           # Unit tests
│   │   ├── test_local_trainer.py       # Tests for LocalTrainer
│   │   ├── test_global_aggregator.py   # Tests for GlobalAggregator
│   │   ├── test_fed_learning_client.py # Tests for client library
│   │   └── ...                         # Other unit tests
│   ├── integration/                    # Integration tests
│   │   ├── test_api_middleware.py      # API-Middleware integration
│   │   ├── test_middleware_blockchain.py # Middleware-Blockchain integration
│   │   └── ...                         # Other integration tests
│   └── e2e/                            # End-to-end tests
│       ├── test_training_workflow.py   # Training workflow tests
│       └── test_governance_workflow.py # Governance workflow tests
│
├── db/                                 # Database
│   └── init/                           # Database initialization scripts
│
├── scripts/                            # Utility scripts
│   └── prepare_for_public.sh           # Script to sanitize project for public sharing
│
└── .dist/                              # Distribution files
```

### Running Tests

Execute the test suite:

```bash
# Unit tests
python -m unittest discover -s tests/unit

# Integration tests
python -m unittest discover -s tests/integration

# Smart contract tests
cd blockchain
npx hardhat test
```

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Security Considerations


Key security features:
- Enhanced Ethereum signature verification
- Rate limiting for API protection
- Smart contract security audits
- Differential privacy for local training
- Memory-efficient handling of sensitive data

### Repository Security

To maintain security when using this codebase, certain files are intentionally excluded from version control:

1. **Environment Files**: All `.env` files containing secrets (except the template)
2. **Private Keys**: Any files with private keys, including Ethereum wallet keys
3. **Authentication Secrets**: JWT secrets and API keys
4. **Deployment Configurations**: Production deployment configurations with real endpoints
5. **Model Files**: Trained model weights that may contain sensitive information
6. **Database Data**: Local database files and data directories

Before deployment to production:
- Generate new secure private keys for all system accounts
- Set strong JWT secrets
- Update API endpoints to use secure URLs
- Configure proper rate limiting based on your expected traffic

**Never commit sensitive information to public repositories.** If you accidentally commit sensitive data, consider it compromised and regenerate all affected credentials immediately.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TensorFlow Privacy](https://github.com/tensorflow/privacy) for differential privacy implementations
- [OpenZeppelin](https://github.com/OpenZeppelin/openzeppelin-contracts) for secure smart contract components
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Streamlit](https://streamlit.io/) for the dashboard framework

---

## Troubleshooting

### Common Issues

#### Dashboard Cannot Connect to API

If you see "Connection error" or "No platform statistics available" in the dashboard:

1. Verify that the API service is running:
   ```bash
   docker-compose ps api
   ```

2. Check API health endpoint:
   ```bash
   curl http://localhost:8000/api/health
   ```

3. Ensure the dashboard is using the correct API URL. In Docker environment, services communicate using service names (e.g., `http://api:8000`), not localhost.

#### Token Balance Not Showing

1. The token balance feature requires authentication. Enter a valid JWT token in the dashboard sidebar.
2. If you're testing with an empty auth token, the dashboard will gracefully display "Login to view" instead of your balance.

#### Smart Contract Deployment Issues

1. Check if the blockchain container is healthy:
   ```bash
   docker-compose ps blockchain
   ```

2. Verify the deployer has enough ETH:
   ```bash
   docker-compose logs contract-deployer
   ```

3. If necessary, rebuild the contract-deployer:
   ```bash
   docker-compose build contract-deployer
   docker-compose up -d contract-deployer
   ```

### Restarting Services

To restart individual services after configuration changes:

```bash
docker-compose restart [service_name]
```

For example, to restart the dashboard after changes:

```bash
docker-compose restart dashboard
```

## Recent Updates

### Dashboard Improvements (July 2024)

- **Fixed API Connectivity**: Resolved communication issues between the dashboard and API containers in Docker environment
- **Enhanced Error Handling**: Added better error reporting and graceful degradation when services are unavailable
- **Improved Token Balance Display**: Dashboard now handles authentication gracefully, showing helpful messages when not authenticated
- **Internal Service Communication**: Fixed Docker network service name resolution for stable inter-service communication
- **Data Visualization**: Extended charts and visualizations for better platform monitoring
- **Configuration Settings**: Added default settings suitable for Docker environment

### Smart Contract Deployment Enhancements

- **Deployment Reliability**: Added health checks to ensure blockchain service is ready before contract deployment
- **Account Funding**: Updated account configuration to use Ganache's default accounts with sufficient ETH
- **Dependency Management**: Improved container dependencies to ensure ordered service startup
- **Contract ABIs**: Enhanced handling of contract ABIs for middleware integration
