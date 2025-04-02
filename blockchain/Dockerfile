FROM node:16-alpine

WORKDIR /app

# Install system dependencies
RUN apk add --no-cache git bash python3 make g++

# Copy package files
COPY blockchain/package*.json ./

# Install dependencies
RUN npm install

# Copy contract files
COPY blockchain/contracts ./contracts
COPY blockchain/migrations ./migrations
COPY blockchain/scripts ./scripts
COPY blockchain/test ./test
COPY blockchain/truffle-config.js ./
COPY blockchain/hardhat.config.js ./

# Create directory for deployed contract information
RUN mkdir -p deployed

# Create deployment script
RUN echo '#!/bin/sh\n\
echo "Waiting for blockchain to be ready..."\n\
sleep 5\n\
if [ "$ENVIRONMENT" = "production" ]; then\n\
  echo "Deploying to production network..."\n\
  npx hardhat run scripts/deploy.js --network mainnet\n\
elif [ "$ENVIRONMENT" = "testnet" ]; then\n\
  echo "Deploying to testnet..."\n\
  npx hardhat run scripts/deploy.js --network testnet\n\
else\n\
  echo "Deploying to local development network..."\n\
  npx hardhat run scripts/deploy.js --network localhost\n\
fi\n\
\n\
# Copy deployment artifacts to shared volume\n\
cp -r artifacts/contracts/* deployed/\n\
echo "Deployment complete!"' > /app/deploy.sh

RUN chmod +x /app/deploy.sh

CMD ["/app/deploy.sh"]
