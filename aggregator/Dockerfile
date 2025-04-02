FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    netcat-openbsd \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$PATH:/root/.local/bin"

# Copy Poetry configuration
COPY aggregator/pyproject.toml aggregator/poetry.lock* /app/

# Configure Poetry to not use virtual environments in Docker
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Install TensorFlow and other ML dependencies
RUN pip install --no-cache-dir tensorflow==2.10.0 tensorflow-privacy==0.8.0 psutil==5.9.0

# Copy application code
COPY aggregator /app/
COPY utils /app/utils/

# Create models directory
RUN mkdir -p /app/models

# Add wait script for dependency readiness
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.9.0/wait /wait
RUN chmod +x /wait

# Create startup script
RUN echo '#!/bin/bash\n\
/wait\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "aggregator.service"]
