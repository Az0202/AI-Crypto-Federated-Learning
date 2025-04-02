FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies directly with pip
RUN pip install --no-cache-dir \
    streamlit==1.20.0 \
    plotly==5.13.0 \
    pandas==1.5.3 \
    numpy==1.23.5 \
    aiohttp==3.8.4 \
    requests \
    python-dotenv \
    pydantic \
    web3

# Copy application code
COPY visualization /app/

# Set Streamlit configuration
RUN mkdir -p /root/.streamlit
RUN echo '\
[server]\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
' > /root/.streamlit/config.toml

# Add wait script
ADD https://github.com/ufoscout/docker-compose-wait/releases/download/2.9.0/wait /wait
RUN chmod +x /wait

# Create startup script
RUN echo '#!/bin/bash\n\
/wait\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
