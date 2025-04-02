#!/bin/bash
# Database initialization script for Federated Learning Platform

set -e  # Exit immediately if a command exits with a non-zero status

echo "Initializing Federated Learning Platform database..."

# Wait for the database to be ready
if [ -n "$POSTGRES_HOST" ]; then
    echo "Waiting for PostgreSQL to be ready at $POSTGRES_HOST:$POSTGRES_PORT..."
    until PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d postgres -c '\q'; do
        echo "PostgreSQL is unavailable - sleeping"
        sleep 1
    done
    echo "PostgreSQL is up and running!"
fi

# Create database if it doesn't exist
PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DB'" | grep -q 1 || \
    PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d postgres -c "CREATE DATABASE $POSTGRES_DB"

echo "Running schema creation script..."
PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -f /docker-entrypoint-initdb.d/01_schema.sql

echo "Running seed data script..."
PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -f /docker-entrypoint-initdb.d/02_seed_data.sql

echo "Database initialization completed!" 