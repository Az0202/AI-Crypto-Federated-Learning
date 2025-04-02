# Database Layer

This directory contains database initialization and migration scripts for the Federated Learning Platform.

## Directory Structure

- `init/`: Database initialization scripts
  - `00_init_db.sh`: Shell script to initialize the database
  - `01_schema.sql`: DDL statements to create database tables
  - `02_seed_data.sql`: Initial data for database tables

## Database Schema

The platform uses PostgreSQL and includes the following key tables:

- `users`: Store client/participant information
- `models`: Track global AI models
- `model_versions`: Track different versions of models
- `training_rounds`: Track federated learning rounds
- `model_contributions`: Track client contributions to model training
- `governance_proposals`: Store DAO governance proposals
- `governance_votes`: Store votes on governance proposals
- `system_events`: System event log for auditing

## Usage

### Local Development

1. Start a PostgreSQL database:
   ```bash
   docker run --name postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres
   ```

2. Run the initialization scripts:
   ```bash
   export POSTGRES_HOST=localhost
   export POSTGRES_PORT=5432
   export POSTGRES_USER=postgres
   export POSTGRES_PASSWORD=password
   export POSTGRES_DB=federated_learning
   
   cd db/init
   chmod +x 00_init_db.sh
   ./00_init_db.sh
   ```

### Docker Compose

When using Docker Compose, the initialization scripts are automatically run when the database container starts. The scripts are mounted to the `/docker-entrypoint-initdb.d/` directory in the container.

## Migrations

Database migrations should be placed in a `migrations/` directory (to be implemented). Consider using a migration tool like Alembic or Flyway for managing database schema changes.

## Backup and Restore

It's recommended to set up regular database backups. Here's a simple example:

```bash
# Backup
pg_dump -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -F c -b -v -f backup.dump

# Restore
pg_restore -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -v backup.dump
``` 