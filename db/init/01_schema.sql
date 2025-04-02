-- Database schema for the Federated Learning Platform
-- This script creates the necessary tables and relationships

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table to store client information
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ethereum_address VARCHAR(42) NOT NULL UNIQUE,
    username VARCHAR(100) UNIQUE,
    email VARCHAR(255) UNIQUE,
    reputation_score NUMERIC(10, 2) DEFAULT 0,
    total_contributions INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Models table to track global models
CREATE TABLE IF NOT EXISTS models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    current_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    architecture_json JSONB,
    initial_parameters_hash VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(user_id)
);

-- Model versions table to track different versions of models
CREATE TABLE IF NOT EXISTS model_versions (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    version VARCHAR(20) NOT NULL,
    parameters_hash VARCHAR(64) NOT NULL,
    parameters_location VARCHAR(255) NOT NULL,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, version)
);

-- Training rounds to track federated learning rounds
CREATE TABLE IF NOT EXISTS training_rounds (
    round_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active', -- active, completed, failed
    target_num_contributions INTEGER DEFAULT 10,
    actual_num_contributions INTEGER DEFAULT 0,
    aggregation_algorithm VARCHAR(50) DEFAULT 'fedavg',
    aggregation_parameters JSONB,
    result_model_version_id UUID REFERENCES model_versions(version_id),
    UNIQUE(model_id, round_number)
);

-- Model contributions to track client contributions
CREATE TABLE IF NOT EXISTS model_contributions (
    contribution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_id UUID NOT NULL REFERENCES training_rounds(round_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id),
    training_samples INTEGER NOT NULL,
    loss NUMERIC(10, 6),
    metrics JSONB,
    contribution_hash VARCHAR(64) NOT NULL,
    weight_update_location VARCHAR(255) NOT NULL,
    verification_status VARCHAR(20) DEFAULT 'pending', -- pending, verified, rejected
    reward_amount NUMERIC(18, 8) DEFAULT 0,
    transaction_hash VARCHAR(66),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(round_id, user_id)
);

-- Governance proposals for platform DAO
CREATE TABLE IF NOT EXISTS governance_proposals (
    proposal_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    proposer_id UUID NOT NULL REFERENCES users(user_id),
    proposal_type VARCHAR(50) NOT NULL, -- parameter_change, feature_request, etc.
    parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    voting_start_time TIMESTAMP WITH TIME ZONE,
    voting_end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'draft' -- draft, active, passed, rejected, implemented
);

-- Votes on governance proposals
CREATE TABLE IF NOT EXISTS governance_votes (
    vote_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposal_id UUID NOT NULL REFERENCES governance_proposals(proposal_id) ON DELETE CASCADE,
    voter_id UUID NOT NULL REFERENCES users(user_id),
    vote BOOLEAN NOT NULL, -- true for yes, false for no
    voting_power NUMERIC(18, 8) NOT NULL,
    transaction_hash VARCHAR(66),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(proposal_id, voter_id)
);

-- System events for audit trail
CREATE TABLE IF NOT EXISTS system_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL, -- user, model, round, etc.
    entity_id UUID NOT NULL,
    data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id UUID REFERENCES users(user_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_model_contributions_round_id ON model_contributions(round_id);
CREATE INDEX IF NOT EXISTS idx_model_contributions_user_id ON model_contributions(user_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_model_id ON model_versions(model_id);
CREATE INDEX IF NOT EXISTS idx_training_rounds_model_id ON training_rounds(model_id);
CREATE INDEX IF NOT EXISTS idx_governance_votes_proposal_id ON governance_votes(proposal_id);
CREATE INDEX IF NOT EXISTS idx_system_events_entity_id ON system_events(entity_id);
CREATE INDEX IF NOT EXISTS idx_system_events_event_type ON system_events(event_type); 