-- Seed data for the Federated Learning Platform
-- This script inserts initial data required for the platform to function

-- Create an admin user
INSERT INTO users (
    user_id,
    ethereum_address,
    username,
    email,
    reputation_score,
    is_active
) VALUES (
    '00000000-0000-0000-0000-000000000001',
    '0x0000000000000000000000000000000000000001',
    'admin',
    'admin@federated-learning.org',
    100.0,
    TRUE
) ON CONFLICT (ethereum_address) DO NOTHING;

-- Create a system user for automated operations
INSERT INTO users (
    user_id,
    ethereum_address,
    username,
    email,
    reputation_score,
    is_active
) VALUES (
    '00000000-0000-0000-0000-000000000002',
    '0x0000000000000000000000000000000000000002',
    'system',
    'system@federated-learning.org',
    100.0,
    TRUE
) ON CONFLICT (ethereum_address) DO NOTHING;

-- Create a test user for development
INSERT INTO users (
    user_id,
    ethereum_address,
    username,
    email,
    reputation_score,
    is_active
) VALUES (
    '00000000-0000-0000-0000-000000000003',
    '0x0000000000000000000000000000000000000003',
    'test_user',
    'test@federated-learning.org',
    50.0,
    TRUE
) ON CONFLICT (ethereum_address) DO NOTHING;

-- Create an initial sample model for image classification
INSERT INTO models (
    model_id,
    name,
    description,
    current_version,
    architecture_json,
    created_by
) VALUES (
    '00000000-0000-0000-0000-000000000001',
    'ImageClassifier-MNIST',
    'Simple convolutional neural network for MNIST digit classification',
    '1.0.0',
    '{
        "model_type": "cnn",
        "input_shape": [28, 28, 1],
        "layers": [
            {"type": "Conv2D", "filters": 32, "kernel_size": 3, "activation": "relu"},
            {"type": "MaxPooling2D", "pool_size": 2},
            {"type": "Conv2D", "filters": 64, "kernel_size": 3, "activation": "relu"},
            {"type": "MaxPooling2D", "pool_size": 2},
            {"type": "Flatten"},
            {"type": "Dense", "units": 128, "activation": "relu"},
            {"type": "Dense", "units": 10, "activation": "softmax"}
        ],
        "loss": "categorical_crossentropy",
        "optimizer": "adam",
        "metrics": ["accuracy"]
    }',
    '00000000-0000-0000-0000-000000000001'
) ON CONFLICT DO NOTHING;

-- Create a model version for the initial model
INSERT INTO model_versions (
    version_id,
    model_id,
    version,
    parameters_hash,
    parameters_location,
    metrics
) VALUES (
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001',
    '1.0.0',
    'init_random_seed_42_hash',
    'model_storage/00000000-0000-0000-0000-000000000001_1.0.0_weights.pkl',
    '{
        "accuracy": 0.10,
        "loss": 2.3
    }'
) ON CONFLICT DO NOTHING;

-- Create an initial training round for the model
INSERT INTO training_rounds (
    round_id,
    model_id,
    round_number,
    status,
    target_num_contributions,
    aggregation_algorithm,
    aggregation_parameters
) VALUES (
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001',
    1,
    'active',
    5,
    'fedavg',
    '{
        "min_contributions": 3,
        "contribution_weight_cap": 1000,
        "discard_outlier_threshold": 2.0
    }'
) ON CONFLICT DO NOTHING;

-- Create an initial tabular data model for healthcare
INSERT INTO models (
    model_id,
    name,
    description,
    current_version,
    architecture_json,
    created_by
) VALUES (
    '00000000-0000-0000-0000-000000000002',
    'DiabetesPrediction',
    'Random forest model for diabetes prediction based on patient data',
    '1.0.0',
    '{
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "features": [
            "age", "gender", "bmi", "blood_pressure", "insulin", "glucose"
        ],
        "target": "diabetes"
    }',
    '00000000-0000-0000-0000-000000000001'
) ON CONFLICT DO NOTHING;

-- Create a model version for the healthcare model
INSERT INTO model_versions (
    version_id,
    model_id,
    version,
    parameters_hash,
    parameters_location,
    metrics
) VALUES (
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000002',
    '1.0.0',
    'init_random_healthcare_hash',
    'model_storage/00000000-0000-0000-0000-000000000002_1.0.0_weights.pkl',
    '{
        "accuracy": 0.72,
        "precision": 0.68,
        "recall": 0.75,
        "f1_score": 0.71
    }'
) ON CONFLICT DO NOTHING;

-- Create an initial governance proposal
INSERT INTO governance_proposals (
    proposal_id,
    title,
    description,
    proposer_id,
    proposal_type,
    parameters,
    status
) VALUES (
    '00000000-0000-0000-0000-000000000001',
    'Update Minimum Contribution Requirements',
    'Proposal to update the minimum number of training samples required for a valid contribution',
    '00000000-0000-0000-0000-000000000001',
    'parameter_change',
    '{
        "parameter": "min_contribution_samples",
        "current_value": 500,
        "proposed_value": 1000,
        "reason": "Increase data quality and model performance"
    }',
    'active'
) ON CONFLICT DO NOTHING;

-- Record system initialization event
INSERT INTO system_events (
    event_id,
    event_type,
    entity_type,
    entity_id,
    data,
    user_id
) VALUES (
    '00000000-0000-0000-0000-000000000001',
    'system_initialization',
    'system',
    '00000000-0000-0000-0000-000000000000',
    '{
        "initialization_time": CURRENT_TIMESTAMP,
        "database_version": "1.0.0",
        "schema_hash": "initial"
    }',
    '00000000-0000-0000-0000-000000000002'
) ON CONFLICT DO NOTHING; 