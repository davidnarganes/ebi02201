# Define hyperparameter options
GRID_SEARCH = {
    'MODEL_NAME': ['distilbert-base-uncased', 'dmis-lab/TinyPubMedBERT-v1.0', 'dmis-lab/biobert-base-cased-v1.2'],
    'LEARNING_RATE': [2e-05, 5e-05],
    'DROPOUT_RATE': [0, 0.1],
    'LOSS_TYPE': ['bce', 'focal', 'smooth_focal'],
    'BATCH_SIZE': [16, 32],
    'MAX_SEQ_LENGTH': [128, 256]
}