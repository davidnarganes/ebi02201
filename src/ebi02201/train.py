import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AdamW
import logging
from tqdm import tqdm
from myModel import EntityModel
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import collections
from gridsearch import GRID_SEARCH
import itertools
import mlflow
from mlflow import log_params

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'attention_mask': self.data[idx]['attention_mask'],
            'labels': self.data[idx]['labels']
        }
    
def calculate_metrics(y_true, y_pred, epsilon=1e-9):
    """
    Calculate true positives, false positives, false negatives, and true negatives for multi-class classification,
    along with F1 score, precision, recall, and MCC for each class.
    
    Args:
        y_true (np.ndarray): Ground truth labels (shape: batch_size x num_tokens x num_classes).
        y_pred (np.ndarray): Predicted labels (shape: batch_size x num_tokens x num_classes).
    
    Returns:
        metrics_dict (dict): A dictionary containing TP, FP, FN, TN, F1 Score, Precision, Recall, and MCC for each class.
    """
    num_classes = y_true.shape[-1]
    metrics_dict = {}
    
    for i in range(num_classes):
        y_true_class = y_true[:, :, i]
        y_pred_class = y_pred[:, :, i]
        
        TP = np.sum((y_true_class == 1) & (y_pred_class == 1))
        FP = np.sum((y_true_class == 0) & (y_pred_class == 1))
        FN = np.sum((y_true_class == 1) & (y_pred_class == 0))
        TN = np.sum((y_true_class == 0) & (y_pred_class == 0))
        
        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        mcc = (TP * TN - FP * FN) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + epsilon)
        
        metrics_dict[f"Class_{i+1}"] = {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'MCC': mcc
        }
    
    return metrics_dict

def get_tokens_and_labels(texts, annotations, tokenizer, max_seq_length=128, annotation_types=None, label_to_id=None):
    """
    Tokenizes input texts and generates labels for entity recognition.

    Args:
        texts (list): List of input texts.
        annotations (list): List of annotations for each text.
        tokenizer: Tokenizer for text tokenization.
        max_seq_length (int, optional): Maximum sequence length for tokenized texts. Defaults to 128.
        annotation_types (list, optional): List of entity annotation types. Defaults to None.
        label_to_id (dict, optional): Mapping from entity types to label ids. Defaults to None.

    Returns:
        list: List of dictionaries containing tokenized input data and labels.
    """
    data = []

    for i, text in enumerate(texts):
        # Tokenize the text
        tokens = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            padding='max_length',
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt"
        )

        # Initialize label tensor
        labels = torch.zeros(size=(max_seq_length, len(annotation_types)))

        for annotation in annotations[i]:
            start, end, _, entity_type = annotation
            m1 = tokens['offset_mapping'][0][:, 0] >= start
            m2 = tokens['offset_mapping'][0][:, 1] <= end
            labels[m1 & m2, label_to_id[entity_type]] = 1

        data.append({
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': labels
        })

    return data
    
def evaluate(loaders, model):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model.eval()
    val_losses = {}

    for key in ['val', 'test']:
        dataloader = loaders[key]
        val_loss = 0
        total_samples = 0
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for val_batch in tqdm(dataloader, desc=f"Evaluating {key}", leave=False):
                val_probabilities, val_loss_batch = model(val_batch)
                batch_size = val_batch['input_ids'].size(0)
                val_loss += val_loss_batch.item() * batch_size
                total_samples += batch_size

                y_true = val_batch['labels'].cpu().numpy()
                y_pred = (val_probabilities > 0.4).cpu().numpy()  # Adjust threshold as needed
                all_y_true.append(y_true)
                all_y_pred.append(y_pred)

        val_loss /= total_samples
        val_losses[key] = val_loss

        y_true = np.concatenate(all_y_true, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)
        metrics = calculate_metrics(y_true, y_pred)

        logger.info(f'Evaluating {key}:')
        for label, metrics_dict in metrics.items():
            f1 = metrics_dict["F1 Score"]
            precision = metrics_dict["Precision"]
            recall = metrics_dict["Recall"]
            mcc = metrics_dict["MCC"]
            fp = metrics_dict["FP"]
            fn = metrics_dict["FN"]
            tp = metrics_dict["TP"]
            tn = metrics_dict["TN"]

            logger.info(f'Class: {label}, TP: {tp}, FP: {fp}, '
                        f'FN: {fn}, TN: {tn}, '
                        f'F1: {f1:.2f}, Precision: {precision:.2f}, '
                        f'Recall: {recall:.2f}, MCC: {mcc:.2f}')

            # Log metrics to MLflow
            mlflow.log_metric(f'{key}_{label}_F1', f1)
            mlflow.log_metric(f'{key}_{label}_Precision', precision)
            mlflow.log_metric(f'{key}_{label}_Recall', recall)
            mlflow.log_metric(f'{key}_{label}_MCC', mcc)
            mlflow.log_metric(f'{key}_{label}_FP', fp)
            mlflow.log_metric(f'{key}_{label}_FN', fn)
            mlflow.log_metric(f'{key}_{label}_TP', tp)
            mlflow.log_metric(f'{key}_{label}_TN', tn)

    return val_losses

def train_and_evaluate(loaders, model, optimizer, num_epochs, log_interval=20, val_interval=100):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(loaders['train'], desc=f"Epoch {epoch + 1}/{num_epochs}")):
            optimizer.zero_grad()
            probabilities, loss = model(batch)
            loss.backward()
            optimizer.step()

            batch_size = batch['input_ids'].size(0)
            train_loss += loss.item() * batch_size
            total_samples += batch_size

            if batch_idx % log_interval == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
                # TODO: I should maybe log something here to MLFlow but it will cause issues with the eval()

            # Perform validation after 'val_interval' batches
            if batch_idx % val_interval == 0 and batch_idx > 0:
                val_losses = evaluate(loaders, model)
                for key, val_loss in val_losses.items():
                    logger.info(f'Intermediate Validation Loss ({key}, Epoch {epoch}, Batch {batch_idx}): {val_loss:.4f}')

        train_loss /= total_samples
        logger.info(f'Training Loss (Epoch {epoch}): {train_loss:.4f}')

        # Perform validation at the end of the epoch
        val_losses = evaluate(loaders, model)
        for key, val_loss in val_losses.items():
            logger.info(f'Validation Loss ({key}, Epoch {epoch}): {val_loss:.4f}')

    logger.info("Training finished.")


if __name__ == '__main__':

    # Configure the logging settings
    logging.basicConfig(level=logging.INFO)

    # Other variables
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    SEED = 123
    RS = np.random.RandomState(SEED)
    SETS = ['train', 'val', 'test'] 

    # Read the CSV file with UTF-8 encoding
    file_path = '/Users/dnarganes/repos/exscientia/ebi02201/data/raw/biggest_test.csv'
    df = pd.read_csv(file_path, encoding='utf-8')
    df['ner'] = df['ner'].apply(eval)

    # Add here the splits for training, testing and validation
    df['set'] = RS.choice(
        SETS,
        replace=True,
        size=df.shape[0],
        p=(0.7, 0.15, 0.15)
        )
    
    # Find the five PMC IDs with the least sentences
    pmc_id_counts = df['pmc_id'].value_counts()
    pmc_ids_with_least_sentences = pmc_id_counts.nsmallest(5).index.tolist()
    # Set the 'set' column to 'test' for these PMC IDs
    df.loc[df['pmc_id'].isin(pmc_ids_with_least_sentences), 'set'] = 'test'

    # Define hyperparameter options
    GRID_SEARCH = {
        'MODEL_NAME': ['distilbert-base-uncased',
                    #    'dmis-lab/TinyPubMedBERT-v1.0', 
                    #    'bert-base-uncased',
                    #    'dmis-lab/biobert-base-cased-v1.2'
                       ],
        'LEARNING_RATE': [2e-05, 5e-05],
        'DROPOUT_RATE': [0, 0.1],
        'LOSS_TYPE': ['bce', 'focal', 'smooth_focal'],
        'BATCH_SIZE': [16, 32],
        'MAX_SEQ_LENGTH': [64, 128]
    }

    # MLFlow local server setup (for reference)
    """
    To start locally MLFlow:
    mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5002
    """
    mlflow.set_tracking_uri("http://localhost:5002")
    experiment_name = "EBI test"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)

    # Generate hyperparameter combinations
    hyperparameter_combinations = list(itertools.product(*GRID_SEARCH.values()))

    # Iterate over the hyperparameter combinations
    for combination in hyperparameter_combinations:
        hyperparameters = dict(zip(GRID_SEARCH.keys(), combination))

        with mlflow.start_run():

            # Log the hyperparameters
            log_params(hyperparameters)

            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hyperparameters['MODEL_NAME'])

            # Which entities there are?
            nested_list = diff_entities = df['ner'].apply(lambda x: [i[-1] for i in x])
            flatten_list = [x for sublist in nested_list for x in sublist]
            print(collections.Counter(flatten_list))
            annotation_types = ['Gene Mutations', 'Cell', 'Cell Line', 'Organ Tissue']
            # annotation_types = ['GM', 'CL', 'CLine', 'OT']
            label_to_id = {v:k for k,v in enumerate(annotation_types)}

            # Define the datasets and loaders
            loaders = dict()
            datasets = dict()
            for key in SETS:

                # Input text
                text = df['sentence'].tolist()
                annotations = df['ner'].values

                # Prepare your data
                mask = df['set'] == key
                tokens_and_labels = get_tokens_and_labels(
                    df.loc[mask, 'sentence'].loc[mask].tolist(),
                    df.loc[mask, 'ner'].values,
                    tokenizer, 
                    max_seq_length=hyperparameters['MAX_SEQ_LENGTH'],
                    annotation_types=annotation_types,
                    label_to_id=label_to_id,
                    )

                # Create an instance of CustomDataset
                datasets[key] = CustomDataset(tokens_and_labels)

                # Create a DataLoader for training
                loaders[key] = DataLoader(datasets[key], batch_size=hyperparameters['BATCH_SIZE'], shuffle=True)

            # Check
            for batch in loaders['train']:
                break

            model = EntityModel(
                model_name=hyperparameters['MODEL_NAME'],
                num_labels=4,
                dropout_rate=hyperparameters['DROPOUT_RATE'], 
                loss_type='focal',
                )

                
            # Define optimizer and scheduler
            optimizer = AdamW(model.parameters(), hyperparameters['LEARNING_RATE'])

            # Train the model
            train_and_evaluate(
                loaders,
                model,
                optimizer,
                num_epochs=1,
                log_interval=5,
                val_interval=50
                )
            
            # Extract hyperparameters
            model_name = hyperparameters['MODEL_NAME']
            learning_rate = hyperparameters['LEARNING_RATE']
            dropout_rate = hyperparameters['DROPOUT_RATE']
            loss_type = hyperparameters['LOSS_TYPE']
            batch_size = hyperparameters['BATCH_SIZE']
            max_seq_length = hyperparameters['MAX_SEQ_LENGTH']

            # Define a file path for saving the model
            model_path = (
                f"../../models/entity_model_"
                f"{model_name.replace('/', '-')}_{TIMESTAMP}_"
                f"lr{learning_rate:.0e}_dropout{dropout_rate}_"
                f"batch{batch_size}_seed{SEED}.pth"
            )
            # Save the model to the specified file path
            torch.save(model.state_dict(), model_path)
            logging.info(f'Saved model to: {model_path}')