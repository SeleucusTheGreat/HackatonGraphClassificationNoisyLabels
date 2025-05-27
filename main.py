import argparse
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm 
from loadData import GraphDataset
import os
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import f1_score 
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import logging
from datetime import datetime
import models 

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()
    
def add_zeros(data):
    data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
    return data


def setup_logging(dataset_name):
    """Setup logging for the specific dataset"""
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/log_{dataset_name}.log"
    
    # Create logger
    logger = logging.getLogger(f'training_{dataset_name}')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def train_epoch(model, data_loader, optimizer, criterion, device, epoch_num, logger):
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch_num + 1} Training", unit="batch")
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = total_loss / len(data_loader)
    torch.cuda.empty_cache()
    return avg_loss

def evaluate(model, data_loader, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total_samples_with_labels = 0 
    all_predictions = []
    all_true_labels = [] 

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())

            if calculate_accuracy and data.y is not None:
                all_true_labels.extend(data.y.cpu().numpy())
                correct += (pred == data.y).sum().item()
                total_samples_with_labels += data.y.size(0)
        
    accuracy = 0.0
    f1 = 0.0
    
    if calculate_accuracy and total_samples_with_labels > 0:
        accuracy = correct / total_samples_with_labels
        f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    elif calculate_accuracy:
        print("Warning: Accuracy and F1-score could not be calculated because no labels were found.")
            
    return all_predictions, accuracy, f1

def get_dataset_name(path):
    return os.path.basename(os.path.dirname(path))

def get_checkpoint_path(dataset_name, epoch=None, best=True):
    os.makedirs("checkpoints", exist_ok=True)
    if epoch is not None:
        return f"checkpoints/model_{dataset_name}_epoch_{epoch}.pth"
    if best:
        return f"checkpoints/model_{dataset_name}_best.pth"
    else:
        return f"checkpoints/model_{dataset_name}_best.pth"

def save_checkpoint(model, dataset_name, epoch, is_best=False):
    checkpoint_path = get_checkpoint_path(dataset_name, epoch)
    torch.save(model.state_dict(), checkpoint_path)
    
    if is_best:
        best_path = get_checkpoint_path(dataset_name)
        torch.save(model.state_dict(), best_path)

def load_best_checkpoint(model, dataset_name, device):
    best_path = get_checkpoint_path(dataset_name)
    if os.path.exists(best_path):
        print(f"Loading best checkpoint from {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device))
        return True
    else:
        print(f"No checkpoint found at {best_path}")
        return False

def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(420)
    
    # Extract dataset name from test path
    dataset_name = get_dataset_name(args.test_path)
    print(f"Processing dataset: {dataset_name}")
    
    
    logger = setup_logging(dataset_name)
    logger.info(f"Starting processing for dataset {dataset_name}")
    logger.info(f"Device: {device}")
    
    
    # Initialize model
    if dataset_name == "B" or dataset_name == "C":

        node_dim = 1 
        hidden_dim = 64
        output_dim = 6
        edge_dim = 7  
        dropout = 0.2
        learning_rate = 0.005
        num_epochs = 100
        batch_size = 64
   
           
        model = models.FatEdgeCentricGNN (
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            edge_dim=edge_dim,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = NoisyCrossEntropyLoss(p_noisy=0.2) 

    elif dataset_name == "A":
        node_dim = 1 
        hidden_dim = 64
        output_dim = 6
        edge_dim = 7  
        dropout = 0.2
        learning_rate = 0.001
        num_epochs = 100
        batch_size = 16
                 
        model = models.EdgeCentricGNN(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            edge_dim=edge_dim,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = NoisyCrossEntropyLoss(p_noisy=0.2) 

    elif dataset_name == "D":
        node_dim = 1 
        hidden_dim = 64
        output_dim = 6
        edge_dim = 7  
        dropout = 0.2
        learning_rate = 0.005
        num_epochs = 100
        batch_size = 64
                 
        model = models.FatEdgeCentricGNN  (
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            edge_dim=edge_dim,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = NoisyCrossEntropyLoss(p_noisy=0.4) 

    #default:
    else:
        node_dim = 1 
        hidden_dim = 64
        output_dim = 6
        edge_dim = 7  
        dropout = 0.2
        learning_rate = 0.005
        num_epochs = 100
        batch_size = 64
   
           
        model =  models.FatEdgeCentricGNN (
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            edge_dim=edge_dim,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = NoisyCrossEntropyLoss(p_noisy=0.2) 
    
   

    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if args.train_path:
        logger.info("Training mode: Training path provided")
        
        # Check if we should resume from the best checkpoint
        if args.resume_from_checkpoint:
            logger.info("Resuming from the best checkpoint...")
            if not load_best_checkpoint(model, dataset_name, device):
                logger.warning("No checkpoint found. Starting training from scratch.")

        # Load training dataset
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        
        # Split dataset into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size, shuffle=False)
        
        logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
        
        best_val_f1 = 0.0
        checkpoint_epochs = []  # Track which epochs we save checkpoints for
        
        for epoch in range(num_epochs):
            # Training
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)
            
            # Evaluation
            _, train_acc, train_f1 = evaluate(model, train_loader, device, calculate_accuracy=True)
            _, val_acc, val_f1 = evaluate(model, val_loader, device, calculate_accuracy=True)
            
            # Log every 10 epochs as required
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.6f}, "
                            f"Training Accuracy: {train_acc:.6f}, Training F1: {train_f1:.6f}, "
                            f"Validation Accuracy: {val_acc:.6f}, Validation F1: {val_f1:.6f}")
        
            
            print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.6f}, Validation Accuracy: {val_acc:.6f}, F1: {val_f1:.6f}") 

            
            # Save checkpoint if validation F1 improved
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_checkpoint(model, dataset_name, epoch + 1, is_best=True)
                logger.info(f"  --> New best model saved with Val F1: {val_f1:.6f}, epoch: {epoch + 1}")
            
            # Save regular checkpoints (at least 5 as required)
        
        logger.info(f"Training completed. Checkpoints saved at epochs: {checkpoint_epochs}")


    # Load the best model for inference
    logger.info("Loading best model for inference...")
    model.load_state_dict(torch.load(get_checkpoint_path(dataset_name, epoch=None, best=True), map_location=device))
    
    # Generate predictions
    logger.info("Generating predictions...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Predicting"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    
    # Create submission directory and save predictions
    os.makedirs("submission", exist_ok=True)
    output_csv_path = f"submission/testset_{dataset_name}.csv"
    
    test_graph_ids = list(range(len(test_dataset)))
    output_df = pd.DataFrame({
        "id": test_graph_ids[:len(predictions)],
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    logger.info(f"Predictions saved to {output_csv_path}")
    print(f"Predictions saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate EdgeCentric GNN model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--resume_from_checkpoint", action='store_true', help="Resume training from the best checkpoint if available.")
    args = parser.parse_args()
    main(args)