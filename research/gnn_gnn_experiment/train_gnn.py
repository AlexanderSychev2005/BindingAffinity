import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from dataset_gnn import ProteinLigandGraphDataset, collate_fn
from model_gnn import GNNGNNModel

def train():
    # 1. Hyperparameters
    batch_size = 32
    lr = 1e-4
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Data
    df = pd.read_csv("research/pdbbind_refined_dataset.csv")
    dataset = ProteinLigandGraphDataset(df)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Model
    # Get input dims from a sample
    sample = dataset[0]
    l_in = sample['ligand_x'].shape[1]
    p_in = sample['protein_x'].shape[1]
    
    model = GNNGNNModel(l_in, p_in).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for l_batch, p_batch, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            l_batch, p_batch, y = l_batch.to(device), p_batch.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(l_batch, p_batch)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for l_batch, p_batch, y in val_loader:
                l_batch, p_batch, y = l_batch.to(device), p_batch.to(device), y.to(device)
                pred = model(l_batch, p_batch)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "research/gnn_gnn_experiment/best_model_gnn.pth")
            print("Model saved!")

if __name__ == "__main__":
    train()
