import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.DawamaOFormerDataset import DawamaOFormerDataset
from helpers.prepare import prepare_data
from model.OFormerDawama import OFormerDawama

def train_model():
    
    csv_filename = "data.csv"
    
    if not os.path.exists(csv_filename):
        print(f"Error: Please make sure the file is in the current directory.")
        return
        
    print("Loading Dataset...")
    train_loader, test_loader = prepare_data(csv_filename, batch_size=256)
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = OFormerDawama(num_sensors=8, d_model=64, nhead=4).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 20
    print("\nStarting Training Loop...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_sensors, batch_queries, batch_labels in train_loader:
            batch_sensors = batch_sensors.to(device)
            batch_queries = batch_queries.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward Pass
            optimizer.zero_grad()
            preds = model(batch_sensors, batch_queries)
            loss = criterion(preds, batch_labels)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_sensors.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_sensors, batch_queries, batch_labels in test_loader:
                batch_sensors = batch_sensors.to(device)
                batch_queries = batch_queries.to(device)
                batch_labels = batch_labels.to(device)
                
                preds = model(batch_sensors, batch_queries)
                loss = criterion(preds, batch_labels)
                test_loss += loss.item() * batch_sensors.size(0)
                
        test_loss /= len(test_loader.dataset)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] -> Train MSE Loss: {train_loss:.6f} | Test MSE Loss: {test_loss:.6f}")

    print("\nTraining Completed!")
    torch.save(model.state_dict(), "oformer_dawama_model.pth")
    print("Model saved as 'oformer_dawama_model.pth'")