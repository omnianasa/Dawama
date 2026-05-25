import os
import torch
import torch.nn as nn
from helpers.prepare import prepare_data
from model.FittedDawamaOFormer import FittedDawamaOFormer


def train_model():
    csv_filename = "data.csv"
    
    if not os.path.exists(csv_filename):
        print(f"Error: Please make sure '{csv_filename}' is in the current directory.")
        return
        
    print("Loading Dataset...")
    train_loader, test_loader = prepare_data(csv_filename, batch_size=256)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = FittedDawamaOFormer(num_sensors=8, d_model=64, nhead=4).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    epochs = 20
    print("\nStarting Bulletproof OFormer Training Loop)...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        prev_latent = None 
        
        for batch_sensors, batch_queries, batch_labels in train_loader:
            batch_sensors = batch_sensors.to(device)
            batch_queries = batch_queries.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            preds, updated_kv = model(batch_sensors, batch_queries, prev_latent)
            prev_latent = updated_kv.detach()
            
            loss = criterion(preds, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * batch_sensors.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        scheduler.step()
        
        model.eval()
        test_loss = 0.0
        test_latent = None
        with torch.no_grad():
            for batch_sensors, batch_queries, batch_labels in test_loader:
                batch_sensors = batch_sensors.to(device)
                batch_queries = batch_queries.to(device)
                batch_labels = batch_labels.to(device)
                
                preds, updated_kv = model(batch_sensors, batch_queries, test_latent)
                test_latent = updated_kv.detach()
                
                loss = criterion(preds, batch_labels)
                test_loss += loss.item() * batch_sensors.size(0)
                
        test_loss /= len(test_loader.dataset)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{epochs}] | LR: {current_lr:.6f} -> Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")

    print("\nSuper-OFormer Training Complete!")
    torch.save(model.state_dict(), "fitted_oformer_dawama_stable.pth")