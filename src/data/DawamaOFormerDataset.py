import torch
from torch.utils.data import Dataset


class DawamaOFormerDataset(Dataset):
    def __init__(self, sensors, queries, labels):
        self.sensors = torch.tensor(sensors, dtype=torch.float32)
        self.queries = torch.tensor(queries, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sensors[idx], self.queries[idx], self.labels[idx]