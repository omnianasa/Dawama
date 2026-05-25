import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from data.DawamaOFormerDataset import DawamaOFormerDataset


def prepare_data(csv_path, batch_size=256):
    df = pd.read_csv(csv_path)
    
    sensor_cols = [f'Sensor_P{i}' for i in range(1, 9)]
    query_cols = ['Query_X', 'Query_Y']
    label_col = df.columns[-1]

    X_sensors = df[sensor_cols].values
    X_queries = df[query_cols].values
    y_labels = df[label_col].values
    
    scaler_s = StandardScaler()
    scaler_q = StandardScaler()
    
    X_sensors = scaler_s.fit_transform(X_sensors)
    X_queries = scaler_q.fit_transform(X_queries)
    
    X_s_train, X_s_test, X_q_train, X_q_test, y_train, y_test = train_test_split(
        X_sensors, X_queries, y_labels, test_size=0.2, random_state=42
    )
    
    train_dataset = DawamaOFormerDataset(X_s_train, X_q_train, y_train)
    test_dataset = DawamaOFormerDataset(X_s_test, X_q_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader