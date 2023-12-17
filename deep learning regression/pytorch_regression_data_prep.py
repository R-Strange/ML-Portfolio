import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import utils, from_numpy
from torch.utils.data import DataLoader

def get_diabetes_data(batch_size = 64, normalized=False, min_max=False):
    data_df = load_diabetes(as_frame=True).frame.astype(np.float32)

    if normalized:
        scaler = StandardScaler()
        for column in data_df.columns:
            data_df[column] = scaler.fit_transform(data_df[column].values.reshape(-1, 1))

    if min_max:
        scaler = MinMaxScaler()
        for column in data_df.columns:
            data_df[column] = scaler.fit_transform(data_df[column].values.reshape(-1, 1))

    train_df, test_df = train_test_split(data_df, test_size=0.3, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=42)

    train_target = from_numpy(train_df.pop('target').values)
    val_target = from_numpy(val_df.pop('target').values)
    test_target = from_numpy(test_df.pop('target').values)

    train_tensor = from_numpy(train_df.values)
    test_tensor = from_numpy(test_df.values)
    val_tensor = from_numpy(val_df.values)

    train_dataset = utils.data.TensorDataset(train_tensor, train_target)
    val_dataset = utils.data.TensorDataset(val_tensor, val_target)
    test_dataset = utils.data.TensorDataset(test_tensor, test_target)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader