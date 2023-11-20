import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import tsensor
import torchinfo
import tensorboard

import torch
from torch import nn, optim, utils, tensor, from_numpy
from torch.utils.data import Dataset, DataLoader
import lightning as l
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def get_diabetes_data(batch_size = 64):
    data_df = load_diabetes(as_frame=True).frame.astype(np.float32)

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

class LightningModel(l.LightningModule):
    def __init__(self, model, training_step, validation_step, test_step, configure_optimizers):
        super().__init__()
        self.model = model
        self.passed_training_step = training_step
        self.passed_validation_step = validation_step
        self.passed_test_step = test_step
        self.passed_configure_optimizers = configure_optimizers

    def training_step(self, batch, batch_idx):
        return self.passed_training_step(self, batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.passed_test_step(self, batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = self.passed_configure_optimizers(self)
        return optimizer
    
    def validation_step(self, batch):
        return self.passed_validation_step(self, batch)
    