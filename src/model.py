import torch
import torch.nn as nn
import torch.nn.functional as F
from src.logger import logger

class HeartDiseaseNet(nn.Module):
    """
    Class for a neural network model to predict heart disease.
    This model consists of multiple fully connected layers with batch normalization,
    ReLU activation, and dropout for regularization.  
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.5, num_classes=1):
        super(HeartDiseaseNet, self).__init__()

        layers = []
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            else:
                layers.append(nn.Dropout(dropout_rate * 0.5))
            in_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.out(x)
        return torch.sigmoid(x)

class EarlyStopping:
    """
    Early stopping to halt training when validation loss does not improve.
    
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best validation loss.
        model (nn.Module): The model to be monitored.
        val_loss (float): The validation loss to monitor.
        counter (int): Counter for epochs without improvement.
        best_loss (float): Best validation loss observed.
        best_weights (dict): Weights of the model at the epoch with the best validation loss.   
    """
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True 

        return False

    def save_checkpoint(self, model: nn.Module):
        self.best_weights = model.state_dict().copy()