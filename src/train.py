import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F

from src.model import EarlyStopping
from src.logger import logger

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Trainer:
    """
    Class for training the heart disease prediction model.
    This class handles the training and validation of the model, including
    logging the training history and implementing early stopping.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.history = {
            'train_loss': [], 
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """
        Epoch for model training

        Args:
            train_loader (DataLoader): DataLoader for the training data
            criterion (nn.Module): Loss function
            optimizer (torch.optim.Optimizer): Optimizer for model parameters
        Returns:
            avg_loss (float): Average loss for the epoch
            accuracy (float): Accuracy of the model on the training data
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
            data, target = data.to(self.device), target.float().to(self.device).unsqueeze(1) 

            optimizer.zero_grad()
            output = self.model(data) 
            loss = criterion(output, target)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        """
        Epoch for model validation 

        Args:
            val_loader (DataLoader): DataLoader for the validation data
            criterion (nn.Module): Loss function
        Returns:
            avg_loss (float): Average loss for the epoch
            accuracy (float): Accuracy of the model on the validation data
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validating'):
                data, target = data.to(self.device), target.float().to(self.device).unsqueeze(1)  
                output = self.model(data) 
                loss = criterion(output, target)

                total_loss += loss.item()
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item() 
    
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs=150, lr=0.001, weight_decay=1e-4, 
              use_class_weights=False, use_focal_loss=False, label_smoothing=0.0):
        """
        Prosty training loop bez skomplikowanych technik

        Args:
            train_loader (DataLoader): DataLoader for the training data
            val_loader (DataLoader): DataLoader for the validation data
            epochs (int): Number of epochs to train the model
            lr (float): Learning rate for the optimizer
            weight_decay (float): Weight decay for the optimizer
            use_class_weights (bool): Whether to use class weights for imbalanced data
            use_focal_loss (bool): Whether to use focal loss instead of cross entropy
            label_smoothing (float): Label smoothing factor
        Returns:
            history (dict): Dictionary containing training and validation loss and accuracy history 
        """
        
        criterion = nn.BCELoss()
        logger.info("Using BCELoss for binary classification")

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Scheduler: ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

        # EarlyStopping z większą cierpliwością
        early_stopping = EarlyStopping(patience=15, min_delta=0.001, restore_best_weights=True)  

        logger.info(msg=f'Starting binary classification training for {epochs} epochs...')

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)            
            scheduler.step(val_loss)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            logger.info(msg=f'Epoch {epoch+1}/{epochs}:')
            logger.info(msg=f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(msg=f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(msg=f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            logger.info(msg='-' * 50)
            if early_stopping(val_loss, self.model):
                logger.info(msg=f'Early stopping at epoch {epoch + 1}')
                break
        return self.history
