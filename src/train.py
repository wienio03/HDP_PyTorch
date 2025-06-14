import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from model import EarlyStopping
from logger import logger

class Trainer:
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
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data).squeeze()
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
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data).squeeze()
                loss = criterion(output, target)

                total_loss += loss.item()
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct = (predicted == target).sum().item()
    
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs=100, lr=0.001, weight_decay=1e-5):
        """
        Main training loop
        """
        criterion =  nn.BCELoss()

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        early_stopping = EarlyStopping(patience=15, min_delta=0.001)

        logger.log(f'Starting training for {epochs} epochs...')

        for epoch in range(epochs):
            # training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)            
            
            # learning rate schedulign
            scheduler.step(val_loss)

            # save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            logger.log(f'Epoch {epoch+1}/{epochs}:')
            logger.log(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            logger.log(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            logger.log(f'LR: {optimizer.param_groups[0]['lr']:.6f}')
            logger.log('-' * 50)

            if early_stopping(val_loss, self.model):
                logger.log(f'Early stopping at epoch {epoch + 1}')
                break

        return self.history
