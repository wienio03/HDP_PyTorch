import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import logger

class Evaluator:
    """
    Class for evaluating the heart disease prediction model.
    It computes predictions, probabilities, and generates a comprehensive evaluation report.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def evaluate(self, test_loader):
        """
        Evaluate model on test set for multi-class classification (0-4).
        
        Returns:
            predictions, probabilities, targets for multi-class evaluation
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                probabilities = torch.sigmoid(output).cpu().numpy()
                predictions = (torch.sigmoid(output) > 0.5).float().cpu().numpy()
                targets = target.cpu().numpy()

                all_predictions.extend(predictions.flatten())
                all_probabilities.extend(probabilities.flatten())
                all_targets.extend(targets)
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)
    
    def generate_report(self, predictions, probabilities, targets):
        """
        Generates complete evaluation report for binary classification (0-1).

        Args:
            predictions (np.ndarray): Model predictions (0,1).
            probabilities (np.ndarray): Predicted probabilities.
            targets (np.ndarray): True labels (0,1).
        Returns:
            dict: Evaluation report containing classification metrics and confusion matrix.
        """
        logger.info(msg='==== EVALUATION REPORT (Binary Classification) ====')
        
        # Class names for binary classification
        class_names = ['No Disease (0)', 'Disease (1)']
        
        logger.info(msg='Classification Report:')
        logger.info(msg=classification_report(targets, predictions, target_names=class_names, zero_division=0))

        cm = confusion_matrix(targets, predictions)
        logger.info(msg='Confusion Matrix:')
        logger.info(msg=cm)

        return {
            'classification_report': classification_report(targets, predictions, output_dict=True),
            'confusion_matrix': cm,
            'class_names': class_names
        }

    def plot_results(self, history, predictions, probabilities, targets): 
        """
        Plots training history and confusion matrix for binary classification.

        Args:
            history (dict): Training history containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
            predictions (np.ndarray): Model predictions (0,1).
            probabilities (np.ndarray): Predicted probabilities.
            targets (np.ndarray): True labels (0,1).
        Returns:
            None: Displays the plots.
        """ 
        fig, axes = plt.subplots(2, 2, figsize=(15, 12)) 
        
        # Training history 
        axes[0, 0].plot(history['train_loss'], label='Train Loss') 
        axes[0, 0].plot(history['val_loss'], label='Validation Loss') 
        axes[0, 0].set_title('Training and Validation Loss') 
        axes[0, 0].set_xlabel('Epoch') 
        axes[0, 0].set_ylabel('Loss') 
        axes[0, 0].legend() 
        axes[0, 0].grid(True) 
        
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy') 
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy') 
        axes[0, 1].set_title('Training and Validation Accuracy') 
        axes[0, 1].set_xlabel('Epoch') 
        axes[0, 1].set_ylabel('Accuracy (%)') 
        axes[0, 1].legend() 
        axes[0, 1].grid(True) 
        
        # Class distribution
        unique, counts = np.unique(targets, return_counts=True)
        axes[1, 0].bar(unique, counts, color='skyblue', alpha=0.7)
        axes[1, 0].set_title('Class Distribution in Test Set')
        axes[1, 0].set_xlabel('Heart Disease (0=No, 1=Yes)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion Matrix 
        cm = confusion_matrix(targets, predictions)
        class_names = ['No Disease (0)', 'Disease (1)']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                   xticklabels=class_names, yticklabels=class_names) 
        axes[1, 1].set_title('Confusion Matrix') 
        axes[1, 1].set_xlabel('Predicted') 
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].tick_params(axis='y', rotation=0) 
        plt.tight_layout() 
        plt.show()