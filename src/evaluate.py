import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib as plt
import seaborn as sns
from logger import logger

class Evaluator:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def evaluate(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data).squeeze()

                probabilities = output.cpu().numpy()
                predicitons = (output > 0.5).float().cpu().numpy()
                targets = target.cpu().numpy()

                all_predictions.extend(predicitons)
                all_probabilities.extend(probabilities)
                all_targets.extend(targets)
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)
    
    def generate_report(self, predictions, probabilities, targets):
        """
        Generates complete evaluation report
        """
        logger.log('==== EVALUATION REPORT ====')
        logger.log('Classification:')
        logger.log(classification_report(targets, predictions, target_names=['No Disease', 'Disease']))

        logger.log(f'ROC AUC Score: {roc_auc_score(targets, probabilities)}')

        cm = confusion_matrix(targets, predictions)
        logger.log('Confusion matrix:')
        logger.log(cm)

        return {
            'classification_report': classification_report(targets, predictions, output_dict=True),
            'roc_auc': roc_auc_score(targets, probabilities),
            'confusion_matrix': cm
        }


def plot_results(self, history, predictions, probabilities, targets): 
    """Tworzy wykresy wyników""" 
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
    # ROC Curve
    fpr, tpr, _ = roc_curve(targets, probabilities) 
    auc_score = roc_auc_score(targets, probabilities) 
    axes[1, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})') 
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier') 
    axes[1, 0].set_title('ROC Curve') 
    axes[1, 0].set_xlabel('False Positive Rate') 
    axes[1, 0].set_ylabel('True Positive Rate') 
    axes[1, 0].legend() 
    axes[1, 0].grid(True) 
    # Confusion Matrix 
    cm = confusion_matrix(targets, predictions) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1]) 
    axes[1, 1].set_title('Confusion Matrix') 
    axes[1, 1].set_xlabel('Predicted') 
    axes[1, 1].set_ylabel('Actual') 
    plt.tight_layout() 
    plt.show()