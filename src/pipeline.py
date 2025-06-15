import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Any

from src.data.data_processing import DataPreprocessor
from src.model import HeartDiseaseNet
from src.train import Trainer
from src.evaluate import Evaluator
from src.utils import save_model, load_model, count_parameters, set_seed
from src.logger import logger


class HeartDiseasePipeline:
    """
    Complete pipeline for heart disease prediction including:
    - Data loading and preprocessing
    - Model training
    - Model evaluation
    - Results visualization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Dictionary containing model and training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        
        # Set random seed for reproducibility
        set_seed(42)
        logger.info(msg=f"Using device: {self.device}")
        logger.info(msg=f"Pipeline configuration: {self.config}")
    
    def load_and_preprocess_data(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """
        Load and preprocess the heart disease dataset.
        
        Returns:
            Tuple of DataLoaders for train, validation, and test sets
        """
        logger.info(msg="\n=== DATA LOADING AND PREPROCESSING ===")
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Load data
        df = self.preprocessor.load_data()
        logger.info(msg=f"Dataset shape: {df.shape}")
        
        # Target column name - the UCIML dataset uses 'num' for multi-class classification
        if 'num' in df.columns:
            logger.info(msg="Using 'num' column for multi-class classification (0=no disease, 1-4=disease severity)")
            logger.info(msg=f"Target distribution:\n{df['num'].value_counts().sort_index()}")
        else:
            logger.error(msg="Column 'num' not found in dataset!")
            raise ValueError("Dataset must contain 'num' column for multi-class classification")
        
        # Preprocess data
        X, y = self.preprocessor.preprocess_data(df)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=42, 
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['val_size']/(1-self.config['test_size']),
            random_state=42,
            stratify=y_temp
        )
        
        logger.info(msg=f"Train set: {X_train.shape[0]} samples")
        logger.info(msg=f"Validation set: {X_val.shape[0]} samples")
        logger.info(msg=f"Test set: {X_test.shape[0]} samples")
        
        # Create DataLoaders
        train_loader, val_loader, test_loader = self.preprocessor.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test,
            batch_size=self.config['batch_size']
        )
        
        return train_loader, val_loader, test_loader, X_train.shape[1]
    
    def create_model(self, input_dim: int) -> HeartDiseaseNet:
        """
        Create and initialize the neural network model.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Initialized HeartDiseaseNet model
        """
        logger.info(msg="\n=== MODEL CREATION ===")
        
        self.model = HeartDiseaseNet(
            input_dim=input_dim,
            hidden_dims=self.config['hidden_dims'],
            dropout_rate=self.config['dropout_rate'],
            num_classes=1
        )
        
        logger.info(msg=f"Model architecture:\n{self.model}")
        count_parameters(self.model)
        
        return self.model
    
    def train_model(self, train_loader, val_loader) -> Dict[str, list]:
        """
        Train the model using the training and validation data.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Returns:
            Training history dictionary
        """
        logger.info(msg="\n=== MODEL TRAINING ===")
        
        self.trainer = Trainer(self.model, self.device)
        history = self.trainer.train(
            train_loader, val_loader,
            epochs=self.config['epochs'],
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            use_class_weights=self.config.get('use_class_weights', False),
            use_focal_loss=self.config.get('use_focal_loss', False),
            label_smoothing=self.config.get('label_smoothing', 0.0)
        )
        
        return history
    
    def evaluate_model(self, test_loader) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(msg="\n=== MODEL EVALUATION ===")
        
        self.evaluator = Evaluator(self.model, self.device)
        predictions, probabilities, targets = self.evaluator.evaluate(test_loader)
        
        # Generate comprehensive report
        results = self.evaluator.generate_report(predictions, probabilities, targets)
        
        return results, predictions, probabilities, targets
    
    def visualize_results(self, history, predictions, probabilities, targets):
        """
        Create visualizations of training and evaluation results.
        
        Args:
            history: Training history
            predictions: Model predictions
            probabilities: Prediction probabilities
            targets: True labels
        """
        logger.info(msg="\n=== RESULTS VISUALIZATION ===")
        
        self.evaluator.plot_results(history, predictions, probabilities, targets)
    
    def save_model(self, results: Dict[str, Any]):
        """
        Save the trained model with metadata.
        
        Args:
            results: Evaluation results for metadata
        """
        logger.info(msg="\n=== MODEL SAVING ===")
        
        os.makedirs('models', exist_ok=True)
        metadata = {
            'config': self.config,
            'feature_names': self.preprocessor.feature_names,
            'test_accuracy': results['classification_report']['accuracy'],
            'test_roc_auc': results.get('roc_auc', 'N/A')  # Safe access
        }
        
        save_model(self.model, 'models/heart_disease_model.pth', metadata)
        logger.info(msg="Model saved successfully!")
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete ML pipeline.
        
        Returns:
            Final evaluation results
        """
        logger.info(msg="=== HEART DISEASE PREDICTION PIPELINE ===")
        
        # Step 1: Data loading and preprocessing
        train_loader, val_loader, test_loader, input_dim = self.load_and_preprocess_data()
        
        # Step 2: Model creation
        self.create_model(input_dim)
        
        # Step 3: Model training
        history = self.train_model(train_loader, val_loader)
        
        # Step 4: Model evaluation
        results, predictions, probabilities, targets = self.evaluate_model(test_loader)
        
        # Step 5: Results visualization
        self.visualize_results(history, predictions, probabilities, targets)
        
        # Step 6: Model saving
        self.save_model(results)
        
        # Final results
        logger.info(msg="\n=== PIPELINE COMPLETED ===")
        logger.info(msg=f"Final Test Accuracy: {results['classification_report']['accuracy']:.4f}")
        
        return results


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the heart disease prediction pipeline.
     
    Returns:
        Ultra simple configuration - prawie jak logistic regression
    """
    return {
        'batch_size': 64,               
        'learning_rate': 0.001,          
        'epochs': 300,                    
        'hidden_dims': [64, 32, 16],      
        'dropout_rate': 0.5,             
        'weight_decay': 1e-3,               
        'test_size': 0.15,                 
        'val_size': 0.15,                   
        'use_class_weights': False,            
        'use_focal_loss': False,               
        'label_smoothing': 0.0                 
    }
