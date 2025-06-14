import torch 
import torch.nn as nn 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import os 

from src.data.data_processing import DataPreprocessor 
from src.model import HeartDiseaseNet 
from src.train import Trainer 
from src.evaluate import Evaluator 
from src.utils import save_model, load_model, count_parameters, set_seed 
from logger import logger

def main(): 
    # Configuring set_seed(42) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    logger.log(f"Using device: {device}") 

    # Parameters
    config = { 'batch_size': 32, 'learning_rate': 0.001, 'epochs': 100, 'hidden_dims': [128, 64, 32], 'dropout_rate': 0.3, 'weight_decay': 1e-5, 'test_size': 0.2, 'val_size': 0.2 } 
    logger.log("=== HEART DISEASE PREDICTION ===") 
    logger.log(f"Configuration: {config}") 

    # Data preparation
    logger.log("\n1. Loading and preprocessing data...") 
    preprocessor = DataPreprocessor() 

    # Data loading
    df = preprocessor.load_data()  
    logger.log(f"Dataset shape: {df.shape}") 
    logger.log(f"Target distribution:\n{df['target'].value_counts()}") 

    # Preprocessing 
    X, y = preprocessor.preprocess_data(df) 

    # Data split 
    X_temp, X_test, y_temp, y_test = train_test_split( X, y, test_size=config['test_size'], random_state=42, stratify=y ) 
    X_train, X_val, y_train, y_val = train_test_split( X_temp, y_temp, test_size=config['val_size']/(1-config['test_size']), random_state=42, stratify=y_temp ) 
    logger.log(f"Train set: {X_train.shape[0]} samples") 
    logger.log(f"Validation set: {X_val.shape[0]} samples") 
    logger.log(f"Test set: {X_test.shape[0]} samples") 

    # DataLoaders 
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders( X_train, X_val, X_test, y_train, y_val, y_test, batch_size=config['batch_size'] ) 

    # 2. Model 
    logger.log("\n2. Creating model...") 
    input_dim = X_train.shape[1] 
    model = HeartDiseaseNet(input_dim=input_dim, hidden_dims=config['hidden_dims'], dropout_rate=config['dropout_rate'] ) 
    logger.log(f"Model architecture:\n{model}") 
    count_parameters(model) 

    # 3. Training 
    logger.log("\n3. Training model...") 
    trainer = Trainer(model, device) 
    history = trainer.train( train_loader, val_loader, epochs=config['epochs'], lr=config['learning_rate'], weight_decay=config['weight_decay'] ) 

    # 4. Evaluation 
    logger.log("\n4. Evaluating model...") 
    evaluator = Evaluator(model, device) 
    predictions, probabilities, targets = evaluator.evaluate(test_loader) 

    # Generate report 
    results = evaluator.generate_report(predictions, probabilities, targets) 

    # Plots 
    evaluator.plot_results(history, predictions, probabilities, targets) 
    
    #5. Save model 
    logger.log("\n5. Saving model...") 
    os.makedirs('models', exist_ok=True) 
    metadata = { 'config': config, 'feature_names': preprocessor.feature_names, 'test_accuracy': results['classification_report']['accuracy'], 'test_roc_auc': results['roc_auc'] } 
    save_model(model, 'models/heart_disease_model.pth', metadata) 
    logger.log("\n=== PROJECT COMPLETED ===") 
    logger.log(f"Final Test Accuracy: {results['classification_report']['accuracy']:.4f}") 
    logger.log(f"Final Test ROC AUC: {results['roc_auc']:.4f}") 
    if __name__ == "_main_": main()