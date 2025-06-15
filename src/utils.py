import torch 
import os 
import json 
import numpy as np

from src.logger import logger

def save_model(model, filepath, metadata=None): 
    """
    Saves model
    
    Args:
        model (torch.nn.Module): The PyTorch model to save.
        filepath (str): Path where the model will be saved.
        metadata (dict, optional): Additional metadata to save with the model.
    Returns:
        None
    """ 
    os.makedirs(os.path.dirname(filepath), exist_ok=True) 
    checkpoint = { 'model_state_dict': model.state_dict(), 'model_architecture': str(model), 'metadata': metadata or {} } 
    torch.save(checkpoint, filepath) 
    logger.info(msg=f"Model saved to {filepath}") 

def load_model(model, filepath): 
    """
    Loads model
    
    Args:
        model (torch.nn.Module): The PyTorch model to load the state into.
        filepath (str): Path from where the model will be loaded.
    Returns:
        dict: Metadata associated with the model.   
    """ 
    checkpoint = torch.load(filepath, map_location='cpu') 
    model.load_state_dict(checkpoint['model_state_dict']) 
    logger.info(msg=f"Model loaded from {filepath}") 
    return checkpoint.get('metadata', {}) 

def count_parameters(model): 
    """
    Counts parameters
    
    Args:
        model (torch.nn.Module): The PyTorch model for which to count parameters.
    Returns:
        tuple: Total number of parameters and number of trainable parameters.   
    """ 
    total_params = sum(p.numel() for p in model.parameters()) 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    logger.info(msg=f"Total parameters: {total_params:,}") 
    logger.info(msg=f"Trainable parameters: {trainable_params:,}") 
    return total_params, trainable_params 

def set_seed(seed=42): 
    """
    Sets seed for reproducing
    
    Args:
        seed (int): Seed value for random number generators.
    Returns:
        None
    """ 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False