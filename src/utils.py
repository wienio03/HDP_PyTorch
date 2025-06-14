import torch 
import os 
import json 
import numpy as np

from logger import logger

def save_model(model, filepath, metadata=None): 
    """Saves model""" 
    os.makedirs(os.path.dirname(filepath), exist_ok=True) 
    checkpoint = { 'model_state_dict': model.state_dict(), 'model_architecture': str(model), 'metadata': metadata or {} } 
    torch.save(checkpoint, filepath) 
    logger.log(f"Model saved to {filepath}") 

def load_model(model, filepath): 
    """Loads model""" 
    checkpoint = torch.load(filepath, map_location='cpu') 
    model.load_state_dict(checkpoint['model_state_dict']) 
    logger.log(f"Model loaded from {filepath}") 
    return checkpoint.get('metadata', {}) 

def count_parameters(model): 
    """Counts parameters""" 
    total_params = sum(p.numel() for p in model.parameters()) 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    logger.log(f"Total parameters: {total_params:,}") 
    logger.log(f"Trainable parameters: {trainable_params:,}") 
    return total_params, trainable_params 

def set_seed(seed=42): 
    """Sets seed for reproducing""" 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False