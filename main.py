import torch 
import torch.nn as nn 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import os 

from src.pipeline import HeartDiseasePipeline, get_default_config
from src.logger import logger

def main(): 
    """
    Main function to run the heart disease prediction pipeline.
    This function initializes the pipeline with default configuration,
    runs the pipeline, and returns the final evaluation results.
    It handles the entire process from data loading to model evaluation and saving.

    Returns:
        Final evaluation results from the pipeline
    """
    # Get configuration
    config = get_default_config()
    
    # Create and run pipeline
    pipeline = HeartDiseasePipeline(config)
    results = pipeline.run_pipeline()
    
    return results

if __name__ == "__main__": 
    main()