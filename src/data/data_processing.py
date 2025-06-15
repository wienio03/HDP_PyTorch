import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import Dataset, DataLoader
import os

from src.logger import logger


class HeartDiseaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataPreprocessor:
    """
    Class for preprocessing heart disease dataset.
    It handles loading, scaling, encoding, and creating DataLoaders for training, validation, and test sets.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
    
    def load_data(self, data_path='src/data/raw/heart.csv'):
        """
        Loads the heart disease dataset. 
        If dataset is not present it loads a synthetic dataset.

        Args:
            data_path (str): Path to the dataset CSV file.
        """
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(msg=f'Successfully loaded data from {data_path}')
            return df
        else:
            logger.error(f'Dataset not found at {data_path}')
            raise FileNotFoundError(f'{data_path} not found')
    
    def preprocess_data(self, df: pd.DataFrame, fit_transform=True):
        """
        Preprocessess data.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            fit_transform (bool): If True, fit the scaler and transform the data.
                                  If False, only transform the data using the fitted scaler.
        
        Returns:
            Tuple of (X_scaled, y) where:
                X_scaled (np.ndarray): Scaled feature matrix.
                y (np.ndarray): Target variable.
        """
        df = df.copy()

        logger.info(msg=f'Preprocessing data with fit_transform {fit_transform}')
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if any(df[numeric_columns].isnull().sum()):
            imputer = SimpleImputer(strategy='median')
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        # Convert categorical columns to numeric using label encoding
        # To keep multiclass you should not do y_multiclass > 0 
        X = df.drop('num', axis=1)
        y_multiclass = df['num'].values.astype(int)
        y = (y_multiclass > 0).astype(int) 
        
        logger.info(f"Converted to binary classification: {np.bincount(y)} samples per class")

        self.feature_names = X.columns.tolist()

        if fit_transform:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """
        Create DataLoaders for training, validation, and test sets with data augmentation
        
        Args:
            X_train (np.ndarray): Training features.
            X_val (np.ndarray): Validation features.
            X_test (np.ndarray): Test features.
            y_train (np.ndarray): Training labels.
            y_val (np.ndarray): Validation labels.
            y_test (np.ndarray): Test labels.
            batch_size (int): Batch size for DataLoader.
        Returns:
            Tuple of DataLoaders for train, validation, and test sets
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train) 
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets without augmentation
        train_dataset = HeartDiseaseDataset(X_train_tensor, y_train_tensor, augment=False)
        val_dataset = HeartDiseaseDataset(X_val_tensor, y_val_tensor, augment=False)
        test_dataset = HeartDiseaseDataset(X_test_tensor, y_test_tensor, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader