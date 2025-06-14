import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import Dataset, DataLoader
import os

from logger import logger

class HeartDiseaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataPreprocessor:
    def __init__(self, X, y):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
    
    def load_data(self, data_path=None):
        """
        Loads the heart disease dataset. 
        If dataset is not present it loads a synthetic dataset.
        """
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            logger.error('No dataset is present')
            raise FileNotFoundError(f'{data_path} not found')
    
    def preprocess_data(self, df: pd.DataFrame, fit_transform=True):
        """
        Preprocessess data
        """
        df = df.copy()

        logger.log(f'Preprocessing data with fit_transform {fit_transform}')
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if any(df[numeric_columns].isnull().sum()):
            imputer = SimpleImputer(strategy='median')
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        X = df.drop('target', axis=1)
        y = df['target'].values

        self.feature_names = X.columns.tolist()

        if fit_transform:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def create_data_loader(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        train_dataset = HeartDiseaseDataset(X_train, y_train)
        val_dataset = HeartDiseaseDataset(X_val, y_val)
        test_dataset = HeartDiseaseDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)