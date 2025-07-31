"""
Data loader module for credit card fraud detection.

This module handles loading the credit card dataset and provides
basic data operations and information.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import streamlit as st


class DataLoader:
    """Class to handle loading and basic operations on credit card fraud data."""
    
    def __init__(self, file_path: str = 'creditcard.csv'):
        """
        Initialize the DataLoader.
        
        Args:
            file_path (str): Path to the credit card CSV file
        """
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the credit card fraud dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If the CSV file is not found
        """
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except FileNotFoundError:
            st.error(f"File {self.file_path} not found. Please ensure the file is in the project directory.")
            return pd.DataFrame()
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the dataset.
        
        Returns:
            dict: Dictionary containing dataset information
        """
        if self.data is None:
            return {}
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'fraud_count': int(self.data['Class'].sum()),
            'non_fraud_count': int((self.data['Class'] == 0).sum()),
            'fraud_percentage': float((self.data['Class'].sum() / len(self.data)) * 100)
        }
        return info
    
    def get_feature_columns(self) -> list:
        """
        Get list of feature columns (excluding target variable).
        
        Returns:
            list: List of feature column names
        """
        if self.data is None:
            return []
        
        return [col for col in self.data.columns if col != 'Class']
    
    def get_numerical_features(self) -> list:
        """
        Get list of numerical feature columns.
        
        Returns:
            list: List of numerical feature column names
        """
        if self.data is None:
            return []
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        return [col for col in numerical_cols if col != 'Class']
    
    def get_sample_data(self, n_samples: int = 5) -> pd.DataFrame:
        """
        Get a sample of the data for display purposes.
        
        Args:
            n_samples (int): Number of samples to return
            
        Returns:
            pd.DataFrame: Sample of the dataset
        """
        if self.data is None:
            return pd.DataFrame()
        
        return self.data.head(n_samples)
    
    def get_fraud_distribution(self) -> pd.Series:
        """
        Get the distribution of fraud vs non-fraud transactions.
        
        Returns:
            pd.Series: Series with fraud distribution
        """
        if self.data is None:
            return pd.Series()
        
        return self.data['Class'].value_counts()


def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for demonstration when actual CSV is not available.
    
    Returns:
        pd.DataFrame: Sample credit card transaction data
    """
    # Generate sample data similar to credit card fraud dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features similar to credit card data
    data = {}
    
    # Time feature
    data['Time'] = np.random.uniform(0, 172792, n_samples)
    
    # Amount feature
    data['Amount'] = np.random.exponential(88.35, n_samples)
    
    # V1-V28 features (PCA transformed features)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Class (fraud indicator) - typically very imbalanced
    fraud_ratio = 0.0017  # Typical fraud ratio in credit card data
    data['Class'] = np.random.choice([0, 1], size=n_samples, p=[1-fraud_ratio, fraud_ratio])
    
    return pd.DataFrame(data) 