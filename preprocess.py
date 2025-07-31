"""
Data preprocessing module for credit card fraud detection.

This module handles data cleaning, feature scaling, and train-test splitting
for the credit card fraud detection model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import streamlit as st
from typing import Tuple, Optional


class DataPreprocessor:
    """Class to handle data preprocessing for credit card fraud detection."""
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        # Check for missing values
        missing_values = data.isnull().sum()
        
        if missing_values.sum() > 0:
            st.warning(f"Found missing values: {missing_values[missing_values > 0].to_dict()}")
            
            # For numerical columns, fill with median
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].mode()[0], inplace=True)
        
        return data
    
    def remove_outliers(self, data: pd.DataFrame, columns: list, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from specified columns.
        
        Args:
            data (pd.DataFrame): Input dataset
            columns (list): List of columns to process
            method (str): Method to detect outliers ('iqr' or 'zscore')
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        data_clean = data.copy()
        
        for col in columns:
            if col in data.columns:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Remove outliers
                    outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    data_clean = data_clean[~outliers_mask]
                    
                elif method == 'zscore':
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outliers_mask = z_scores > 3
                    data_clean = data_clean[~outliers_mask]
        
        removed_count = len(data) - len(data_clean)
        if removed_count > 0:
            st.info(f"Removed {removed_count} outliers from the dataset.")
        
        return data_clean
    
    def balance_dataset(self, data: pd.DataFrame, target_col: str = 'Class', method: str = 'undersample') -> pd.DataFrame:
        """
        Balance the dataset to handle class imbalance.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            method (str): Balancing method ('undersample', 'oversample', or 'none')
            
        Returns:
            pd.DataFrame: Balanced dataset
        """
        if method == 'none':
            return data
        
        # Separate majority and minority classes
        majority_class = data[data[target_col] == 0]
        minority_class = data[data[target_col] == 1]
        
        if method == 'undersample':
            # Undersample majority class
            majority_downsampled = resample(
                majority_class,
                replace=False,
                n_samples=len(minority_class),
                random_state=42
            )
            balanced_data = pd.concat([majority_downsampled, minority_class])
            
        elif method == 'oversample':
            # Oversample minority class
            minority_upsampled = resample(
                minority_class,
                replace=True,
                n_samples=len(majority_class),
                random_state=42
            )
            balanced_data = pd.concat([majority_class, minority_upsampled])
        
        # Shuffle the data
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        st.success(f"Dataset balanced using {method}. New shape: {balanced_data.shape}")
        return balanced_data
    
    def prepare_features(self, data: pd.DataFrame, target_col: str = 'Class') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Store feature columns for later use
        self.feature_columns = [col for col in data.columns if col != target_col]
        
        # Separate features and target
        X = data[self.feature_columns]
        y = data[target_col]
        
        return X, y
    
    def scale_features(self, X: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X (pd.DataFrame): Feature matrix
            fit_scaler (bool): Whether to fit the scaler or use existing one
            
        Returns:
            pd.DataFrame: Scaled features
        """
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transforming new data")
            X_scaled = self.scaler.transform(X)
        
        # Convert back to DataFrame with original column names
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled_df
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        st.success(f"Data split: Train set {X_train.shape}, Test set {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, data: pd.DataFrame, target_col: str = 'Class', 
                           balance_method: str = 'none', remove_outliers: bool = False,
                           outlier_columns: Optional[list] = None) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                          pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            balance_method (str): Method for balancing dataset
            remove_outliers (bool): Whether to remove outliers
            outlier_columns (Optional[list]): Columns to check for outliers
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        # Step 1: Handle missing values
        data_clean = self.handle_missing_values(data.copy())
        
        # Step 2: Remove outliers if requested
        if remove_outliers and outlier_columns:
            data_clean = self.remove_outliers(data_clean, outlier_columns)
        
        # Step 3: Balance dataset
        data_balanced = self.balance_dataset(data_clean, target_col, balance_method)
        
        # Step 4: Prepare features and target
        X, y = self.prepare_features(data_balanced, target_col)
        
        # Step 5: Scale features
        X_scaled = self.scale_features(X, fit_scaler=True)
        
        # Step 6: Split data
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y)
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted scaler.
        
        Args:
            data (pd.DataFrame): New data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Run prepare_features first.")
        
        # Ensure data has the same columns as training data
        if set(data.columns) != set(self.feature_columns):
            missing_cols = set(self.feature_columns) - set(data.columns)
            extra_cols = set(data.columns) - set(self.feature_columns)
            
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            if extra_cols:
                st.warning(f"Extra columns will be ignored: {extra_cols}")
            
            # Select only the feature columns
            data = data[self.feature_columns]
        
        # Scale the data
        data_scaled = self.scale_features(data, fit_scaler=False)
        
        return data_scaled
    
    def get_feature_importance_ranking(self, data: pd.DataFrame, target_col: str = 'Class') -> pd.Series:
        """
        Get feature importance ranking based on correlation with target.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Target column name
            
        Returns:
            pd.Series: Feature importance ranking
        """
        feature_cols = [col for col in data.columns if col != target_col]
        correlations = data[feature_cols].corrwith(data[target_col]).abs().sort_values(ascending=False)
        
        return correlations 