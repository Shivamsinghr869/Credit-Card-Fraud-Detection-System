"""
Machine Learning model module for credit card fraud detection.

This module contains classes and functions for training, evaluating, and making
predictions with fraud detection models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Tuple, Any, Optional
import joblib
import os


class FraudDetectionModel:
    """Class to handle fraud detection model training and evaluation."""
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Initialize the FraudDetectionModel.
        
        Args:
            model_type (str): Type of model ('logistic_regression' or 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.model_params = {}
        
        # Initialize model based on type
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model_params = {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs'
            }
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42)
            self.model_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def set_hyperparameters(self, **kwargs):
        """
        Set hyperparameters for the model.
        
        Args:
            **kwargs: Hyperparameters to set
        """
        for param, value in kwargs.items():
            if hasattr(self.model, param):
                setattr(self.model, param, value)
                self.model_params[param] = value
            else:
                st.warning(f"Parameter {param} not found for {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the fraud detection model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1_score': f1_score(y_train, y_train_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        }
        
        st.success(f"{self.model_type.replace('_', ' ').title()} model trained successfully!")
        return metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        return metrics, y_pred, y_proba
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure data has the same columns as training data
        if set(X.columns) != set(self.feature_columns):
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                return None, None
            
            # Select only the feature columns
            X = X[self.feature_columns]
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.
        
        Returns:
            pd.Series: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.model_type == 'logistic_regression':
            # For logistic regression, use absolute coefficients
            importance = np.abs(self.model.coef_[0])
        elif self.model_type == 'random_forest':
            # For random forest, use feature_importances_
            importance = self.model.feature_importances_
        else:
            return pd.Series()
        
        return pd.Series(importance, index=self.feature_columns).sort_values(ascending=False)
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'model_params': self.model_params,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        st.success(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.model_params = model_data['model_params']
        self.is_trained = model_data['is_trained']
        
        st.success(f"Model loaded from {filepath}")


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """
    Create a confusion matrix plot.
    
    Args:
        y_true (pd.Series): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        go.Figure: Plotly figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Non-Fraud', 'Predicted Fraud'],
        y=['Actual Non-Fraud', 'Actual Fraud'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        title_x=0.5,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    
    return fig


def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray) -> go.Figure:
    """
    Create a ROC curve plot.
    
    Args:
        y_true (pd.Series): True labels
        y_proba (np.ndarray): Predicted probabilities
        
    Returns:
        go.Figure: Plotly figure object
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='#4682B4', width=2)
    ))
    
    # Diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        title_x=0.5,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig


def plot_precision_recall_curve(y_true: pd.Series, y_proba: np.ndarray) -> go.Figure:
    """
    Create a Precision-Recall curve plot.
    
    Args:
        y_true (pd.Series): True labels
        y_proba (np.ndarray): Predicted probabilities
        
    Returns:
        go.Figure: Plotly figure object
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name='Precision-Recall Curve',
        line=dict(color='#2E8B57', width=2)
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        title_x=0.5,
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig


def plot_feature_importance(feature_importance: pd.Series, top_n: int = 15) -> go.Figure:
    """
    Create a feature importance plot.
    
    Args:
        feature_importance (pd.Series): Feature importance scores
        top_n (int): Number of top features to display
        
    Returns:
        go.Figure: Plotly figure object
    """
    top_features = feature_importance.head(top_n)
    
    fig = go.Figure(data=go.Bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        marker_color='#4682B4'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importance',
        title_x=0.5,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=500
    )
    
    return fig


def compare_models(models_dict: Dict[str, FraudDetectionModel], 
                   X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple models and return their metrics.
    
    Args:
        models_dict (Dict[str, FraudDetectionModel]): Dictionary of trained models
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        pd.DataFrame: Comparison table of model metrics
    """
    comparison_data = []
    
    for model_name, model in models_dict.items():
        if model.is_trained:
            metrics, _, _ = model.evaluate(X_test, y_test)
            metrics['Model'] = model_name
            comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Model')
    
    return comparison_df


def create_model_summary(model: FraudDetectionModel, metrics: Dict[str, float]) -> str:
    """
    Create a summary string for the model.
    
    Args:
        model (FraudDetectionModel): Trained model
        metrics (Dict[str, float]): Model metrics
        
    Returns:
        str: Model summary
    """
    summary = f"""
    **Model Type:** {model.model_type.replace('_', ' ').title()}
    
    **Model Parameters:**
    """
    
    for param, value in model.model_params.items():
        summary += f"- {param}: {value}\n"
    
    summary += f"""
    **Performance Metrics:**
    - Accuracy: {metrics['accuracy']:.4f}
    - Precision: {metrics['precision']:.4f}
    - Recall: {metrics['recall']:.4f}
    - F1-Score: {metrics['f1_score']:.4f}
    - ROC-AUC: {metrics['roc_auc']:.4f}
    """
    
    return summary 