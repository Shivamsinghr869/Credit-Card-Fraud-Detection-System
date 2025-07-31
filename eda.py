"""
Exploratory Data Analysis module for credit card fraud detection.

This module contains functions for analyzing and visualizing the credit card
fraud dataset to understand patterns and relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Tuple, List


def plot_fraud_distribution(data: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing the distribution of fraud vs non-fraud transactions.
    
    Args:
        data (pd.DataFrame): The credit card dataset
        
    Returns:
        go.Figure: Plotly figure object
    """
    fraud_counts = data['Class'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Non-Fraud', 'Fraud'],
        values=[fraud_counts[0], fraud_counts[1]],
        hole=0.3,
        marker_colors=['#2E8B57', '#DC143C']
    )])
    
    fig.update_layout(
        title='Distribution of Fraud vs Non-Fraud Transactions',
        title_x=0.5,
        showlegend=True
    )
    
    return fig


def plot_correlation_heatmap(data: pd.DataFrame, max_features: int = 15) -> go.Figure:
    """
    Create a correlation heatmap for numerical features.
    
    Args:
        data (pd.DataFrame): The credit card dataset
        max_features (int): Maximum number of features to include in heatmap
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Select numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limit features for better visualization
    if len(numerical_cols) > max_features:
        # Select features with highest correlation with target
        correlations = data[numerical_cols].corrwith(data['Class']).abs().sort_values(ascending=False)
        selected_features = correlations.head(max_features).index.tolist()
        selected_features.append('Class')
    else:
        selected_features = numerical_cols
    
    # Calculate correlation matrix
    corr_matrix = data[selected_features].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Correlation Heatmap of Features',
        title_x=0.5,
        xaxis_title="Features",
        yaxis_title="Features",
        width=800,
        height=600
    )
    
    return fig


def plot_amount_distribution(data: pd.DataFrame) -> go.Figure:
    """
    Create box plots showing amount distribution by fraud class.
    
    Args:
        data (pd.DataFrame): The credit card dataset
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Amount Distribution by Class', 'Amount Distribution (Log Scale)'),
        specs=[[{"type": "box"}, {"type": "box"}]]
    )
    
    # Regular box plot
    for class_val in [0, 1]:
        class_data = data[data['Class'] == class_val]['Amount']
        fig.add_trace(
            go.Box(
                y=class_data,
                name=f'Class {class_val}',
                boxpoints='outliers',
                marker_color='#2E8B57' if class_val == 0 else '#DC143C'
            ),
            row=1, col=1
        )
    
    # Log scale box plot
    for class_val in [0, 1]:
        class_data = data[data['Class'] == class_val]['Amount']
        # Add small constant to avoid log(0)
        log_data = np.log1p(class_data)
        fig.add_trace(
            go.Box(
                y=log_data,
                name=f'Class {class_val} (Log)',
                boxpoints='outliers',
                marker_color='#2E8B57' if class_val == 0 else '#DC143C'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title='Transaction Amount Distribution Analysis',
        title_x=0.5,
        height=500
    )
    
    return fig


def plot_feature_importance_correlation(data: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create a bar plot showing feature correlations with the target variable.
    
    Args:
        data (pd.DataFrame): The credit card dataset
        top_n (int): Number of top features to display
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Calculate correlations with target
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('Class')
    
    correlations = data[numerical_cols].corrwith(data['Class']).abs().sort_values(ascending=False)
    top_features = correlations.head(top_n)
    
    fig = go.Figure(data=go.Bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        marker_color='#4682B4'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Features by Correlation with Fraud',
        title_x=0.5,
        xaxis_title="Absolute Correlation with Class",
        yaxis_title="Features",
        height=500
    )
    
    return fig


def plot_time_series_fraud(data: pd.DataFrame) -> go.Figure:
    """
    Create a time series plot showing fraud occurrences over time.
    
    Args:
        data (pd.DataFrame): The credit card dataset
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Group by time intervals and count fraud
    time_bins = pd.cut(data['Time'], bins=50)
    fraud_by_time = data.groupby(time_bins)['Class'].sum()
    total_by_time = data.groupby(time_bins)['Class'].count()
    fraud_rate = (fraud_by_time / total_by_time * 100).fillna(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fraud_rate.index.astype(str),
        y=fraud_rate.values,
        mode='lines+markers',
        name='Fraud Rate (%)',
        line=dict(color='#DC143C', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Fraud Rate Over Time',
        title_x=0.5,
        xaxis_title="Time Intervals",
        yaxis_title="Fraud Rate (%)",
        height=400
    )
    
    return fig


def create_summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for the dataset.
    
    Args:
        data (pd.DataFrame): The credit card dataset
        
    Returns:
        pd.DataFrame: Summary statistics table
    """
    # Basic statistics
    stats = data.describe()
    
    # Add fraud-specific statistics
    fraud_stats = data[data['Class'] == 1].describe()
    non_fraud_stats = data[data['Class'] == 0].describe()
    
    # Combine statistics
    combined_stats = pd.concat([
        stats,
        fraud_stats.add_suffix('_fraud'),
        non_fraud_stats.add_suffix('_non_fraud')
    ], axis=1)
    
    return combined_stats


def plot_feature_distributions(data: pd.DataFrame, features: List[str], n_cols: int = 3) -> go.Figure:
    """
    Create distribution plots for selected features.
    
    Args:
        data (pd.DataFrame): The credit card dataset
        features (List[str]): List of features to plot
        n_cols (int): Number of columns in the subplot grid
        
    Returns:
        go.Figure: Plotly figure object
    """
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=features,
        specs=[[{"type": "histogram"} for _ in range(n_cols)] for _ in range(n_rows)]
    )
    
    for i, feature in enumerate(features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Plot for each class
        for class_val in [0, 1]:
            class_data = data[data['Class'] == class_val][feature]
            fig.add_trace(
                go.Histogram(
                    x=class_data,
                    name=f'Class {class_val}',
                    opacity=0.7,
                    marker_color='#2E8B57' if class_val == 0 else '#DC143C'
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title='Feature Distributions by Class',
        title_x=0.5,
        height=300 * n_rows,
        showlegend=False
    )
    
    return fig 