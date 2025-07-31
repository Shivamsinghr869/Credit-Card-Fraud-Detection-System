"""
Sample Data Generator for Credit Card Fraud Detection

This script generates sample credit card transaction data similar to the Kaggle dataset
for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta


def generate_sample_data(n_samples=10000, fraud_ratio=0.0017, random_state=42):
    """
    Generate sample credit card transaction data.
    
    Args:
        n_samples (int): Number of samples to generate
        fraud_ratio (float): Ratio of fraudulent transactions
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated sample data
    """
    np.random.seed(random_state)
    
    # Generate base features
    data = {}
    
    # Time feature (seconds elapsed since first transaction)
    start_time = datetime.now() - timedelta(days=2)
    time_seconds = np.random.uniform(0, 172792, n_samples)  # ~2 days in seconds
    data['Time'] = time_seconds
    
    # Amount feature (exponential distribution similar to real data)
    data['Amount'] = np.random.exponential(88.35, n_samples)
    
    # V1-V28 features (PCA transformed features - normally distributed)
    for i in range(1, 29):
        # Different distributions for different features to simulate real patterns
        if i in [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        elif i == 4:
            # V4 has some correlation with fraud
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        elif i == 8:
            # V8 has some correlation with fraud
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Class (fraud indicator) - very imbalanced
    fraud_count = int(n_samples * fraud_ratio)
    non_fraud_count = n_samples - fraud_count
    
    # Create class array with exact counts
    classes = np.concatenate([np.zeros(non_fraud_count), np.ones(fraud_count)])
    np.random.shuffle(classes)
    data['Class'] = classes
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic patterns for fraud transactions
    fraud_indices = df[df['Class'] == 1].index
    
    if len(fraud_indices) > 0:
        # Fraud transactions tend to have higher amounts
        df.loc[fraud_indices, 'Amount'] = df.loc[fraud_indices, 'Amount'] * np.random.uniform(1.5, 3.0, len(fraud_indices))
        
        # Some V features have different patterns for fraud
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
            if np.random.random() < 0.3:  # 30% chance to modify feature for fraud
                df.loc[fraud_indices, f'V{i}'] = df.loc[fraud_indices, f'V{i}'] + np.random.normal(0, 0.5, len(fraud_indices))
    
    return df


def main():
    """Main function to generate and save sample data."""
    parser = argparse.ArgumentParser(description='Generate sample credit card fraud data')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--fraud-ratio', type=float, default=0.0017, help='Ratio of fraudulent transactions')
    parser.add_argument('--output', type=str, default='creditcard.csv', help='Output file name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Generating {args.samples} samples with {args.fraud_ratio*100:.3f}% fraud ratio...")
    
    # Generate data
    df = generate_sample_data(
        n_samples=args.samples,
        fraud_ratio=args.fraud_ratio,
        random_state=args.seed
    )
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    # Print summary
    print(f"Data saved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Fraud count: {df['Class'].sum()}")
    print(f"Non-fraud count: {len(df) - df['Class'].sum()}")
    print(f"Fraud percentage: {(df['Class'].sum() / len(df)) * 100:.3f}%")
    
    # Show sample
    print("\nSample data:")
    print(df.head())
    
    # Show statistics
    print("\nBasic statistics:")
    print(df.describe())


if __name__ == "__main__":
    main() 