#!/usr/bin/env python3
"""
Quick Start Script for Credit Card Fraud Detection System

This script helps users quickly set up and run the fraud detection system.
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit', 'pandas', 'sklearn', 'matplotlib', 
        'seaborn', 'numpy', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def check_data_file():
    """Check if creditcard.csv exists, if not generate sample data."""
    if os.path.exists('creditcard.csv'):
        print("✅ creditcard.csv found")
        return True
    
    print("❌ creditcard.csv not found")
    print("Generating sample data...")
    
    try:
        from generate_sample_data import generate_sample_data
        df = generate_sample_data(n_samples=10000, fraud_ratio=0.0017, random_state=42)
        df.to_csv('creditcard.csv', index=False)
        print("✅ Sample data generated successfully!")
        return True
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")
        return False


def start_application():
    """Start the Streamlit application."""
    print("\n🚀 Starting Credit Card Fraud Detection System...")
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")


def main():
    """Main function to run the quick start process."""
    print("=" * 60)
    print("💳 Credit Card Fraud Detection System - Quick Start")
    print("=" * 60)
    
    # Check if we're in the right directory
    required_files = ['app.py', 'data_loader.py', 'eda.py', 'preprocess.py', 'model.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please make sure you're in the correct project directory.")
        return
    
    print("✅ All required files found")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        return
    
    # Check data file
    print("\n📊 Checking data file...")
    if not check_data_file():
        return
    
    # Start application
    start_application()


if __name__ == "__main__":
    main() 