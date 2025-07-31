#!/bin/bash
# Automated Screenshot Capture Script for Credit Card Fraud Detection System

echo "📸 Starting automated screenshot capture..."

# Create screenshots directory
mkdir -p screenshots

# Wait for user to navigate to each page
echo "Please navigate to each page in your browser and press Enter when ready"

echo "1. Navigate to Home page and press Enter..."
read
scrot screenshots/gui_overview.png

echo "2. Navigate to 'Data Overview & EDA' page and press Enter..."
read  
scrot screenshots/eda_dashboard.png

echo "3. Navigate to 'Model Training' page and press Enter..."
read
scrot screenshots/model_training.png

echo "4. Navigate to 'Fraud Prediction' page and press Enter..."
read
scrot screenshots/prediction_interface.png

echo "5. Navigate to 'Model Evaluation' page and press Enter..."
read
scrot screenshots/model_evaluation.png

echo "6. Show results and press Enter..."
read
scrot screenshots/results_dashboard.png

echo "✅ All screenshots captured!"
ls -la screenshots/
