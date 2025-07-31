#!/bin/bash
# Quick Screenshot Capture Script

echo "📸 Quick Screenshot Capture for Fraud Detection System"
echo "======================================================"

echo ""
echo "Instructions:"
echo "1. Navigate to each page in your browser"
echo "2. Press Enter when ready to capture"
echo "3. Use F12 → Screenshot or Print Screen"
echo "4. Save with the exact filename shown"
echo ""

read -p "Press Enter when you're on the HOME page..."
echo "📸 Capture: screenshots/gui_overview.png"

read -p "Press Enter when you're on the EDA page..."
echo "📸 Capture: screenshots/eda_dashboard.png"

read -p "Press Enter when you're on the MODEL TRAINING page..."
echo "📸 Capture: screenshots/model_training.png"

read -p "Press Enter when you're on the PREDICTION page..."
echo "📸 Capture: screenshots/prediction_interface.png"

read -p "Press Enter when you're on the EVALUATION page..."
echo "📸 Capture: screenshots/model_evaluation.png"

read -p "Press Enter when you're showing RESULTS..."
echo "📸 Capture: screenshots/results_dashboard.png"

echo ""
echo "✅ All screenshots captured!"
echo "Now run: python fraud_detection_presentation_with_screenshots.py"
