#!/usr/bin/env python3
"""
Screenshot Capture Guide for Credit Card Fraud Detection System

This script provides instructions and automation for capturing screenshots
from the running Streamlit application for the presentation.
"""

import os
import subprocess
import time
from datetime import datetime

def create_screenshots_directory():
    """Create screenshots directory if it doesn't exist"""
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
        print("✅ Created 'screenshots' directory")
    else:
        print("✅ Screenshots directory already exists")

def print_capture_instructions():
    """Print detailed instructions for capturing screenshots"""
    print("\n" + "="*60)
    print("📸 SCREENSHOT CAPTURE INSTRUCTIONS")
    print("="*60)
    
    print("\n🔧 **Prerequisites:**")
    print("1. Make sure your Streamlit app is running at http://localhost:8501")
    print("2. Have your browser open and ready")
    print("3. Use browser developer tools or screenshot tools")
    
    print("\n📱 **Screenshot Tools Options:**")
    print("• Browser: F12 → Screenshot (Chrome/Firefox)")
    print("• System: Print Screen key or Snipping Tool")
    print("• Browser Extension: Full Page Screenshot")
    print("• Command Line: scrot (Linux) or screencapture (Mac)")
    
    print("\n🎯 **Required Screenshots:**")
    
    screenshots = [
        {
            "filename": "gui_overview.png",
            "description": "Home page with overview and navigation",
            "url": "http://localhost:8501",
            "instructions": "Take screenshot of the main home page showing the welcome message and navigation"
        },
        {
            "filename": "eda_dashboard.png", 
            "description": "EDA page with fraud distribution and correlation analysis",
            "url": "http://localhost:8501",
            "instructions": "Go to 'Data Overview & EDA' page, show fraud distribution pie chart and correlation heatmap"
        },
        {
            "filename": "model_training.png",
            "description": "Model training page with hyperparameter configuration",
            "url": "http://localhost:8501", 
            "instructions": "Go to 'Model Training' page, show the preprocessing options and model configuration sliders"
        },
        {
            "filename": "prediction_interface.png",
            "description": "Fraud prediction page with input forms",
            "url": "http://localhost:8501",
            "instructions": "Go to 'Fraud Prediction' page, show the single transaction prediction form"
        },
        {
            "filename": "model_evaluation.png",
            "description": "Model evaluation page with metrics and visualizations",
            "url": "http://localhost:8501",
            "instructions": "Go to 'Model Evaluation' page, show confusion matrix and ROC curves"
        },
        {
            "filename": "results_dashboard.png",
            "description": "Results and performance metrics",
            "url": "http://localhost:8501",
            "instructions": "Show model comparison table and performance metrics"
        }
    ]
    
    for i, screenshot in enumerate(screenshots, 1):
        print(f"\n{i}. **{screenshot['filename']}**")
        print(f"   Description: {screenshot['description']}")
        print(f"   Instructions: {screenshot['instructions']}")
        print(f"   Save as: screenshots/{screenshot['filename']}")

def check_streamlit_running():
    """Check if Streamlit is running"""
    try:
        import requests
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit app is running at http://localhost:8501")
            return True
        else:
            print("❌ Streamlit app is not responding properly")
            return False
    except:
        print("❌ Streamlit app is not running")
        print("   Please start it with: streamlit run app.py")
        return False

def generate_screenshot_script():
    """Generate a shell script for automated screenshot capture (Linux)"""
    script_content = '''#!/bin/bash
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
'''
    
    with open("capture_screenshots.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("capture_screenshots.sh", 0o755)
    print("✅ Created automated screenshot script: capture_screenshots.sh")
    print("   Run it with: ./capture_screenshots.sh")

def main():
    """Main function"""
    print("🎯 Credit Card Fraud Detection System - Screenshot Capture Guide")
    print("="*70)
    
    # Create directory
    create_screenshots_directory()
    
    # Check if Streamlit is running
    if not check_streamlit_running():
        print("\n🚀 To start the app, run:")
        print("   streamlit run app.py")
        print("\nThen come back and run this script again.")
        return
    
    # Print instructions
    print_capture_instructions()
    
    # Generate automated script
    generate_screenshot_script()
    
    print("\n" + "="*60)
    print("📋 **NEXT STEPS:**")
    print("="*60)
    print("1. Follow the instructions above to capture screenshots")
    print("2. Save them in the 'screenshots' directory with the exact filenames")
    print("3. Run: python fraud_detection_presentation_with_screenshots.py")
    print("4. Your PDF will be generated with the screenshots included!")
    
    print("\n💡 **Tips:**")
    print("• Use browser zoom (Ctrl/Cmd +) to make text more readable")
    print("• Capture full page screenshots when possible")
    print("• Ensure good lighting/contrast in the screenshots")
    print("• Test the PDF generation after adding each screenshot")

if __name__ == "__main__":
    main() 