#!/usr/bin/env python3
"""
Simple Screenshot Guide for Credit Card Fraud Detection System

This provides easy-to-follow instructions for capturing screenshots
using built-in browser and system tools.
"""

import os
import webbrowser
import time

def create_screenshots_directory():
    """Create screenshots directory"""
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
        print("✅ Created 'screenshots' directory")

def print_simple_instructions():
    """Print simple screenshot instructions"""
    print("\n" + "="*70)
    print("📸 SIMPLE SCREENSHOT CAPTURE GUIDE")
    print("="*70)
    
    print("\n🎯 **Your Streamlit app is running at:**")
    print("   🌐 http://localhost:8501")
    
    print("\n📱 **How to Take Screenshots:**")
    print("1. **Browser Method (Recommended):**")
    print("   • Press F12 to open Developer Tools")
    print("   • Press Ctrl+Shift+P (Cmd+Shift+P on Mac)")
    print("   • Type 'screenshot' and select 'Capture full size screenshot'")
    print("   • Save with the exact filename shown below")
    
    print("\n2. **System Method:**")
    print("   • Press Print Screen key")
    print("   • Or use Snipping Tool (Windows) / Screenshot (Mac)")
    print("   • Save with the exact filename shown below")
    
    print("\n3. **Browser Extension Method:**")
    print("   • Install 'Full Page Screenshot' extension")
    print("   • Click the extension icon")
    print("   • Save with the exact filename shown below")

def print_screenshot_list():
    """Print the list of required screenshots"""
    print("\n📋 **Required Screenshots:**")
    print("="*50)
    
    screenshots = [
        {
            "filename": "gui_overview.png",
            "page": "Home",
            "description": "Main dashboard with welcome message and navigation",
            "what_to_show": "Show the home page with fraud distribution chart"
        },
        {
            "filename": "eda_dashboard.png",
            "page": "Data Overview & EDA",
            "description": "Exploratory data analysis page",
            "what_to_show": "Show fraud distribution pie chart and correlation heatmap"
        },
        {
            "filename": "model_training.png",
            "page": "Model Training",
            "description": "Model training interface",
            "what_to_show": "Show preprocessing options and hyperparameter sliders"
        },
        {
            "filename": "prediction_interface.png",
            "page": "Fraud Prediction",
            "description": "Prediction interface",
            "what_to_show": "Show single transaction prediction form"
        },
        {
            "filename": "model_evaluation.png",
            "page": "Model Evaluation",
            "description": "Model evaluation page",
            "what_to_show": "Show confusion matrix and ROC curves"
        },
        {
            "filename": "results_dashboard.png",
            "page": "Model Evaluation",
            "description": "Results and metrics",
            "what_to_show": "Show model comparison table and performance metrics"
        }
    ]
    
    for i, screenshot in enumerate(screenshots, 1):
        print(f"\n{i}. **{screenshot['filename']}**")
        print(f"   📄 Page: {screenshot['page']}")
        print(f"   📝 Description: {screenshot['description']}")
        print(f"   🎯 What to show: {screenshot['what_to_show']}")
        print(f"   💾 Save as: screenshots/{screenshot['filename']}")

def open_browser():
    """Open the Streamlit app in browser"""
    print("\n🌐 **Opening your app in browser...**")
    try:
        webbrowser.open("http://localhost:8501")
        print("✅ Browser opened!")
    except:
        print("❌ Could not open browser automatically")
        print("   Please manually open: http://localhost:8501")

def generate_quick_script():
    """Generate a quick capture script"""
    script_content = '''#!/bin/bash
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
'''
    
    with open("quick_capture.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("quick_capture.sh", 0o755)
    print("✅ Created quick capture script: quick_capture.sh")

def main():
    """Main function"""
    print("🎯 Credit Card Fraud Detection System - Simple Screenshot Guide")
    print("="*70)
    
    # Create directory
    create_screenshots_directory()
    
    # Print instructions
    print_simple_instructions()
    
    # Print screenshot list
    print_screenshot_list()
    
    # Generate quick script
    generate_quick_script()
    
    # Open browser
    open_browser()
    
    print("\n" + "="*70)
    print("🚀 **READY TO CAPTURE!**")
    print("="*70)
    print("1. Your browser should now be open to the app")
    print("2. Navigate through each page")
    print("3. Take screenshots using the methods above")
    print("4. Save them in the 'screenshots' folder")
    print("5. Run: python fraud_detection_presentation_with_screenshots.py")
    
    print("\n💡 **Pro Tips:**")
    print("• Use browser zoom (Ctrl/Cmd +) for better quality")
    print("• Capture full page when possible")
    print("• Make sure text is readable")
    print("• Test the PDF after adding each screenshot")
    
    print("\n🎉 **You're all set! Happy screenshotting!**")

if __name__ == "__main__":
    main() 