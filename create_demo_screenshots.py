#!/usr/bin/env python3
"""
Create Demo Screenshots for Credit Card Fraud Detection System

This script generates sample screenshots that simulate what the actual
application would look like for the presentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

def create_screenshots_directory():
    """Create screenshots directory"""
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
        print("✅ Created 'screenshots' directory")

def create_gui_overview():
    """Create demo screenshot for GUI overview"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Background
    ax.set_facecolor('#f0f2f6')
    fig.patch.set_facecolor('#f0f2f6')
    
    # Title
    ax.text(0.5, 0.95, '💳 Credit Card Fraud Detection System', 
            fontsize=24, fontweight='bold', ha='center', va='center',
            transform=ax.transAxes, color='#1f77b4')
    
    # Navigation sidebar
    sidebar = patches.Rectangle((0.02, 0.1), 0.25, 0.8, 
                               facecolor='white', edgecolor='#ddd', linewidth=2)
    ax.add_patch(sidebar)
    
    # Navigation items
    nav_items = ['🏠 Home', '📊 Data Overview & EDA', '🔧 Model Training', 
                 '🎯 Fraud Prediction', '📈 Model Evaluation']
    for i, item in enumerate(nav_items):
        y_pos = 0.85 - i * 0.15
        ax.text(0.15, y_pos, item, fontsize=14, ha='center', va='center',
                transform=ax.transAxes, color='#333')
    
    # Main content area
    main_area = patches.Rectangle((0.3, 0.1), 0.68, 0.8, 
                                 facecolor='white', edgecolor='#ddd', linewidth=2)
    ax.add_patch(main_area)
    
    # Content
    ax.text(0.64, 0.8, 'Welcome to the Credit Card Fraud Detection System', 
            fontsize=18, fontweight='bold', ha='center', va='center',
            transform=ax.transAxes, color='#333')
    
    # Metrics boxes
    metrics = [('Total Transactions', '284,807'), ('Fraudulent', '492'), 
               ('Legitimate', '284,315'), ('Fraud Rate', '0.17%')]
    
    for i, (label, value) in enumerate(metrics):
        x_pos = 0.35 + (i % 2) * 0.3
        y_pos = 0.6 - (i // 2) * 0.15
        
        # Metric box
        box = patches.Rectangle((x_pos-0.12, y_pos-0.05), 0.24, 0.1, 
                               facecolor='#e8f4fd', edgecolor='#1f77b4', linewidth=2)
        ax.add_patch(box)
        
        ax.text(x_pos, y_pos+0.02, value, fontsize=16, fontweight='bold', 
                ha='center', va='center', transform=ax.transAxes, color='#1f77b4')
        ax.text(x_pos, y_pos-0.02, label, fontsize=12, ha='center', va='center',
                transform=ax.transAxes, color='#666')
    
    # Fraud distribution chart placeholder
    chart_area = patches.Rectangle((0.35, 0.15), 0.6, 0.35, 
                                  facecolor='#f8f9fa', edgecolor='#ddd', linewidth=1)
    ax.add_patch(chart_area)
    ax.text(0.65, 0.32, 'Fraud Distribution Chart', fontsize=14, ha='center', va='center',
            transform=ax.transAxes, color='#666')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('screenshots/gui_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Created: gui_overview.png")

def create_eda_dashboard():
    """Create demo screenshot for EDA dashboard"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('📊 Exploratory Data Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Fraud distribution pie chart
    sizes = [284315, 492]
    labels = ['Non-Fraud', 'Fraud']
    colors = ['#2E8B57', '#DC143C']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Fraud vs Non-Fraud Distribution')
    
    # Correlation heatmap
    np.random.seed(42)
    corr_data = np.random.rand(10, 10)
    im = ax2.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_title('Feature Correlation Heatmap')
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xticklabels([f'V{i}' for i in range(1, 11)], rotation=45)
    ax2.set_yticklabels([f'V{i}' for i in range(1, 11)])
    plt.colorbar(im, ax=ax2)
    
    # Amount distribution
    np.random.seed(42)
    fraud_amounts = np.random.exponential(100, 100)
    non_fraud_amounts = np.random.exponential(50, 1000)
    ax3.hist([non_fraud_amounts, fraud_amounts], bins=30, alpha=0.7, 
             label=['Non-Fraud', 'Fraud'], color=['#2E8B57', '#DC143C'])
    ax3.set_title('Transaction Amount Distribution')
    ax3.set_xlabel('Amount')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # Feature importance
    features = [f'V{i}' for i in range(1, 11)]
    importance = np.random.rand(10)
    importance = importance / importance.sum()
    ax4.barh(features, importance, color='#4682B4')
    ax4.set_title('Feature Importance')
    ax4.set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig('screenshots/eda_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Created: eda_dashboard.png")

def create_model_training():
    """Create demo screenshot for model training"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Background
    ax.set_facecolor('#f0f2f6')
    fig.patch.set_facecolor('#f0f2f6')
    
    # Title
    ax.text(0.5, 0.95, '🔧 Model Training Interface', 
            fontsize=24, fontweight='bold', ha='center', va='center',
            transform=ax.transAxes, color='#1f77b4')
    
    # Preprocessing options
    preprocess_box = patches.Rectangle((0.05, 0.7), 0.4, 0.2, 
                                      facecolor='white', edgecolor='#ddd', linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(0.25, 0.85, 'Data Preprocessing Options', fontsize=16, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    
    options = ['Class Balancing: Undersample', 'Remove Outliers: Yes', 
               'Test Size: 30%', 'Feature Scaling: StandardScaler']
    for i, option in enumerate(options):
        ax.text(0.07, 0.8 - i * 0.03, f'• {option}', fontsize=12,
                ha='left', va='center', transform=ax.transAxes)
    
    # Model configuration
    model_box = patches.Rectangle((0.55, 0.7), 0.4, 0.2, 
                                 facecolor='white', edgecolor='#ddd', linewidth=2)
    ax.add_patch(model_box)
    ax.text(0.75, 0.85, 'Model Configuration', fontsize=16, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    
    # Slider representations
    sliders = [('Logistic Regression C:', '1.0'), ('Random Forest Trees:', '100'),
               ('Max Depth:', '10'), ('Min Samples Split:', '2')]
    for i, (label, value) in enumerate(sliders):
        y_pos = 0.8 - i * 0.03
        ax.text(0.57, y_pos, f'{label} {value}', fontsize=12,
                ha='left', va='center', transform=ax.transAxes)
    
    # Training button
    button = patches.Rectangle((0.35, 0.5), 0.3, 0.08, 
                              facecolor='#28a745', edgecolor='#1e7e34', linewidth=2)
    ax.add_patch(button)
    ax.text(0.5, 0.54, '🚀 Train Models', fontsize=16, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes, color='white')
    
    # Training results
    results_box = patches.Rectangle((0.05, 0.1), 0.9, 0.35, 
                                   facecolor='white', edgecolor='#ddd', linewidth=2)
    ax.add_patch(results_box)
    ax.text(0.5, 0.4, 'Training Results', fontsize=18, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    
    # Model metrics
    models = ['Logistic Regression', 'Random Forest']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for i, model in enumerate(models):
        x_pos = 0.15 + i * 0.4
        ax.text(x_pos, 0.35, model, fontsize=14, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)
        
        for j, metric in enumerate(metrics):
            y_pos = 0.3 - j * 0.04
            value = f"{np.random.uniform(0.85, 0.98):.3f}"
            ax.text(x_pos - 0.15, y_pos, metric, fontsize=11,
                    ha='left', va='center', transform=ax.transAxes)
            ax.text(x_pos + 0.15, y_pos, value, fontsize=11, fontweight='bold',
                    ha='right', va='center', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('screenshots/model_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Created: model_training.png")

def create_prediction_interface():
    """Create demo screenshot for prediction interface"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Background
    ax.set_facecolor('#f0f2f6')
    fig.patch.set_facecolor('#f0f2f6')
    
    # Title
    ax.text(0.5, 0.95, '🎯 Fraud Prediction Interface', 
            fontsize=24, fontweight='bold', ha='center', va='center',
            transform=ax.transAxes, color='#1f77b4')
    
    # Input form
    form_box = patches.Rectangle((0.1, 0.2), 0.8, 0.6, 
                                facecolor='white', edgecolor='#ddd', linewidth=2)
    ax.add_patch(form_box)
    
    # Form fields
    fields = ['Time:', 'Amount:', 'V1:', 'V2:', 'V3:', 'V4:', 'V5:']
    values = ['0.0', '100.0', '0.5', '-0.2', '0.8', '-0.1', '0.3']
    
    for i, (field, value) in enumerate(zip(fields, values)):
        y_pos = 0.75 - i * 0.08
        
        # Field label
        ax.text(0.15, y_pos, field, fontsize=14, fontweight='bold',
                ha='left', va='center', transform=ax.transAxes)
        
        # Input box
        input_box = patches.Rectangle((0.4, y_pos-0.02), 0.4, 0.04, 
                                     facecolor='#f8f9fa', edgecolor='#ddd', linewidth=1)
        ax.add_patch(input_box)
        ax.text(0.6, y_pos, value, fontsize=12,
                ha='center', va='center', transform=ax.transAxes, color='#666')
    
    # Predict button
    button = patches.Rectangle((0.35, 0.15), 0.3, 0.08, 
                              facecolor='#007bff', edgecolor='#0056b3', linewidth=2)
    ax.add_patch(button)
    ax.text(0.5, 0.19, '🔍 Predict Fraud', fontsize=16, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes, color='white')
    
    # Prediction result
    result_box = patches.Rectangle((0.1, 0.05), 0.8, 0.08, 
                                  facecolor='#e8f5e8', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(result_box)
    ax.text(0.5, 0.09, '✅ LEGITIMATE TRANSACTION - Fraud Probability: 0.023', 
            fontsize=14, fontweight='bold', ha='center', va='center', 
            transform=ax.transAxes, color='#2e7d32')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('screenshots/prediction_interface.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Created: prediction_interface.png")

def create_model_evaluation():
    """Create demo screenshot for model evaluation"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('📈 Model Evaluation Dashboard', fontsize=20, fontweight='bold')
    
    # Confusion matrix
    cm = np.array([[850, 50], [20, 80]])
    im1 = ax1.imshow(cm, cmap='Blues', interpolation='nearest')
    ax1.set_title('Confusion Matrix')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Predicted\nNon-Fraud', 'Predicted\nFraud'])
    ax1.set_yticklabels(['Actual\nNon-Fraud', 'Actual\nFraud'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16, fontweight='bold')
    
    # ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = 0.9 * fpr + 0.1 * np.random.rand(100)
    tpr = np.clip(tpr, 0, 1)
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = 0.92)')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    ax2.set_title('ROC Curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Model comparison
    models = ['Logistic\nRegression', 'Random\nForest']
    accuracy = [0.89, 0.93]
    precision = [0.85, 0.91]
    recall = [0.82, 0.88]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax3.bar(x - width, accuracy, width, label='Accuracy', color='#1f77b4')
    ax3.bar(x, precision, width, label='Precision', color='#ff7f0e')
    ax3.bar(x + width, recall, width, label='Recall', color='#2ca02c')
    
    ax3.set_title('Model Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend()
    ax3.set_ylabel('Score')
    
    # Feature importance
    features = [f'V{i}' for i in range(1, 11)]
    importance = np.random.rand(10)
    importance = importance / importance.sum()
    ax4.barh(features, importance, color='#4682B4')
    ax4.set_title('Feature Importance (Random Forest)')
    ax4.set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig('screenshots/model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Created: model_evaluation.png")

def create_results_dashboard():
    """Create demo screenshot for results dashboard"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Background
    ax.set_facecolor('#f0f2f6')
    fig.patch.set_facecolor('#f0f2f6')
    
    # Title
    ax.text(0.5, 0.95, '📊 Results Dashboard', 
            fontsize=24, fontweight='bold', ha='center', va='center',
            transform=ax.transAxes, color='#1f77b4')
    
    # Results table
    table_box = patches.Rectangle((0.1, 0.3), 0.8, 0.5, 
                                 facecolor='white', edgecolor='#ddd', linewidth=2)
    ax.add_patch(table_box)
    
    # Table headers
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    for i, header in enumerate(headers):
        x_pos = 0.12 + i * 0.15
        ax.text(x_pos, 0.75, header, fontsize=14, fontweight='bold',
                ha='center', va='center', transform=ax.transAxes)
    
    # Table data
    models = ['Logistic Regression', 'Random Forest']
    data = [
        [0.89, 0.85, 0.82, 0.83, 0.91],
        [0.93, 0.91, 0.88, 0.89, 0.95]
    ]
    
    for i, model in enumerate(models):
        y_pos = 0.65 - i * 0.15
        ax.text(0.12, y_pos, model, fontsize=12, fontweight='bold',
                ha='left', va='center', transform=ax.transAxes)
        
        for j, value in enumerate(data[i]):
            x_pos = 0.12 + (j + 1) * 0.15
            ax.text(x_pos, y_pos, f'{value:.3f}', fontsize=12,
                    ha='center', va='center', transform=ax.transAxes)
    
    # Summary metrics
    summary_box = patches.Rectangle((0.1, 0.1), 0.8, 0.15, 
                                   facecolor='#e8f5e8', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(summary_box)
    
    summary_text = [
        '✅ High accuracy and recall for fraud detection',
        '✅ Real-time prediction capability',
        '✅ User-friendly interface for non-technical users'
    ]
    
    for i, text in enumerate(summary_text):
        y_pos = 0.2 - i * 0.04
        ax.text(0.12, y_pos, text, fontsize=12,
                ha='left', va='center', transform=ax.transAxes, color='#2e7d32')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('screenshots/results_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Created: results_dashboard.png")

def main():
    """Main function to create all demo screenshots"""
    print("🎯 Creating Demo Screenshots for Credit Card Fraud Detection System")
    print("="*70)
    
    # Create directory
    create_screenshots_directory()
    
    # Create all screenshots
    create_gui_overview()
    create_eda_dashboard()
    create_model_training()
    create_prediction_interface()
    create_model_evaluation()
    create_results_dashboard()
    
    print("\n" + "="*70)
    print("✅ All demo screenshots created successfully!")
    print("📁 Location: screenshots/ directory")
    print("\n🚀 Next step: Generate PDF with screenshots")
    print("   Run: python fraud_detection_presentation_with_screenshots.py")

if __name__ == "__main__":
    main() 