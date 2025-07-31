from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Frame, Spacer, Image
import os

SLIDES = [
    {
        "title": "Credit Card Fraud Detection System",
        "subtitle": "Detecting Fraud with Machine Learning & Streamlit",
        "footer": "Your Name    |    Date",
        "image": None,
    },
    {
        "title": "Project Overview",
        "bullets": [
            "Goal: Detect fraudulent credit card transactions using machine learning.",
            "Features:",
            "- Data analysis & visualization",
            "- Model training & evaluation",
            "- Real-time fraud prediction",
            "- User-friendly Streamlit GUI",
        ],
        "image": None,
    },
    {
        "title": "Dataset",
        "bullets": [
            "Source: Kaggle Credit Card Fraud Dataset",
            "Size: 284,807 transactions, 492 frauds (0.17%)",
            "Features:",
            "- Time, Amount",
            "- V1–V28 (PCA components)",
            "- Class (0: Non-Fraud, 1: Fraud)",
        ],
        "image": None,
    },
    {
        "title": "Tech Stack",
        "bullets": [
            "Python 3",
            "pandas, numpy",
            "scikit-learn",
            "matplotlib, seaborn, plotly",
            "Streamlit (GUI)",
        ],
        "image": None,
    },
    {
        "title": "Exploratory Data Analysis (EDA)",
        "bullets": [
            "Fraud vs. Non-Fraud distribution",
            "Correlation heatmap",
            "Boxplots for amount",
            "Feature distributions",
            "Time-based fraud analysis",
        ],
        "image": "screenshots/eda_dashboard.png",
        "image_caption": "EDA Dashboard - Fraud Distribution & Correlation Analysis",
    },
    {
        "title": "Data Preprocessing",
        "bullets": [
            "Handle missing values",
            "Feature scaling (StandardScaler)",
            "Outlier removal (optional)",
            "Class balancing (under/over-sampling)",
            "Train/test split (70/30)",
        ],
        "image": None,
    },
    {
        "title": "Model Building",
        "bullets": [
            "Logistic Regression",
            "Random Forest Classifier",
            "Hyperparameter tuning via GUI",
            "Model training and saving",
        ],
        "image": "screenshots/model_training.png",
        "image_caption": "Model Training Interface - Hyperparameter Configuration",
    },
    {
        "title": "Model Evaluation",
        "bullets": [
            "Metrics:",
            "- Accuracy",
            "- Precision",
            "- Recall",
            "- F1-Score",
            "- ROC-AUC",
            "Visuals:",
            "- Confusion Matrix",
            "- ROC Curve",
            "- Precision-Recall Curve",
            "- Feature Importance",
        ],
        "image": "screenshots/model_evaluation.png",
        "image_caption": "Model Evaluation - Confusion Matrix & ROC Curves",
    },
    {
        "title": "Streamlit GUI Demo",
        "bullets": [
            "Data overview & EDA",
            "Model training",
            "Fraud prediction (single, batch, random)",
            "Model evaluation",
        ],
        "image": "screenshots/gui_overview.png",
        "image_caption": "Streamlit GUI - Main Dashboard",
    },
    {
        "title": "Fraud Prediction Interface",
        "bullets": [
            "Single transaction prediction",
            "Batch processing with CSV upload",
            "Random sample testing",
            "Real-time confidence scores",
        ],
        "image": "screenshots/prediction_interface.png",
        "image_caption": "Fraud Prediction - Real-time Detection Interface",
    },
    {
        "title": "Results",
        "bullets": [
            "High accuracy and recall for fraud detection",
            "Real-time prediction capability",
            "User-friendly interface for non-technical users",
        ],
        "image": "screenshots/results_dashboard.png",
        "image_caption": "Results Dashboard - Performance Metrics",
    },
    {
        "title": "Challenges & Solutions",
        "bullets": [
            "Imbalanced data: Used resampling techniques",
            "Feature anonymization: Relied on statistical patterns",
            "Real-time prediction: Optimized preprocessing pipeline",
        ],
        "image": None,
    },
    {
        "title": "Future Work",
        "bullets": [
            "Add more ML models (XGBoost, Neural Networks)",
            "Deploy as a web service (Docker, cloud)",
            "Integrate with real-time transaction streams",
            "Advanced feature engineering",
        ],
        "image": None,
    },
    {
        "title": "Conclusion",
        "bullets": [
            "End-to-end fraud detection system",
            "Modular, extensible, and user-friendly",
            "Ready for real-world applications",
        ],
        "image": None,
    },
    {
        "title": "Q&A",
        "bullets": [
            "Questions?",
            "Thank you!",
        ],
        "image": None,
    },
]

def draw_slide(c, title, bullets=None, subtitle=None, footer=None, image_path=None, image_caption=None):
    width, height = landscape(A4)
    margin = 0.7 * inch
    
    # Background
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, fill=1)
    
    # Title
    c.setFillColor(colors.HexColor("#1f77b4"))
    c.setFont("Helvetica-Bold", 36)
    c.drawString(margin, height - margin - 20, title)
    
    # Subtitle
    if subtitle:
        c.setFont("Helvetica", 22)
        c.setFillColor(colors.HexColor("#444444"))
        c.drawString(margin, height - margin - 60, subtitle)
    
    # Content area
    content_start_y = height - margin - 100
    content_width = width - 2 * margin
    
    # Image (if provided)
    if image_path and os.path.exists(image_path):
        try:
            img = Image(image_path)
            img.drawHeight = 3 * inch
            img.drawWidth = 4 * inch
            img_x = width - margin - img.drawWidth
            img_y = content_start_y - img.drawHeight
            
            # Draw image
            img.drawOn(c, img_x, img_y)
            
            # Image caption
            if image_caption:
                c.setFont("Helvetica-Oblique", 12)
                c.setFillColor(colors.HexColor("#666666"))
                c.drawString(img_x, img_y - 20, image_caption)
            
            # Adjust content width for image
            content_width = img_x - margin - 20
        except:
            pass
    else:
        # Draw placeholder for image
        if image_path:
            placeholder_x = width - margin - 4 * inch
            placeholder_y = content_start_y - 3 * inch
            c.setFillColor(colors.HexColor("#f0f0f0"))
            c.rect(placeholder_x, placeholder_y, 4 * inch, 3 * inch, fill=1)
            c.setFillColor(colors.HexColor("#999999"))
            c.setFont("Helvetica", 14)
            c.drawString(placeholder_x + 1 * inch, placeholder_y + 1.5 * inch, "[Screenshot Placeholder]")
            if image_caption:
                c.setFont("Helvetica-Oblique", 12)
                c.setFillColor(colors.HexColor("#666666"))
                c.drawString(placeholder_x, placeholder_y - 20, image_caption)
            
            # Adjust content width for placeholder
            content_width = placeholder_x - margin - 20
    
    # Bullets
    if bullets:
        styles = getSampleStyleSheet()
        bullet_style = ParagraphStyle(
            'Bullets', parent=styles['Normal'], fontSize=20, leftIndent=20, bulletIndent=10, leading=25
        )
        y = content_start_y
        for bullet in bullets:
            if bullet.startswith("-"):
                bullet = bullet[1:].strip()
                p = Paragraph(f"<bullet>&bull;</bullet> {bullet}", bullet_style)
            else:
                p = Paragraph(bullet, bullet_style)
            w, h = p.wrap(content_width, 40)
            if y - h < margin + 50:
                break
            p.drawOn(c, margin, y - h)
            y -= h + 5
    
    # Footer
    if footer:
        c.setFont("Helvetica-Oblique", 14)
        c.setFillColor(colors.HexColor("#888888"))
        c.drawRightString(width - margin, margin / 2, footer)
    
    c.showPage()

def create_screenshots_directory():
    """Create screenshots directory if it doesn't exist"""
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
        print("Created 'screenshots' directory")
        print("Please add your screenshots to this directory with the following names:")
        print("- eda_dashboard.png")
        print("- model_training.png") 
        print("- model_evaluation.png")
        print("- gui_overview.png")
        print("- prediction_interface.png")
        print("- results_dashboard.png")

def create_pdf(filename="Credit_Card_Fraud_Detection_Presentation_With_Screenshots.pdf"):
    create_screenshots_directory()
    
    c = canvas.Canvas(filename, pagesize=landscape(A4))
    for slide in SLIDES:
        draw_slide(
            c,
            slide["title"],
            bullets=slide.get("bullets"),
            subtitle=slide.get("subtitle"),
            footer=slide.get("footer"),
            image_path=slide.get("image"),
            image_caption=slide.get("image_caption"),
        )
    c.save()
    print(f"PDF created: {filename}")

if __name__ == "__main__":
    create_pdf() 