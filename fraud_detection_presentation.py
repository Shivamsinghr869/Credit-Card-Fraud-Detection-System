from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Frame, Spacer

SLIDES = [
    {
        "title": "Credit Card Fraud Detection System",
        "subtitle": "Detecting Fraud with Machine Learning & Streamlit",
        "footer": "Your Name    |    Date",
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
    },
    {
        "title": "Exploratory Data Analysis (EDA)",
        "bullets": [
            "Fraud vs. Non-Fraud distribution",
            "Correlation heatmap",
            "Boxplots for amount",
            "Feature distributions",
            "Time-based fraud analysis",
            "[Insert EDA screenshots here]",
        ],
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
    },
    {
        "title": "Model Building",
        "bullets": [
            "Logistic Regression",
            "Random Forest Classifier",
            "Hyperparameter tuning via GUI",
            "Model training and saving",
        ],
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
    },
    {
        "title": "Streamlit GUI Demo",
        "bullets": [
            "Data overview & EDA",
            "Model training",
            "Fraud prediction (single, batch, random)",
            "Model evaluation",
            "[Insert GUI screenshots or demo video link]",
        ],
    },
    {
        "title": "Results",
        "bullets": [
            "High accuracy and recall for fraud detection",
            "Real-time prediction capability",
            "User-friendly interface for non-technical users",
        ],
    },
    {
        "title": "Challenges & Solutions",
        "bullets": [
            "Imbalanced data: Used resampling techniques",
            "Feature anonymization: Relied on statistical patterns",
            "Real-time prediction: Optimized preprocessing pipeline",
        ],
    },
    {
        "title": "Future Work",
        "bullets": [
            "Add more ML models (XGBoost, Neural Networks)",
            "Deploy as a web service (Docker, cloud)",
            "Integrate with real-time transaction streams",
            "Advanced feature engineering",
        ],
    },
    {
        "title": "Conclusion",
        "bullets": [
            "End-to-end fraud detection system",
            "Modular, extensible, and user-friendly",
            "Ready for real-world applications",
        ],
    },
    {
        "title": "Q&A",
        "bullets": [
            "Questions?",
            "Thank you!",
        ],
    },
]

def draw_slide(c, title, bullets=None, subtitle=None, footer=None):
    width, height = landscape(A4)
    margin = 0.7 * inch
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, fill=1)
    c.setFillColor(colors.HexColor("#1f77b4"))
    c.setFont("Helvetica-Bold", 36)
    c.drawString(margin, height - margin - 20, title)
    if subtitle:
        c.setFont("Helvetica", 22)
        c.setFillColor(colors.HexColor("#444444"))
        c.drawString(margin, height - margin - 60, subtitle)
    if bullets:
        styles = getSampleStyleSheet()
        bullet_style = ParagraphStyle(
            'Bullets', parent=styles['Normal'], fontSize=22, leftIndent=30, bulletIndent=10, leading=30
        )
        y = height - margin - 100
        for bullet in bullets:
            if bullet.startswith("-"):
                bullet = bullet[1:].strip()
                p = Paragraph(f"<bullet>&bull;</bullet> {bullet}", bullet_style)
            else:
                p = Paragraph(bullet, bullet_style)
            w, h = p.wrap(width - 2 * margin, 40)
            if y - h < margin:
                break
            p.drawOn(c, margin, y - h)
            y -= h + 5
    if footer:
        c.setFont("Helvetica-Oblique", 14)
        c.setFillColor(colors.HexColor("#888888"))
        c.drawRightString(width - margin, margin / 2, footer)
    c.showPage()

def create_pdf(filename="Credit_Card_Fraud_Detection_Presentation.pdf"):
    c = canvas.Canvas(filename, pagesize=landscape(A4))
    for slide in SLIDES:
        draw_slide(
            c,
            slide["title"],
            bullets=slide.get("bullets"),
            subtitle=slide.get("subtitle"),
            footer=slide.get("footer"),
        )
    c.save()

if __name__ == "__main__":
    create_pdf()