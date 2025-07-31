# Credit Card Fraud Detection System

A comprehensive Python application for detecting credit card fraud using machine learning with a beautiful Streamlit GUI.

## 🚀 Features

- **Interactive GUI**: Modern Streamlit interface with intuitive navigation
- **Exploratory Data Analysis**: Comprehensive data visualization and analysis
- **Machine Learning Models**: Logistic Regression and Random Forest classifiers
- **Real-time Prediction**: Single transaction and batch prediction capabilities
- **Model Evaluation**: Detailed performance metrics and visualizations
- **Data Preprocessing**: Automated data cleaning, scaling, and balancing

## 📋 Requirements

- Python 3.8+
- pandas
- scikit-learn
- matplotlib
- seaborn
- streamlit
- numpy
- plotly
## unzip
 before run this project unzip  "proj.zip" file

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Place your `creditcard.csv` file in the project directory
   - If you don't have the file, the app will use sample data for demonstration

## 🎯 Usage

### Starting the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Application Navigation

The app consists of 5 main pages:

#### 🏠 Home
- Overview of the dataset
- Key statistics and metrics
- Quick fraud distribution visualization
- Usage instructions

#### 📊 Data Overview & EDA
- **Basic Statistics**: Summary statistics and transaction amount analysis
- **Correlation Analysis**: Feature correlation heatmap and importance ranking
- **Feature Distributions**: Interactive feature distribution plots
- **Time Analysis**: Fraud patterns over time

#### 🔧 Model Training
- Configure data preprocessing options
- Set model hyperparameters
- Train Logistic Regression and Random Forest models
- View training results and metrics

#### 🎯 Fraud Prediction
- **Single Transaction**: Input individual transaction data for prediction
- **Batch Prediction**: Upload CSV file for bulk predictions
- **Random Sample**: Test predictions on random samples from test data

#### 📈 Model Evaluation
- Compare model performance
- View detailed evaluation metrics
- Interactive visualizations:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - Feature Importance

## 📁 Project Structure

```
fraud-detection-system/
├── app.py                 # Main Streamlit application
├── data_loader.py         # Data loading and basic operations
├── eda.py                 # Exploratory data analysis functions
├── preprocess.py          # Data preprocessing pipeline
├── model.py               # Machine learning models and evaluation
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── creditcard.csv        # Dataset (not included)
```

## 🔧 Data Format

The application expects a CSV file with the following columns:
- `Time`: Time elapsed between transactions
- `Amount`: Transaction amount
- `V1-V28`: PCA transformed features (anonymized)
- `Class`: Target variable (0: Non-Fraud, 1: Fraud)

## 🎨 Features in Detail

### Data Preprocessing
- **Missing Value Handling**: Automatic detection and imputation
- **Outlier Removal**: IQR and Z-score methods
- **Feature Scaling**: StandardScaler for numerical features
- **Class Balancing**: Undersampling and oversampling options
- **Train-Test Split**: Configurable split ratio with stratification

### Machine Learning Models
- **Logistic Regression**: Linear model with regularization
- **Random Forest**: Ensemble method with configurable parameters
- **Hyperparameter Tuning**: Interactive parameter adjustment
- **Model Persistence**: Save and load trained models

### Visualization Features
- **Interactive Plots**: Plotly-based visualizations
- **Real-time Updates**: Dynamic charts that update with data changes
- **Multiple Chart Types**: Heatmaps, box plots, histograms, line charts
- **Responsive Design**: Adapts to different screen sizes

### Prediction Capabilities
- **Single Transaction**: Real-time fraud detection for individual transactions
- **Batch Processing**: Handle multiple transactions efficiently
- **Probability Scores**: Confidence levels for predictions
- **Export Results**: Download prediction results as CSV

## 📊 Model Performance

The system provides comprehensive evaluation metrics:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

## 🎯 Use Cases

- **Financial Institutions**: Detect fraudulent credit card transactions
- **E-commerce Platforms**: Monitor payment processing for fraud
- **Data Scientists**: Explore fraud detection techniques
- **Students**: Learn machine learning and data science concepts

## 🔒 Security Features

- **Data Anonymization**: Uses PCA-transformed features
- **No Sensitive Data**: No actual credit card information is stored
- **Local Processing**: All computations run locally
- **Sample Data**: Fallback to generated sample data when real data unavailable

## 🚀 Getting Started

1. **Install the application** (see Installation section above)

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Explore the data**:
   - Start with the "Data Overview & EDA" page
   - Understand the dataset structure and patterns

4. **Train models**:
   - Go to "Model Training" page
   - Configure preprocessing options
   - Adjust model parameters
   - Train both models

5. **Make predictions**:
   - Use "Fraud Prediction" page
   - Try different prediction methods
   - Analyze results

6. **Evaluate performance**:
   - Visit "Model Evaluation" page
   - Compare model performance
   - View detailed metrics and visualizations

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Adding new machine learning models

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Kaggle for providing the credit card fraud dataset
- Streamlit for the excellent web application framework
- Scikit-learn for the machine learning algorithms
- Plotly for the interactive visualizations

## 📞 Support

If you encounter any issues or have questions:
1. Check the documentation above
2. Review the code comments for guidance
3. Ensure all dependencies are properly installed
4. Verify your data format matches the expected structure

---

**Happy Fraud Detection! 🕵️‍♂️💳** 
