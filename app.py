"""
Credit Card Fraud Detection System - Streamlit Application

This is the main Streamlit application for the credit card fraud detection system.
It provides a comprehensive GUI for data analysis, model training, and fraud prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Import custom modules
from data_loader import DataLoader, load_sample_data
from eda import (
    plot_fraud_distribution, plot_correlation_heatmap, plot_amount_distribution,
    plot_feature_importance_correlation, plot_time_series_fraud, create_summary_statistics,
    plot_feature_distributions
)
from preprocess import DataPreprocessor
from model import (
    FraudDetectionModel, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_feature_importance, compare_models
)

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .safe-transaction {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Overview & EDA", "🔧 Model Training", "🎯 Fraud Prediction", "📈 Model Evaluation"]
    )
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            data = data_loader.load_data()
            if data.empty:
                st.warning("Could not load 'creditcard.csv'. Using sample data for demonstration.")
                data = load_sample_data()
            
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.session_state.data_loader = data_loader
    
    data = st.session_state.data
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(data)
    elif page == "📊 Data Overview & EDA":
        show_eda_page(data)
    elif page == "🔧 Model Training":
        show_training_page(data)
    elif page == "🎯 Fraud Prediction":
        show_prediction_page()
    elif page == "📈 Model Evaluation":
        show_evaluation_page()

def show_home_page(data):
    """Display the home page with overview information."""
    
    st.markdown("## Welcome to the Credit Card Fraud Detection System")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(data):,}")
    
    with col2:
        fraud_count = data['Class'].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")
    
    with col3:
        non_fraud_count = len(data) - fraud_count
        st.metric("Legitimate Transactions", f"{non_fraud_count:,}")
    
    with col4:
        fraud_percentage = (fraud_count / len(data)) * 100
        st.metric("Fraud Rate", f"{fraud_percentage:.3f}%")
    
    # Quick overview
    st.markdown("### Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Information:**")
        info = st.session_state.data_loader.get_data_info()
        if info:
            st.write(f"- **Shape:** {info['shape']}")
            st.write(f"- **Features:** {len(info['columns']) - 1}")  # Exclude target
            st.write(f"- **Target Variable:** Class (0: Non-Fraud, 1: Fraud)")
            st.write(f"- **Missing Values:** {sum(info['missing_values'].values())}")
    
    with col2:
        st.markdown("**Key Features:**")
        st.write("- **Time:** Time elapsed between transactions")
        st.write("- **Amount:** Transaction amount")
        st.write("- **V1-V28:** PCA transformed features (anonymized)")
        st.write("- **Class:** Target variable (fraud indicator)")
    
    # Fraud distribution chart
    st.markdown("### Fraud Distribution")
    fraud_fig = plot_fraud_distribution(data)
    st.plotly_chart(fraud_fig, use_container_width=True)
    
    # Instructions
    st.markdown("### How to Use This System")
    
    st.markdown("""
    1. **📊 Data Overview & EDA**: Explore the dataset and understand patterns
    2. **🔧 Model Training**: Train Logistic Regression and Random Forest models
    3. **🎯 Fraud Prediction**: Make predictions on new transaction data
    4. **📈 Model Evaluation**: Compare model performance and view detailed metrics
    """)

def show_eda_page(data):
    """Display the Exploratory Data Analysis page."""
    
    st.markdown("## 📊 Exploratory Data Analysis")
    
    # Tabs for different EDA sections
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Basic Statistics", "🔍 Correlation Analysis", "📊 Feature Distributions", "⏰ Time Analysis"])
    
    with tab1:
        st.markdown("### Basic Statistics")
        
        # Summary statistics
        summary_stats = create_summary_statistics(data)
        st.dataframe(summary_stats, use_container_width=True)
        
        # Amount distribution
        st.markdown("### Transaction Amount Analysis")
        amount_fig = plot_amount_distribution(data)
        st.plotly_chart(amount_fig, use_container_width=True)
        
        # Sample data
        st.markdown("### Sample Data")
        sample_data = st.session_state.data_loader.get_sample_data(10)
        st.dataframe(sample_data, use_container_width=True)
    
    with tab2:
        st.markdown("### Correlation Analysis")
        
        # Correlation heatmap
        st.markdown("#### Feature Correlation Heatmap")
        corr_fig = plot_correlation_heatmap(data, max_features=15)
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Feature importance by correlation
        st.markdown("#### Top Features by Correlation with Fraud")
        feature_corr_fig = plot_feature_importance_correlation(data, top_n=15)
        st.plotly_chart(feature_corr_fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Feature Distributions")
        
        # Select features to plot
        numerical_features = st.session_state.data_loader.get_numerical_features()
        selected_features = st.multiselect(
            "Select features to visualize:",
            numerical_features,
            default=numerical_features[:6] if len(numerical_features) >= 6 else numerical_features
        )
        
        if selected_features:
            dist_fig = plot_feature_distributions(data, selected_features, n_cols=3)
            st.plotly_chart(dist_fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Time Series Analysis")
        
        # Fraud rate over time
        time_fig = plot_time_series_fraud(data)
        st.plotly_chart(time_fig, use_container_width=True)
        
        # Time statistics
        st.markdown("#### Time-based Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Fraud by Time Period:**")
            time_bins = pd.cut(data['Time'], bins=10)
            fraud_by_time = data.groupby(time_bins)['Class'].sum()
            fraud_by_time.index = fraud_by_time.index.astype(str)  # Fix for Streamlit
            st.bar_chart(fraud_by_time)
        
        with col2:
            st.markdown("**Average Amount by Time Period:**")
            avg_amount_by_time = data.groupby(time_bins)['Amount'].mean()
            avg_amount_by_time.index = avg_amount_by_time.index.astype(str)  # Fix for Streamlit
            st.line_chart(avg_amount_by_time)

def show_training_page(data):
    """Display the model training page."""
    
    st.markdown("## 🔧 Model Training")
    
    # Preprocessing options
    st.markdown("### Data Preprocessing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        balance_method = st.selectbox(
            "Class Balancing Method:",
            ["none", "undersample", "oversample"],
            help="Choose how to handle class imbalance"
        )
        
        remove_outliers = st.checkbox(
            "Remove Outliers",
            value=False,
            help="Remove outliers using IQR method"
        )
    
    with col2:
        test_size = st.slider(
            "Test Set Size:",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help="Proportion of data to use for testing"
        )
        
        outlier_columns = None
        if remove_outliers:
            numerical_features = st.session_state.data_loader.get_numerical_features()
            outlier_columns = st.multiselect(
                "Select columns for outlier removal:",
                numerical_features,
                default=['Amount'] + [f'V{i}' for i in range(1, 6)]
            )
    
    # Model selection
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Logistic Regression")
        lr_c = st.slider("Regularization (C):", 0.1, 10.0, 1.0, 0.1, key="lr_c")
        lr_penalty = st.selectbox("Penalty:", ["l2", "l1"], key="lr_penalty")
    
    with col2:
        st.markdown("#### Random Forest")
        rf_n_estimators = st.slider("Number of Trees:", 50, 200, 100, 10, key="rf_trees")
        rf_max_depth = st.slider("Max Depth:", 5, 20, 10, 1, key="rf_depth")
    
    # Training button
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            # Initialize preprocessor
            preprocessor = DataPreprocessor()
            
            # Preprocess data
            X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
                data=data,
                balance_method=balance_method,
                remove_outliers=remove_outliers,
                outlier_columns=outlier_columns
            )
            
            # Store preprocessed data
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.preprocessor = preprocessor
            
            # Initialize and train models
            models = {}
            
            # Logistic Regression
            lr_model = FraudDetectionModel('logistic_regression')
            lr_model.set_hyperparameters(C=lr_c, penalty=lr_penalty)
            lr_metrics = lr_model.train(X_train, y_train)
            models['Logistic Regression'] = lr_model
            
            # Random Forest
            rf_model = FraudDetectionModel('random_forest')
            rf_model.set_hyperparameters(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
            rf_metrics = rf_model.train(X_train, y_train)
            models['Random Forest'] = rf_model
            
            # Store models
            st.session_state.models = models
            st.session_state.models_trained = True
            
            # Display training results
            st.success("Models trained successfully!")
            
            # Training metrics comparison
            st.markdown("### Training Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Logistic Regression")
                st.metric("Accuracy", f"{lr_metrics['accuracy']:.4f}")
                st.metric("Precision", f"{lr_metrics['precision']:.4f}")
                st.metric("Recall", f"{lr_metrics['recall']:.4f}")
                st.metric("F1-Score", f"{lr_metrics['f1_score']:.4f}")
                st.metric("ROC-AUC", f"{lr_metrics['roc_auc']:.4f}")
            
            with col2:
                st.markdown("#### Random Forest")
                st.metric("Accuracy", f"{rf_metrics['accuracy']:.4f}")
                st.metric("Precision", f"{rf_metrics['precision']:.4f}")
                st.metric("Recall", f"{rf_metrics['recall']:.4f}")
                st.metric("F1-Score", f"{rf_metrics['f1_score']:.4f}")
                st.metric("ROC-AUC", f"{rf_metrics['roc_auc']:.4f}")
    
    # Show training status
    if st.session_state.models_trained:
        st.markdown("### ✅ Models Ready")
        st.success("Models have been trained and are ready for prediction and evaluation!")
        
        # Model selection for prediction
        model_names = list(st.session_state.models.keys())
        selected_model_name = st.selectbox(
            "Select model for prediction:",
            model_names,
            key="prediction_model"
        )
        st.session_state.current_model = st.session_state.models[selected_model_name]

def show_prediction_page():
    """Display the fraud prediction page."""
    
    st.markdown("## 🎯 Fraud Prediction")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the 'Model Training' page.")
        return
    
    # Model selection
    model_names = list(st.session_state.models.keys())
    selected_model_name = st.selectbox(
        "Select model for prediction:",
        model_names,
        key="prediction_model_select"
    )
    
    current_model = st.session_state.models[selected_model_name]
    
    # Prediction options
    prediction_type = st.radio(
        "Choose prediction method:",
        ["Single Transaction", "Batch Prediction", "Random Sample"]
    )
    
    if prediction_type == "Single Transaction":
        show_single_prediction(current_model)
    elif prediction_type == "Batch Prediction":
        show_batch_prediction(current_model)
    else:
        show_random_prediction(current_model)

def show_single_prediction(model):
    """Show single transaction prediction interface."""
    
    st.markdown("### Single Transaction Prediction")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("Time:", value=0.0, help="Time elapsed since first transaction")
            amount = st.number_input("Amount:", value=0.0, help="Transaction amount")
            
            # V1-V14 features
            v1 = st.number_input("V1:", value=0.0)
            v2 = st.number_input("V2:", value=0.0)
            v3 = st.number_input("V3:", value=0.0)
            v4 = st.number_input("V4:", value=0.0)
            v5 = st.number_input("V5:", value=0.0)
            v6 = st.number_input("V6:", value=0.0)
            v7 = st.number_input("V7:", value=0.0)
        
        with col2:
            # V8-V14 features
            v8 = st.number_input("V8:", value=0.0)
            v9 = st.number_input("V9:", value=0.0)
            v10 = st.number_input("V10:", value=0.0)
            v11 = st.number_input("V11:", value=0.0)
            v12 = st.number_input("V12:", value=0.0)
            v13 = st.number_input("V13:", value=0.0)
            v14 = st.number_input("V14:", value=0.0)
        
        # V15-V28 features (collapsed)
        with st.expander("Additional Features (V15-V28)"):
            col1, col2 = st.columns(2)
            v_features = {}
            
            with col1:
                for i in range(15, 22):
                    v_features[f'V{i}'] = st.number_input(f"V{i}:", value=0.0, key=f"v{i}")
            
            with col2:
                for i in range(22, 29):
                    v_features[f'V{i}'] = st.number_input(f"V{i}:", value=0.0, key=f"v{i}")
        
        submitted = st.form_submit_button("🔍 Predict Fraud")
        
        if submitted:
            # Create input data
            input_data = {
                'Time': time,
                'Amount': amount,
                'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5, 'V6': v6, 'V7': v7,
                'V8': v8, 'V9': v9, 'V10': v10, 'V11': v11, 'V12': v12, 'V13': v13, 'V14': v14
            }
            
            # Add V15-V28 features
            input_data.update(v_features)
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Transform using preprocessor
            preprocessor = st.session_state.preprocessor
            input_scaled = preprocessor.transform_new_data(input_df)
            
            # Make prediction
            prediction, probability = model.predict(input_scaled)
            
            if prediction is not None:
                # Display results
                st.markdown("### Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction[0] == 1:
                        st.markdown('<div class="fraud-alert">🚨 **FRAUD DETECTED**</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="safe-transaction">✅ **LEGITIMATE TRANSACTION**</div>', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Fraud Probability", f"{probability[0]:.4f}")
                    st.metric("Confidence", f"{max(probability[0], 1-probability[0]):.4f}")

def show_batch_prediction(model):
    """Show batch prediction interface."""
    
    st.markdown("### Batch Prediction")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with transaction data:",
        type=['csv'],
        help="File should contain the same columns as the training data"
    )
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_data)} transactions")
            
            # Show sample
            st.markdown("#### Sample Data:")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            if st.button("🔍 Predict Batch"):
                with st.spinner("Making predictions..."):
                    # Transform data
                    preprocessor = st.session_state.preprocessor
                    batch_scaled = preprocessor.transform_new_data(batch_data)
                    
                    # Make predictions
                    predictions, probabilities = model.predict(batch_scaled)
                    
                    if predictions is not None:
                        # Add predictions to data
                        batch_data['Prediction'] = predictions
                        batch_data['Fraud_Probability'] = probabilities
                        
                        # Display results
                        st.markdown("### Batch Prediction Results")
                        
                        # Summary
                        fraud_count = predictions.sum()
                        total_count = len(predictions)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", total_count)
                        with col2:
                            st.metric("Predicted Fraud", fraud_count)
                        with col3:
                            st.metric("Fraud Rate", f"{(fraud_count/total_count)*100:.2f}%")
                        
                        # Results table
                        st.markdown("#### Detailed Results:")
                        st.dataframe(batch_data, use_container_width=True)
                        
                        # Download results
                        csv = batch_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="fraud_predictions.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_random_prediction(model):
    """Show random sample prediction interface."""
    
    st.markdown("### Random Sample Prediction")
    
    n_samples = st.slider("Number of samples:", 1, 100, 10)
    
    if st.button("🎲 Generate Random Sample"):
        # Get random sample from test data
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        # Sample indices
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test.iloc[sample_indices]
        y_sample = y_test.iloc[sample_indices]
        
        # Make predictions
        predictions, probabilities = model.predict(X_sample)
        
        if predictions is not None:
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Actual': y_sample.values,
                'Predicted': predictions,
                'Fraud_Probability': probabilities,
                'Correct': (y_sample.values == predictions)
            })
            
            # Display results
            st.markdown("### Random Sample Results")
            
            # Summary metrics
            accuracy = (results_df['Correct'].sum() / len(results_df))
            fraud_detected = predictions.sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Predicted Fraud", fraud_detected)
            with col3:
                st.metric("Actual Fraud", y_sample.sum())
            
            # Results table
            st.markdown("#### Detailed Results:")
            st.dataframe(results_df, use_container_width=True)

def show_evaluation_page():
    """Display the model evaluation page."""
    
    st.markdown("## 📈 Model Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the 'Model Training' page.")
        return
    
    # Model comparison
    st.markdown("### Model Comparison")
    
    models = st.session_state.models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Compare models
    comparison_df = compare_models(models, X_test, y_test)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Detailed evaluation for selected model
    st.markdown("### Detailed Model Evaluation")
    
    model_names = list(models.keys())
    selected_model_name = st.selectbox(
        "Select model for detailed evaluation:",
        model_names,
        key="evaluation_model"
    )
    
    selected_model = models[selected_model_name]
    
    # Get evaluation metrics
    metrics, y_pred, y_proba = selected_model.evaluate(X_test, y_test)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
    with col5:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    
    # Plots
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curve", "📉 Precision-Recall Curve", "🎯 Feature Importance"])
    
    with tab1:
        cm_fig = plot_confusion_matrix(y_test, y_pred)
        st.plotly_chart(cm_fig, use_container_width=True)
    
    with tab2:
        roc_fig = plot_roc_curve(y_test, y_proba)
        st.plotly_chart(roc_fig, use_container_width=True)
    
    with tab3:
        pr_fig = plot_precision_recall_curve(y_test, y_proba)
        st.plotly_chart(pr_fig, use_container_width=True)
    
    with tab4:
        feature_importance = selected_model.get_feature_importance()
        if not feature_importance.empty:
            fi_fig = plot_feature_importance(feature_importance, top_n=15)
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

if __name__ == "__main__":
    main() 