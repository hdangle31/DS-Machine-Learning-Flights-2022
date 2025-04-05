import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import time
import os
import sys

# Set pandas display options
pd.set_option('display.max_columns', None)

try:
    # Load the cleaned data
    print("Loading the cleaned flight data...")
    if not os.path.exists("data/flights2022_cleaned.csv"):
        print("Error: The file flights2022_cleaned.csv does not exist in the current directory.")
        print(f"Current working directory: {os.getcwd()}")
        print("Files in current directory:", os.listdir())
        sys.exit(1)
    
    df = pd.read_csv("data/flights2022_cleaned.csv")
    
    # Display basic information about the dataset
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nChecking for null values:")
    print(df.isnull().sum())
    
    # Check if 'cancel' column exists
    if 'cancel' not in df.columns:
        print("Error: 'cancel' column not found in the dataset.")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)
    
    # Target variable: 'cancel' (0 for not cancelled, 1 for cancelled)
    # Display class distribution
    print("\nClass Distribution (Cancelled vs Not Cancelled):")
    print(df['cancel'].value_counts())
    print(f"Percentage of cancelled flights: {df['cancel'].mean() * 100:.2f}%")
    
    # Feature Selection - using weather and flight-specific features
    # Check if all required features exist
    required_features = ["distance", "temp", "dewp", "humid", "pressure", "precip", "visib"]
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        print("Will use available features instead.")
        available_features = [f for f in required_features if f in df.columns]
        print(f"Available features: {available_features}")
        X = df[available_features]
    else:
        X = df[required_features]
        print(f"\nUsing features: {required_features}")
    
    y = df["cancel"]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create directory for visualizations if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Function to evaluate a model with cross-validation
    def evaluate_model_cv(model, name, X_scaled, y):
        print(f"\nEvaluating {name} model with cross-validation...")
        
        # Define scoring metrics for cross-validation
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        # Perform cross-validation with multiple metrics
        start_time = time.time()
        cv_results = cross_validate(model, X_scaled, y, cv=5, scoring=scoring)
        cv_time = time.time() - start_time
        
        # Extract and format results
        cv_metrics = {
            'Accuracy': cv_results['test_accuracy'].mean(),
            'Precision': cv_results['test_precision'].mean(),
            'Recall': cv_results['test_recall'].mean(),
            'F1-Score': cv_results['test_f1'].mean()
        }
        
        cv_std = {
            'Accuracy': cv_results['test_accuracy'].std(),
            'Precision': cv_results['test_precision'].std(),
            'Recall': cv_results['test_recall'].std(),
            'F1-Score': cv_results['test_f1'].std()
        }
        
        # Print results
        print(f"\n{name} Cross-Validation Results:")
        print(f"Accuracy: {cv_metrics['Accuracy']:.4f} (±{cv_std['Accuracy']:.4f})")
        print(f"Precision: {cv_metrics['Precision']:.4f} (±{cv_std['Precision']:.4f})")
        print(f"Recall: {cv_metrics['Recall']:.4f} (±{cv_std['Recall']:.4f})")
        print(f"F1-Score: {cv_metrics['F1-Score']:.4f} (±{cv_std['F1-Score']:.4f})")
        print(f"Cross-validation time: {cv_time:.2f} seconds")
        
        # Fit the model on the full training data for feature importance and confusion matrix
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'visualizations/confusion_matrix_{name.replace(" ", "_").lower()}.png')
        print(f"Confusion matrix saved as visualizations/confusion_matrix_{name.replace(' ', '_').lower()}.png")
        
        return cv_metrics, model
    
    # Initialize models
    models = {
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # Evaluate all models
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Processing {name} model...")
        cv_metrics, fitted_model = evaluate_model_cv(model, name, X_train_scaled, y_train)
        results[name] = cv_metrics
        fitted_models[name] = fitted_model
    
    # Feature importance analysis for Random Forest (tree-based models provide feature importance)
    if hasattr(fitted_models["Random Forest"], 'feature_importances_'):
        print("\nAnalyzing feature importance...")
        # Get feature importances
        importances = fitted_models["Random Forest"].feature_importances_
        # Create a DataFrame for better visualization
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\nFeature Importance (Random Forest):")
        print(feature_importance)
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Feature Importance for Flight Cancellation Prediction Using Random Forest')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')
        print("Feature importance plot saved as visualizations/feature_importance.png")
    
    # Create comparison DataFrame for all metrics
    comparison_df = pd.DataFrame(results).T
    print("\nModel Performance Comparison (Cross-validation):")
    print(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv('visualizations/model_comparison_metrics.csv')
    print("Model comparison saved to visualizations/model_comparison_metrics.csv")
    
    # Plot comparison for all metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(x=comparison_df.index, y=comparison_df[metric], ax=ax)
        ax.set_title(f'Model Comparison - {metric}')
        ax.set_ylabel(metric)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison_all_metrics.png')
    print("Comprehensive model comparison plot saved as visualizations/model_comparison_all_metrics.png")
    
    # Create a side-by-side bar chart for all metrics together
    plt.figure(figsize=(14, 8))
    comparison_df_melted = comparison_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
    
    sns.barplot(x='index', y='Score', hue='Metric', data=comparison_df_melted)
    plt.title('Model Performance Comparison - All Metrics')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison_metrics_combined.png')
    print("Combined metrics comparison plot saved as visualizations/model_comparison_metrics_combined.png")
    
    # Identify the best model based on F1-Score (good balance of precision and recall)
    best_model = comparison_df['F1-Score'].idxmax()
    print(f"\nBest performing model based on F1-Score: {best_model} with F1-Score: {comparison_df.loc[best_model, 'F1-Score']:.4f}")
    
    print("\nAnalysis complete!")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 