import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from joblib import dump
import time
import warnings
warnings.filterwarnings('ignore')

print("Loading the cleaned flight data...")
df = pd.read_csv("data/flights2022_cleaned.csv")

print("\nDataset Shape:", df.shape)
print("\nChecking for null values:")
print(df.isnull().sum())

# Display class distribution
print("\nClass Distribution (Cancelled vs Not Cancelled):")
print(df['cancel'].value_counts())
print(f"Percentage of cancelled flights: {df['cancel'].mean() * 100:.2f}%")

# Feature Engineering
print("\nPerforming Feature Engineering...")

# Create time-based features if they don't already exist
if 'month' in df.columns:
    # Create season feature (1: Winter, 2: Spring, 3: Summer, 4: Fall)
    if 'season' not in df.columns:
        df['season'] = pd.cut(df['month'], 
                              bins=[0, 3, 6, 9, 12], 
                              labels=[1, 2, 3, 4], 
                              include_lowest=True)

# Create weather severity features
df['temp_extreme'] = ((df['temp'] > 90) | (df['temp'] < 32)).astype(int)
df['low_visibility'] = (df['visib'] < 5).astype(int)
df['high_wind'] = (df['wind_speed'] > 15).astype(int)
df['precipitation'] = (df['precip'] > 0).astype(int)

# Create distance categories if not already in the dataset
if 'distance_category' not in df.columns:
    df['distance_category'] = pd.cut(df['distance'], 
                                   bins=[0, 500, 1000, 2000, 5000], 
                                   labels=['short', 'medium', 'long', 'very_long'])

# Feature Selection - Creating two feature sets
# Basic features (similar to original analysis)
basic_features = ["distance", "temp", "dewp", "humid", "pressure", "precip", "visib"]

# Enhanced features including engineered ones and categorical variables
enhanced_features = basic_features + [
    "temp_extreme", "low_visibility", "high_wind", "precipitation"
]

# Add categorical features if available
categorical_features = []
if 'season' in df.columns:
    enhanced_features.append('season')
    categorical_features.append('season')
if 'day_of_week' in df.columns:
    enhanced_features.append('day_of_week')
    categorical_features.append('day_of_week')
if 'distance_category' in df.columns:
    enhanced_features.append('distance_category')
    categorical_features.append('distance_category')

# Target variable
y = df["cancel"]

# Print feature sets
print("\nBasic Features:", basic_features)
print("\nEnhanced Features:", enhanced_features)
print("\nCategorical Features:", categorical_features)

# Split the data
df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# Function to create appropriate preprocessor for a given feature set
def create_preprocessor(features):
    numeric_features = [f for f in features if f not in categorical_features]
    cat_features = [f for f in features if f in categorical_features]
    
    transformers = [('num', StandardScaler(), numeric_features)]
    
    if cat_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_features))
    
    return ColumnTransformer(transformers=transformers)

# Function to evaluate a model with cross-validation and plot performance metrics
def evaluate_model(model, name, X_train, y_train, X_test, y_test, feature_set_name="basic"):
    # Create appropriate preprocessor for this feature set
    features = basic_features if feature_set_name == "basic" else enhanced_features
    preprocessor = create_preprocessor(features)
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train the model with timing
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"\n{name} Results ({feature_set_name} features):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training time: {train_time:.2f} seconds")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name} ({feature_set_name} features)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}_{feature_set_name}.png')
    
    # If the model has predict_proba method, plot ROC curve and Precision-Recall curve
    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
        # Get prediction probabilities
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name} ({feature_set_name} features)')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_{name.replace(" ", "_").lower()}_{feature_set_name}.png')
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {name} ({feature_set_name} features)')
        plt.legend(loc="lower left")
        plt.savefig(f'pr_curve_{name.replace(" ", "_").lower()}_{feature_set_name}.png')
    
    return accuracy, pipeline

# Function for hyperparameter tuning
def tune_hyperparameters(model_class, param_grid, name, X_train, y_train, X_test, y_test, feature_set_name="basic"):
    # Create appropriate preprocessor for this feature set
    features = basic_features if feature_set_name == "basic" else enhanced_features
    preprocessor = create_preprocessor(features)
    
    # Create pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_class())
    ])
    
    # Create grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid={"classifier__" + key: value for key, value in param_grid.items()},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model with timing
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Print best parameters
    print(f"\n{name} Hyperparameter Tuning Results ({feature_set_name} features):")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {train_time:.2f} seconds")
    
    # Evaluate best model on test set
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy with best parameters: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Tuned {name} ({feature_set_name} features)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_tuned_{name.replace(" ", "_").lower()}_{feature_set_name}.png')
    
    return accuracy, grid_search.best_estimator_

print("\n\n=============== Basic Features (Original) ===============")
# Extract basic features
X_train_basic = df_train[basic_features]
X_test_basic = df_test[basic_features]

# Evaluate models with basic features
print("\nTraining models with basic features...")
models_results_basic = []

# 1. Naive Bayes
print("\nTraining Naive Bayes model...")
nb_accuracy, nb_model = evaluate_model(
    GaussianNB(), "Naive Bayes", X_train_basic, y_train, X_test_basic, y_test, "basic"
)
models_results_basic.append(("Naive Bayes", nb_accuracy))

# 2. Random Forest
print("\nTraining Random Forest model...")
rf_accuracy, rf_model = evaluate_model(
    RandomForestClassifier(n_estimators=100, random_state=42), 
    "Random Forest", X_train_basic, y_train, X_test_basic, y_test, "basic"
)
models_results_basic.append(("Random Forest", rf_accuracy))

# 3. Gradient Boosting
print("\nTraining Gradient Boosting model...")
gb_accuracy, gb_model = evaluate_model(
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting", X_train_basic, y_train, X_test_basic, y_test, "basic"
)
models_results_basic.append(("Gradient Boosting", gb_accuracy))

# 4. Logistic Regression
print("\nTraining Logistic Regression model...")
lr_accuracy, lr_model = evaluate_model(
    LogisticRegression(max_iter=1000, random_state=42),
    "Logistic Regression", X_train_basic, y_train, X_test_basic, y_test, "basic"
)
models_results_basic.append(("Logistic Regression", lr_accuracy))

print("\n\n=============== Enhanced Features ===============")
# Extract enhanced features
X_train_enhanced = df_train[enhanced_features]
X_test_enhanced = df_test[enhanced_features]

# Evaluate models with enhanced features
print("\nTraining models with enhanced features...")
models_results_enhanced = []

# 1. Naive Bayes
print("\nTraining Naive Bayes model...")
nb_accuracy_enhanced, nb_model_enhanced = evaluate_model(
    GaussianNB(), "Naive Bayes", X_train_enhanced, y_train, X_test_enhanced, y_test, "enhanced"
)
models_results_enhanced.append(("Naive Bayes", nb_accuracy_enhanced))

# 2. Random Forest
print("\nTraining Random Forest model...")
rf_accuracy_enhanced, rf_model_enhanced = evaluate_model(
    RandomForestClassifier(n_estimators=100, random_state=42), 
    "Random Forest", X_train_enhanced, y_train, X_test_enhanced, y_test, "enhanced"
)
models_results_enhanced.append(("Random Forest", rf_accuracy_enhanced))

# 3. Gradient Boosting
print("\nTraining Gradient Boosting model...")
gb_accuracy_enhanced, gb_model_enhanced = evaluate_model(
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting", X_train_enhanced, y_train, X_test_enhanced, y_test, "enhanced"
)
models_results_enhanced.append(("Gradient Boosting", gb_accuracy_enhanced))

# 4. Logistic Regression
print("\nTraining Logistic Regression model...")
lr_accuracy_enhanced, lr_model_enhanced = evaluate_model(
    LogisticRegression(max_iter=1000, random_state=42),
    "Logistic Regression", X_train_enhanced, y_train, X_test_enhanced, y_test, "enhanced"
)
models_results_enhanced.append(("Logistic Regression", lr_accuracy_enhanced))

# Hyperparameter tuning on the best model from basic features
print("\n\n=============== Hyperparameter Tuning ===============")
best_model_name_basic, best_accuracy_basic = max(models_results_basic, key=lambda x: x[1])
print(f"\nBest model with basic features: {best_model_name_basic} (Accuracy: {best_accuracy_basic:.4f})")

if best_model_name_basic == "Random Forest":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    tuned_accuracy_basic, tuned_model_basic = tune_hyperparameters(
        RandomForestClassifier, param_grid, "Random Forest", 
        X_train_basic, y_train, X_test_basic, y_test, "basic"
    )
    
elif best_model_name_basic == "Gradient Boosting":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    tuned_accuracy_basic, tuned_model_basic = tune_hyperparameters(
        GradientBoostingClassifier, param_grid, "Gradient Boosting", 
        X_train_basic, y_train, X_test_basic, y_test, "basic"
    )
    
elif best_model_name_basic == "Logistic Regression":
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2']
    }
    tuned_accuracy_basic, tuned_model_basic = tune_hyperparameters(
        LogisticRegression, param_grid, "Logistic Regression", 
        X_train_basic, y_train, X_test_basic, y_test, "basic"
    )

elif best_model_name_basic == "Naive Bayes":
    # Naive Bayes has fewer hyperparameters to tune
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    tuned_accuracy_basic, tuned_model_basic = tune_hyperparameters(
        GaussianNB, param_grid, "Naive Bayes", 
        X_train_basic, y_train, X_test_basic, y_test, "basic"
    )

# Hyperparameter tuning on the best model from enhanced features
best_model_name_enhanced, best_accuracy_enhanced = max(models_results_enhanced, key=lambda x: x[1])
print(f"\nBest model with enhanced features: {best_model_name_enhanced} (Accuracy: {best_accuracy_enhanced:.4f})")

if best_model_name_enhanced == "Random Forest":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    tuned_accuracy_enhanced, tuned_model_enhanced = tune_hyperparameters(
        RandomForestClassifier, param_grid, "Random Forest", 
        X_train_enhanced, y_train, X_test_enhanced, y_test, "enhanced"
    )
    
elif best_model_name_enhanced == "Gradient Boosting":
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    tuned_accuracy_enhanced, tuned_model_enhanced = tune_hyperparameters(
        GradientBoostingClassifier, param_grid, "Gradient Boosting", 
        X_train_enhanced, y_train, X_test_enhanced, y_test, "enhanced"
    )
    
elif best_model_name_enhanced == "Logistic Regression":
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2']
    }
    tuned_accuracy_enhanced, tuned_model_enhanced = tune_hyperparameters(
        LogisticRegression, param_grid, "Logistic Regression", 
        X_train_enhanced, y_train, X_test_enhanced, y_test, "enhanced"
    )

elif best_model_name_enhanced == "Naive Bayes":
    # Naive Bayes has fewer hyperparameters to tune
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    tuned_accuracy_enhanced, tuned_model_enhanced = tune_hyperparameters(
        GaussianNB, param_grid, "Naive Bayes", 
        X_train_enhanced, y_train, X_test_enhanced, y_test, "enhanced"
    )

# Summary comparison of all models
print("\n\n=============== Model Performance Summary ===============")

# Basic features models
print("\nModels with Basic Features:")
for name, accuracy in models_results_basic:
    print(f"{name}: {accuracy:.4f}")

# Enhanced features models
print("\nModels with Enhanced Features:")
for name, accuracy in models_results_enhanced:
    print(f"{name}: {accuracy:.4f}")

# Tuned models
print("\nHyperparameter Tuned Models:")
print(f"Tuned {best_model_name_basic} (Basic Features): {tuned_accuracy_basic:.4f}")
print(f"Tuned {best_model_name_enhanced} (Enhanced Features): {tuned_accuracy_enhanced:.4f}")

# Plot comparison of all models
model_names = [name for name, _ in models_results_basic]
basic_accuracies = [acc for _, acc in models_results_basic]
enhanced_accuracies = [acc for _, acc in models_results_enhanced]

# Create bar plot
plt.figure(figsize=(14, 8))
x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, basic_accuracies, width, label='Basic Features')
plt.bar(x + width/2, enhanced_accuracies, width, label='Enhanced Features')

# Add tuned model results
plt.scatter(model_names.index(best_model_name_basic) - width/2, tuned_accuracy_basic, 
            marker='*', color='red', s=200, zorder=3, 
            label='Tuned Basic Features')
plt.scatter(model_names.index(best_model_name_enhanced) + width/2, tuned_accuracy_enhanced, 
            marker='*', color='darkred', s=200, zorder=3, 
            label='Tuned Enhanced Features')

plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance Comparison', fontsize=16)
plt.xticks(x, model_names, rotation=45)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/all_models_comparison.png')

# Find the overall best model
overall_best_accuracy = max(tuned_accuracy_basic, tuned_accuracy_enhanced)
overall_best_model = "Tuned " + best_model_name_basic + " (Basic Features)" if tuned_accuracy_basic == overall_best_accuracy else "Tuned " + best_model_name_enhanced + " (Enhanced Features)"
best_model_to_save = tuned_model_basic if tuned_accuracy_basic == overall_best_accuracy else tuned_model_enhanced

print(f"\nOverall best model: {overall_best_model} (Accuracy: {overall_best_accuracy:.4f})")

# Save the best model
model_filename = f"best_flight_cancellation_model.joblib"
dump(best_model_to_save, model_filename)
print(f"Best model saved as '{model_filename}'")

print("\nAnalysis complete!") 