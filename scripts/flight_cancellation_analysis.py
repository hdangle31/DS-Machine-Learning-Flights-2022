import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.gridspec as gridspec
import sys
import os
import traceback

# Set style
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

try:
    # Load the cleaned data
    print("Loading the cleaned flight data...")
    if not os.path.exists("data/flights2022_cleaned.csv"):
        print("Error: The file flights2022_cleaned.csv does not exist.")
        sys.exit(1)
    
    df = pd.read_csv("data/flights2022_cleaned.csv")
    
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    # Display basic information about cancellations
    print("\nCancellations Distribution:")
    print(df['cancel'].value_counts())
    print(f"Percentage of cancelled flights: {df['cancel'].mean() * 100:.2f}%")
    
    # Create a directory for visualizations
    print("\nCreating visualizations directory...")
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Analyze cancellations by month and day of week (if available)
    print("\nAnalyzing cancellations by time factors...")
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.3)
    
    # 1.1 Cancellations by month (if month is in the dataset)
    print("  - Creating month analysis...")
    ax1 = plt.subplot(gs[0, 0])
    monthly_cancellations = df.groupby('month')['cancel'].mean() * 100
    monthly_counts = df.groupby('month').size()
    
    sns.barplot(x=monthly_cancellations.index, y=monthly_cancellations.values, ax=ax1)
    ax1.set_title('Cancellation Rate by Month', pad=15, fontsize=14)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Cancellation Rate (%)', fontsize=12)
    


    # 1.2 Cancellations by day of week (if day of week is in the dataset or can be created)
    if 'day_of_week' in df.columns:
        print("  - Creating day of week analysis...")
        ax2 = plt.subplot(gs[0, 1])
        dow_cancellations = df.groupby('day_of_week')['cancel'].mean() * 100
        dow_counts = df.groupby('day_of_week').size()
        
        sns.barplot(x=dow_cancellations.index, y=dow_cancellations.values, ax=ax2)
        ax2.set_title('Cancellation Rate by Day of Week', pad=15, fontsize=14)
        ax2.set_xlabel('Day of Week (1=Monday, 7=Sunday)', fontsize=12)
        ax2.set_ylabel('Cancellation Rate (%)', fontsize=12)
        

    elif 'day' in df.columns and 'month' in df.columns and 'year' in df.columns:
        print("  - Creating day of week from date columns...")
        # Create day of week if not already present
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df['day_of_week'] = df['date'].dt.dayofweek + 1  # 1=Monday, 7=Sunday
        
        ax2 = plt.subplot(gs[0, 1])
        dow_cancellations = df.groupby('day_of_week')['cancel'].mean() * 100
        dow_counts = df.groupby('day_of_week').size()
        
        sns.barplot(x=dow_cancellations.index, y=dow_cancellations.values, ax=ax2)
        ax2.set_title('Cancellation Rate by Day of Week', pad=15, fontsize=14)
        ax2.set_xlabel('Day of Week (0 = Monday, 6 = Sunday)', fontsize=12)
        ax2.set_ylabel('Cancellation Rate (%)', fontsize=12)

     
    # 1.3 Cancellations by hour of day (if hour is available)
    if 'hour' in df.columns:
        print("  - Creating hour analysis...")
        ax3 = plt.subplot(gs[1, 0])
        hour_cancellations = df.groupby('hour')['cancel'].mean() * 100
        hour_counts = df.groupby('hour').size()
        
        sns.barplot(x=hour_cancellations.index, y=hour_cancellations.values, ax=ax3)
        ax3.set_title('Cancellation Rate by Hour of Day', pad=15, fontsize=14)
        ax3.set_xlabel('Hour (24h format)', fontsize=12)
        ax3.set_ylabel('Cancellation Rate (%)', fontsize=12)
        

    # 1.4 Cancellations by season (created from month)
    print("  - Creating season analysis...")
    # Create season if not already present
    if 'season' not in df.columns:
        df['season'] = pd.cut(df['month'], 
                        bins=[0, 3, 6, 9, 12], 
                        labels=['Winter', 'Spring', 'Summer', 'Fall'], 
                        include_lowest=True)
    
    ax4 = plt.subplot(gs[1, 1])
    season_cancellations = df.groupby('season')['cancel'].mean() * 100
    season_counts = df.groupby('season').size()
    
    sns.barplot(x=season_cancellations.index, y=season_cancellations.values, ax=ax4)
    ax4.set_title('Cancellation Rate by Season', pad=15, fontsize=14)
    ax4.set_xlabel('Season', fontsize=12)
    ax4.set_ylabel('Cancellation Rate (%)', fontsize=12)
    


    plt.tight_layout(pad=3.0)
    plt.savefig('visualizations/cancellations_by_time.png', dpi=300)
    print("  - Time analysis visualization saved")
    plt.close()
    
    # 2. Analyze cancellations by weather factors

    print("\nAnalyzing cancellations by weather factors...")
    
    weather_features = [col for col in ['temp', 'dewp', 'humid', 'wind_speed', 
                                          'wind_gust', 'precip', 'pressure', 'visib'] 
                          if col in df.columns]
    
    print(f"  - Found weather features: {weather_features}")
    
    if weather_features:
        # Create binned versions of continuous features for easier visualization
        print("  - Creating binned weather features...")
        for feature in weather_features:
            if feature == 'temp':
                df[f'{feature}_bin'] = pd.cut(df[feature], 
                                              bins=[0, 32, 50, 70, 90], 
                                              labels=['0-32°F', '32-50°F', '50-70°F', '70-90°F'])
            elif feature == 'visib':
                df[f'{feature}_bin'] = pd.cut(df[feature], 
                                              bins=[0, 1, 3, 5, 7, 10], 
                                              labels=['< 1 mile', '1-3 miles', '3-5 miles', '5-7 miles', '7-10 miles'])
            elif feature == 'precip':
                df[f'{feature}_bin'] = pd.cut(df[feature], 
                                              bins=[-0.01, 0, 0.1, 0.25, 0.5], 
                                              labels=['None', '0-0.1 in', '0.1-0.25 in', '0.25-0.5 in'])
            elif feature == 'wind_speed':
                df[f'{feature}_bin'] = pd.cut(df[feature], 
                                              bins=[0, 5, 10, 15, 20, 30, 100], 
                                              labels=['0-5 mph', '5-10 mph', '10-15 mph', '15-20 mph', '20-30 mph', '> 30 mph'])
            else:
                # For other features, create quintile bins
                df[f'{feature}_bin'] = pd.qcut(df[feature], q=5, duplicates='drop')
        
        # Create a multi-panel figure for weather factors
        print("  - Creating weather factors visualization...")
        fig = plt.figure(figsize=(20, 20))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.3)
        
        # Plot cancellation rate by each binned weather feature
        for i, feature in enumerate(weather_features):
            if i < 9:  # Limit to 9 panels
                print(f"    - Processing {feature}...")
                ax = fig.add_subplot(gs[i//3, i%3])
                bin_feature = f'{feature}_bin'
                
                # Calculate cancellation rates and counts
                feature_cancellations = df.groupby(bin_feature)['cancel'].mean() * 100
               
                
                sns.barplot(x=feature_cancellations.index, y=feature_cancellations.values, ax=ax)
                ax.set_title(f'Cancellation Rate by {feature.replace("_", " ").title()}', pad=15, fontsize=14)
                ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
                ax.set_ylabel('Cancellation Rate (%)', fontsize=12)
                
                # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                

        plt.tight_layout(pad=3.0)
        plt.savefig('visualizations/cancellations_by_weather.png', dpi=300)
        print("  - Weather analysis visualization saved")
        plt.close()
    
    # 3. Analyze cancellations by distance and route factors
    print("\nAnalyzing cancellations by distance and route factors...")
    
    # Create distance bins if not already present
    if 'distance' in df.columns and 'distance_category' not in df.columns:
        print("  - Creating distance categories...")
        df['distance_category'] = pd.cut(df['distance'], 
                                       bins=[0, 500, 1000, 2000, 5000], 
                                       labels=['short', 'medium', 'long', 'very_long'])
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 9))
    plt.subplots_adjust(wspace=0.3)
    
    # 3.1 Cancellations by distance category
    if 'distance_category' in df.columns:
        print("  - Analyzing cancellations by distance...")
        distance_cancellations = df.groupby('distance_category')['cancel'].mean() * 100
        distance_counts = df.groupby('distance_category').size()
        
        sns.barplot(x=distance_cancellations.index, y=distance_cancellations.values, ax=axs[0])
        axs[0].set_title('Cancellation Rate by Flight Distance', pad=15, fontsize=14)
        axs[0].set_xlabel('Distance Category', fontsize=12)
        axs[0].set_ylabel('Cancellation Rate (%)', fontsize=12)
        

    # 3.2 Cancellations by top airlines (if airline column exists)
    if 'airline' in df.columns:
        print("  - Analyzing cancellations by airline...")
        # Get top 10 airlines by frequency
        top_airlines = df['airline'].value_counts().nlargest(10).index
        airline_data = df[df['airline'].isin(top_airlines)]
        
        airline_cancellations = airline_data.groupby('airline')['cancel'].mean() * 100
        airline_counts = airline_data.groupby('airline').size()
        
        # Sort by cancellation rate
        airline_cancellations = airline_cancellations.sort_values(ascending=False)
        
        sns.barplot(x=airline_cancellations.values, y=airline_cancellations.index, ax=axs[1])
        axs[1].set_title('Cancellation Rate by Top 10 Airlines', pad=15, fontsize=14)
        axs[1].set_xlabel('Cancellation Rate (%)', fontsize=12)
        axs[1].set_ylabel('Airline', fontsize=12)
        

    plt.tight_layout(pad=3.0)
    plt.savefig('visualizations/cancellations_by_distance_airline.png', dpi=300)
    print("  - Distance and airline analysis visualization saved")
    plt.close()
    
    # 4. Feature importance analysis
    print("\nPerforming feature importance analysis...")
    
    # Select numerical features for importance analysis
    numerical_features = [col for col in df.columns if col not in ['cancel', 'date', 'airline', 'carrier', 
                                                                 'tailnum', 'flight', 'origin', 'dest', 
                                                                 'route', 'time_hour'] 
                          and df[col].dtype != 'object' 
                          and not col.endswith('_bin')
                          and not col.endswith('_category')]
    
    print(f"  - Found numerical features: {numerical_features}")
    
    if numerical_features:
        print("  - Preparing data for feature importance analysis...")
        X = df[numerical_features].copy()
        y = df['cancel']
        
        # Handle missing values if any
        X = X.fillna(X.median())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 4.1 Random Forest Feature Importance
        print("  - Training Random Forest for feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        rf_importances = rf.feature_importances_
        rf_df = pd.DataFrame({'Feature': X.columns, 'RF Importance': rf_importances})
        rf_df = rf_df.sort_values('RF Importance', ascending=False)
        
        # 4.2 Gradient Boosting Feature Importance
        print("  - Training Gradient Boosting for feature importance...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_scaled, y)
        gb_importances = gb.feature_importances_
        gb_df = pd.DataFrame({'Feature': X.columns, 'GB Importance': gb_importances})
        gb_df = gb_df.sort_values('GB Importance', ascending=False)
        
        # 4.3 Logistic Regression Coefficients
        print("  - Training Logistic Regression for feature importance...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled, y)
        lr_importances = np.abs(lr.coef_[0])  # Take absolute values for importance
        lr_df = pd.DataFrame({'Feature': X.columns, 'LR Importance': lr_importances})
        lr_df = lr_df.sort_values('LR Importance', ascending=False)

        # Create a multi-panel figure for feature importance
        print("  - Creating feature importance visualizations...")
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        
        # Plot Random Forest Feature Importance
        sns.barplot(x='RF Importance', y='Feature', data=rf_df.head(10), ax=axs[0])
        axs[0].set_title('Top 10 Features by Random Forest Importance')
        axs[0].set_xlabel('Random Forest Feature Importance')
        
        # Plot Gradient Boosting Feature Importance
        sns.barplot(x='GB Importance', y='Feature', data=gb_df.head(10), ax=axs[1])
        axs[1].set_title('Top 10 Features by Gradient Boosting Importance')
        axs[1].set_xlabel('Gradient Boosting Feature Importance')
        
        # Plot Logistic Regression Coefficients
        sns.barplot(x='LR Importance', y='Feature', data=lr_df.head(10), ax=axs[2])
        axs[2].set_title('Top 10 Features by Logistic Regression Importance')
        axs[2].set_xlabel('Logistic Regression Coefficient Magnitude')
        
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')
        print("  - Feature importance visualization saved")
        plt.close()
        
        # Print top 10 features from each method
        print("\nTop 10 features by Random Forest Importance:")
        print(rf_df.head(10))
        
        print("\nTop 10 features by Gradient Boosting Importance:")
        print(gb_df.head(10))
        
        print("\nTop 10 features by Logistic Regression Importance:")
        print(lr_df.head(10))
    # 5. Correlation Heatmap of numerical features
    print("\nGenerating correlation heatmap...")
    
    if numerical_features:
        print("  - Creating correlation matrix...")
        plt.figure(figsize=(16, 14))
        correlation_matrix = df[numerical_features + ['cancel']].corr()
        
        # Create a mask to hide the upper triangle (for a cleaner look)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Plot the heatmap
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", 
                    cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5,
                    cbar_kws={"shrink": .8})
        
        plt.title('Correlation Heatmap of Numerical Features with Cancellation', fontsize=16)
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png')
        print("  - Correlation heatmap saved")
        plt.close()
    
    # 6. Pairplot of top features related to cancellations
    print("\nGenerating pairplot of top features...")
    
    if numerical_features:
        print("  - Preparing data for pairplot...")
        # Get top 5 most important features from Random Forest
        top5_features = rf_df.head(5)['Feature'].tolist()
        
        # Add the target variable
        plot_data = df[top5_features + ['cancel']].copy()
        
        # Create a more descriptive target for the plot
        plot_data['Status'] = plot_data['cancel'].map({1: 'Cancelled', 0: 'Not Cancelled'})
        
        # Sample data for faster plotting (if dataset is large)
        if len(plot_data) > 10000:
            print("  - Sampling data for pairplot (dataset is large)...")
            # Ensure we have enough cancelled flights in the sample
            cancelled = plot_data[plot_data['Status'] == 'Cancelled'].sample(min(1000, len(plot_data[plot_data['Status'] == 'Cancelled'])))
            not_cancelled = plot_data[plot_data['Status'] == 'Not Cancelled'].sample(min(4000, len(plot_data[plot_data['Status'] == 'Not Cancelled'])))
            plot_data = pd.concat([cancelled, not_cancelled])
        
        # Create the pairplot
        print("  - Creating pairplot (this may take a while)...")
        g = sns.pairplot(plot_data, hue='Status', diag_kind='kde', 
                     plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
                     diag_kws={'alpha': 0.7})
        
        g.fig.suptitle('Pairplot of Top 5 Features Influencing Flight Cancellations', y=1.02, fontsize=16)
        g.fig.tight_layout()
        g.fig.savefig('visualizations/top_features_pairplot.png')
        print("  - Pairplot saved")
        plt.close()
    
    print("\nAnalysis complete! Visualizations saved in the 'visualizations' directory.")

except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1) 