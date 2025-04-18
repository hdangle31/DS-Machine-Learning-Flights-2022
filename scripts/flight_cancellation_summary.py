import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from PIL import Image
from datetime import datetime

# Create a directory for the report
report_dir = 'report'
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

# Start the HTML report
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Flight Cancellation Prediction Analysis</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        .image-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .highlight {{
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 40px;
            border-top: 1px solid #eee;
            padding-top: 20px;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Flight Cancellation Prediction Analysis</h1>
        <p>This report presents an analysis of flight cancellation prediction models using the flights2022 dataset. 
        The dataset contains information about flights in 2022, including various features related to flight schedules, 
        weather conditions, and cancellation status.</p>
        
        <div class="highlight">
            <h3>Key Findings Summary</h3>
            <ul>
                <li>The dataset contains {pd.read_csv('data/flights2022_cleaned.csv').shape[0]:,} flight records with a cancellation rate of 
                {pd.read_csv('data/flights2022_cleaned.csv')['cancel'].mean() * 100:.2f}%.</li>
                <li>Machine learning models can predict flight cancellations with significant accuracy.</li>
                <li>The most important features for predicting cancellations include flight duration, arrival/departure delays, 
                and weather-related factors like humidity and pressure.</li>
                <li>Different airlines and routes have varying cancellation rates, suggesting operational differences.</li>
                <li>Seasonal patterns and day-of-week variations exist in flight cancellations.</li>
            </ul>
        </div>
        
        <h2>1. Dataset Overview</h2>
        <p>The analysis uses the flights2022_cleaned.csv dataset, which contains flight data with weather information 
        and other relevant features. The dataset has been cleaned and preprocessed to handle missing values and 
        create additional features that might be useful for prediction.</p>
"""

# Add dataset information
try:
    df = pd.read_csv('data/flights2022_cleaned.csv')
    html_report += f"""
        <h3>Dataset Statistics</h3>
        <ul>
            <li>Number of flights: {df.shape[0]:,}</li>
            <li>Number of features: {df.shape[1]}</li>
            <li>Cancelled flights: {df['cancel'].sum():,} ({df['cancel'].mean() * 100:.2f}%)</li>
            <li>Time period: {df['month'].min()}-{df['month'].max()}/2022</li>
        </ul>
    """
except Exception as e:
    html_report += f"""
        <p>Error loading dataset statistics: {e}</p>
    """

# Add cancellation analysis by time factors
html_report += """
        <h2>2. Cancellation Patterns Analysis</h2>
        
        <h3>2.1 Temporal Patterns</h3>
        <p>Flight cancellations show distinct patterns across different time dimensions, including months, days of the week,
        and hours of the day. These patterns provide insights into operational challenges and potential factors affecting cancellations.</p>
        
        <div class="image-container">
            <img src="../visualizations/cancellations_by_time.png" alt="Cancellation Patterns by Time">
            <p><em>Figure 1: Cancellation rates by month, day of week, hour, and season</em></p>
        </div>
"""

# Add weather analysis
html_report += """
        <h3>2.2 Weather-Related Patterns</h3>
        <p>Weather conditions significantly impact flight operations. The analysis examines how different weather factors,
        such as temperature, visibility, precipitation, and wind, correlate with flight cancellations.</p>
        
        <div class="image-container">
            <img src="../visualizations/cancellations_by_weather.png" alt="Cancellation Patterns by Weather">
            <p><em>Figure 2: Cancellation rates by weather conditions</em></p>
        </div>
"""

# Add airline and distance analysis
html_report += """
        <h3>2.3 Airline and Distance Analysis</h3>
        <p>Different airlines have varying operational practices and fleet characteristics, resulting in different
        cancellation rates. Similarly, flight distance can affect cancellation probabilities due to operational complexity
        and exposure to changing weather conditions along the route.</p>
        
        <div class="image-container">
            <img src="../visualizations/cancellations_by_distance_airline.png" alt="Cancellation Patterns by Distance and Airline">
            <p><em>Figure 3: Cancellation rates by flight distance and airline</em></p>
        </div>
"""

# Add feature importance section
html_report += """
        <h2>3. Predictive Factors</h2>
        
        <h3>3.1 Feature Importance</h3>
        <p>Machine learning models identify the most influential factors in predicting flight cancellations. 
        Understanding these factors helps prioritize operational improvements and develop more effective predictive models.</p>
        
        <div class="image-container">
            <img src="../visualizations/feature_importance.png" alt="Feature Importance">
            <p><em>Figure 4: Top features for predicting flight cancellations using different importance metrics</em></p>
        </div>
        
        <h3>3.2 Feature Relationships</h3>
        <p>The relationships between features provide additional insights into how different factors interact to affect
        cancellation probability. The pairplot below shows relationships between the top predictive features.</p>
        
        <div class="image-container">
            <img src="../visualizations/top_features_pairplot.png" alt="Feature Relationships">
            <p><em>Figure 5: Relationships between top predictive features</em></p>
        </div>
        
        <h3>3.3 Correlation Analysis</h3>
        <p>Correlation analysis reveals how different features relate to each other and to flight cancellations.
        Strong correlations can indicate redundant information or potential causal relationships.</p>
        
        <div class="image-container">
            <img src="../visualizations/correlation_heatmap.png" alt="Correlation Heatmap">
            <p><em>Figure 6: Correlation heatmap of numerical features</em></p>
        </div>

        <h3>3.4 Model Performance</h3>
        <p>We evaluated multiple machine learning models for predicting flight cancellations. The chart below
        compares the accuracy of these models.</p>
        
        <div class="image-container">
            <img src="../visualizations/model_comparison.png" alt="Model Comparison">
            <p><em>Figure 7: Performance comparison of different prediction models</em></p>
        </div>
"""

# Add confusion matrices
html_report += """
        <h2>4. Predictive Models</h2>
        
        <h3>4.1 Model Performance Comparison</h3>
        <p>Various machine learning models were trained to predict flight cancellations. The performance comparison
        helps identify the most effective modeling approaches for this prediction task.</p>
        
        <div class="image-container">
            <img src="../model_comparison.png" alt="Model Comparison">
            <p><em>Figure 7: Performance comparison of different machine learning models</em></p>
        </div>
"""

# Add confusion matrices
html_report += """
        <h3>4.2 Model Evaluation Details</h3>
        <p>Confusion matrices provide detailed insights into model performance, showing true positives, false positives,
        true negatives, and false negatives. This helps understand the types of errors each model makes.</p>
        
        <div class="image-container" style="display: flex; flex-wrap: wrap; justify-content: center;">
"""

# Add all confusion matrices
confusion_matrices = glob.glob("confusion_matrix_*.png")
for i, cm_file in enumerate(confusion_matrices):
    model_name = cm_file.replace("confusion_matrix_", "").replace(".png", "").replace("_", " ").title()
    html_report += f"""
            <div style="margin: 10px; flex: 0 0 45%;">
                <img src="../{cm_file}" alt="{model_name} Confusion Matrix" style="width: 100%;">
                <p><em>Figure {8+i}: Confusion Matrix - {model_name}</em></p>
            </div>
    """

html_report += """
        </div>
"""

# Add interpretations and recommendations
html_report += """
        <h2>5. Interpretations and Recommendations</h2>
        
        <div class="highlight">
            <h3>5.1 Key Insights</h3>
            <ul>
                <li>Flight duration and air time are the strongest predictors of cancellations, suggesting that longer flights 
                    may be more prone to cancellation due to complex planning and increased exposure to changing conditions.</li>
                <li>Arrival and departure delays strongly correlate with cancellations, indicating that operational disruptions 
                    tend to cascade into cancellations.</li>
                <li>Weather factors, particularly humidity, pressure, and precipitation, significantly influence cancellation rates.</li>
                <li>Different airlines show varying cancellation rates, suggesting differences in operational robustness and policies.</li>
                <li>Seasonal patterns exist, with some months showing higher cancellation rates than others.</li>
            </ul>
        </div>
        
        <h3>5.2 Operational Recommendations</h3>
        <ul>
            <li><strong>Proactive Delay Management:</strong> Since delays correlate strongly with cancellations, implementing 
                more effective delay management strategies could help reduce cancellation rates.</li>
            <li><strong>Weather-Based Planning:</strong> Develop more sophisticated weather contingency plans, particularly 
                focusing on humidity, precipitation, and pressure changes.</li>
            <li><strong>Airline-Specific Strategies:</strong> Airlines with higher cancellation rates should analyze their 
                operations to identify specific areas for improvement.</li>
            <li><strong>Seasonal Preparedness:</strong> Allocate additional resources during seasons with historically higher 
                cancellation rates.</li>
            <li><strong>Enhanced Predictive Systems:</strong> Implement machine learning models (particularly Random Forest or 
                Gradient Boosting) for early cancellation prediction to improve passenger notification and resource reallocation.</li>
        </ul>
        
        <h3>5.3 Further Research Opportunities</h3>
        <ul>
            <li>Develop more detailed analysis of specific routes and airports to identify localized cancellation patterns.</li>
            <li>Investigate the relationship between aircraft type and cancellation probability.</li>
            <li>Explore the impact of staffing levels and crew scheduling on cancellation rates.</li>
            <li>Analyze the economic impact of different cancellation prediction and management strategies.</li>
        </ul>
"""

# Add conclusion
html_report += f"""
        <h2>6. Conclusion</h2>
        <p>Flight cancellations can be predicted with reasonable accuracy using machine learning models, with the most effective 
        models achieving significant predictive power. The analysis reveals that a combination of operational factors (flight duration, 
        delays) and environmental conditions (weather parameters) strongly influence cancellation probability.</p>
        
        <p>By implementing the recommendations provided in this report, airlines and airport operators can potentially reduce 
        cancellation rates, improve operational efficiency, and enhance passenger experience. The predictive models developed 
        in this analysis can serve as valuable tools for proactive flight management and resource allocation.</p>
        
        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Flight Cancellation Prediction Analysis</p>
        </div>
    </div>
</body>
</html>
"""

# Write the report to a file
with open(os.path.join(report_dir, 'flight_cancellation_report.html'), 'w') as f:
    f.write(html_report)

print(f"Report generated successfully and saved to {os.path.join(report_dir, 'flight_cancellation_report.html')}")

# Generate a text summary as well
text_summary = """
FLIGHT CANCELLATION PREDICTION ANALYSIS - EXECUTIVE SUMMARY
==========================================================

1. DATASET OVERVIEW
------------------
"""

try:
    df = pd.read_csv('data/flights2022_cleaned.csv')
    text_summary += f"""
- Dataset contains {df.shape[0]:,} flight records from 2022
- Cancellation rate is {df['cancel'].mean() * 100:.2f}%
- {df.shape[1]} features including flight details, weather conditions, and operational metrics
"""
except Exception as e:
    text_summary += f"""
- Error loading dataset statistics: {e}
"""

text_summary += """
2. KEY FINDINGS
--------------
- Most important predictors of cancellations: flight duration, air time, arrival/departure delays
- Weather factors significantly impact cancellations, particularly humidity and pressure
- Different airlines show varying cancellation rates, suggesting operational differences
- Seasonal patterns exist in cancellation rates
- Machine learning models (especially tree-based models) can predict cancellations with good accuracy

3. MODEL PERFORMANCE
------------------
- Multiple models evaluated: Naive Bayes, Random Forest, Gradient Boosting, Logistic Regression
- Tree-based models (Random Forest, Gradient Boosting) generally performed best
- Feature engineering and hyperparameter tuning improved model performance

4. RECOMMENDATIONS
----------------
- Implement proactive delay management strategies
- Develop more sophisticated weather contingency plans
- Address airline-specific operational issues for carriers with high cancellation rates
- Allocate additional resources during seasons with higher cancellation rates
- Deploy machine learning models for early cancellation prediction

5. NEXT STEPS
-----------
- Analyze specific routes and airports for localized patterns
- Investigate aircraft type impact on cancellations
- Explore staffing and crew scheduling effects
- Analyze economic impact of different cancellation management strategies

Report generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Write the text summary to a file
with open(os.path.join(report_dir, 'executive_summary.txt'), 'w') as f:
    f.write(text_summary)

print(f"Executive summary generated successfully and saved to {os.path.join(report_dir, 'executive_summary.txt')}") 