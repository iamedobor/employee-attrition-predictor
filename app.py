"""
Employee Attrition Prediction API
"""

from flask import Flask, render_template, request, jsonify
import joblib  
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os  

app = Flask(__name__)

# Updated model loading
print("Loading model artifacts...")
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

model = joblib.load(os.path.join(MODEL_DIR, 'attrition_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
explainer = joblib.load(os.path.join(MODEL_DIR, 'shap_explainer.pkl'))
numerical_cols = joblib.load(os.path.join(MODEL_DIR, 'numerical_cols.pkl'))
categorical_cols = joblib.load(os.path.join(MODEL_DIR, 'categorical_cols.pkl'))

print("✓ All models loaded successfully!")

# Debug: Verify model and scaler types
print(f"\nModel type: {type(model)}")
print(f"Model has predict method: {hasattr(model, 'predict')}")
print(f"Scaler type: {type(scaler)}")
print(f"Scaler has transform method: {hasattr(scaler, 'transform')}")

# Categorical options
CATEGORICAL_OPTIONS = {
    'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'EducationField': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'],
    'Gender': ['Male', 'Female'],
    'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 
                'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['No', 'Yes']
}

@app.route('/')
def home():
    return render_template('index.html', 
                         feature_names=feature_names,
                         categorical_options=CATEGORICAL_OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"\nReceived data: {data}")
        
        # Create DataFrame
        input_df = pd.DataFrame([data], columns=feature_names)
        print(f"Input DataFrame shape: {input_df.shape}")
        print(f"Input DataFrame:\n{input_df}")
        
        # Encode categorical features
        for col in categorical_cols:
            if col in input_df.columns and col in label_encoders:
                encoder = label_encoders[col]
                original_value = input_df[col].iloc[0]
                
                # Check if it's a proper LabelEncoder with transform method
                if hasattr(encoder, 'transform'):
                    # It's a sklearn LabelEncoder
                    input_df[col] = encoder.transform(input_df[col])
                    print(f"Encoded {col}: {original_value} → {input_df[col].iloc[0]}")
                elif hasattr(encoder, 'classes_'):
                    # It has classes but no transform - create mapping
                    mapping = {val: idx for idx, val in enumerate(encoder.classes_)}
                    input_df[col] = input_df[col].map(mapping)
                    print(f"Mapped {col}: {original_value} → {input_df[col].iloc[0]}")
                elif isinstance(encoder, (np.ndarray, list)):
                    # It's a numpy array or list - map values manually
                    mapping = {val: idx for idx, val in enumerate(encoder)}
                    input_df[col] = input_df[col].map(mapping)
                    print(f"Mapped {col}: {original_value} → {input_df[col].iloc[0]}")
                else:
                    print(f"Warning: Unknown encoder type for {col}: {type(encoder)}")
        
        # Convert to numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        print(f"\nProcessed DataFrame:\n{input_df}")
        print(f"DataFrame dtypes:\n{input_df.dtypes}")
        
        # Scale ONLY numerical features
        if hasattr(scaler, 'transform'):
            # Scale only the numerical columns
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            input_scaled = input_df.values
            print(f"✓ Data scaled successfully (scaled {len(numerical_cols)} numerical features)")
        else:
            print(f"WARNING: Scaler doesn't have transform method. Using unscaled data.")
            input_scaled = input_df.values
        
        print(f"Scaled data shape: {input_scaled.shape}")
        
        # Make prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            print(f"✓ Prediction: {prediction}, Probability: {probability}")
        else:
            raise ValueError(f"Model object doesn't have predict method. Type: {type(model)}")
        
        # Generate SHAP values
        shap_values = explainer(input_scaled)
        
        # Create SHAP waterfall plot with proper feature names
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create SHAP Explanation object with feature names
        shap_explanation = shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0] if hasattr(shap_values, 'base_values') else explainer.expected_value,
            data=input_scaled[0],
            feature_names=feature_names
        )
        
        shap.plots.waterfall(shap_explanation, max_display=15, show=False)
        plt.title('SHAP Feature Contribution Analysis', fontsize=14, fontweight='bold', pad=20)
        
        # Save plot to base64
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Generate recommendations based on top risk factors
        recommendations = []
        interpretation_guide = []
        
        if prediction == 1:  # High risk - will leave
            # Get top contributing features to attrition risk
            feature_contributions = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Value': shap_values.values[0],
                'Feature_Value': input_scaled[0]
            })
            
            # Get top 3 risk factors (positive SHAP values = increase risk)
            top_risks = feature_contributions[feature_contributions['SHAP_Value'] > 0].nlargest(3, 'SHAP_Value')
            
            for _, row in top_risks.iterrows():
                feature = row['Feature']
                
                if feature == 'OverTime':
                    recommendations.append("⚠️ Address workload: Employee is working overtime - review work-life balance")
                elif 'MonthlyIncome' in feature or 'Salary' in feature or 'HourlyRate' in feature:
                    recommendations.append("💰 Compensation review: Consider salary adjustment or bonus")
                elif 'YearsAtCompany' in feature or 'YearsSinceLastPromotion' in feature:
                    recommendations.append("📈 Career development: Discuss promotion opportunities and growth path")
                elif 'JobSatisfaction' in feature or 'EnvironmentSatisfaction' in feature or 'WorkLifeBalance' in feature:
                    recommendations.append("😊 Engagement: Schedule 1-on-1 to discuss job satisfaction concerns")
                elif 'DistanceFromHome' in feature:
                    recommendations.append("🏠 Flexibility: Consider remote work or flexible schedule options")
                elif 'TotalWorkingYears' in feature or 'Age' in feature:
                    recommendations.append("🎯 Career stage: Tailor retention strategy to career stage needs")
                elif 'NumCompaniesWorked' in feature:
                    recommendations.append("🔄 Job hopping pattern: Address stability concerns and career growth")
                elif 'Department' in feature or 'JobRole' in feature:
                    recommendations.append("💼 Role alignment: Review if current role matches employee's skills and interests")
                else:
                    recommendations.append(f"📊 Review {feature}: Contributing to attrition risk")
            
            interpretation_guide = [
                "📖 How to read the SHAP plot:",
                "• Red bars (→) = Features pushing toward WILL LEAVE",
                "• Blue bars (←) = Features pushing toward WILL STAY",
                "• Longer bars = Stronger impact on prediction",
                "• Values show the magnitude of each feature's contribution"
            ]
        else:  # Low risk - will stay
            # Get retention factors (negative SHAP values = decrease risk)
            feature_contributions = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Value': shap_values.values[0],
                'Feature_Value': input_scaled[0]
            })
            
            top_retention = feature_contributions[feature_contributions['SHAP_Value'] < 0].nsmallest(3, 'SHAP_Value')
            
            recommendations.append("✅ RETENTION SUCCESS FACTORS:")
            for _, row in top_retention.iterrows():
                feature = row['Feature']
                recommendations.append(f"• {feature} is helping retain this employee")
            
            recommendations.append("\n💡 Continue to:")
            recommendations.append("• Maintain current compensation and benefits")
            recommendations.append("• Keep work-life balance initiatives")
            recommendations.append("• Regular check-ins to ensure continued satisfaction")
            
            interpretation_guide = [
                "📖 How to read the SHAP plot:",
                "• Blue bars (←) = Features keeping employee (WILL STAY)",
                "• Red bars (→) = Minor risk factors to monitor",
                "• Longer bars = Stronger impact on prediction",
                "• Overall negative score = Strong likelihood to stay"
            ]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_text': 'WILL LEAVE' if prediction == 1 else 'WILL STAY',
            'probability_leave': float(probability[1]),
            'probability_stay': float(probability[0]),
            'confidence': float(max(probability)),
            'shap_plot': plot_base64,
            'recommendations': recommendations,
            'interpretation_guide': interpretation_guide
        }
        
        print(f"✓ Returning result: {result['prediction_text']}")
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("❌ ERROR occurred:")
        print(error_details)
        return jsonify({'error': str(e), 'details': error_details}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': 'loaded'})

# Updated for production (still works locally)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)