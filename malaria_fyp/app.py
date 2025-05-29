from flask import Flask, request, jsonify, render_template, send_from_directory
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and encoder
try:
    model = joblib.load('C:/Users/HP/Desktop/malaria/malaria_frontpage/malaria_model.pkl')
    strength_encoder = joblib.load('C:/Users/HP/Desktop/malaria/malaria_frontpage/strength_encoder.pkl')
except Exception as e:
    print(f"Error loading model or encoder: {str(e)}")
    model, strength_encoder = None, None

def load_data(filepath):
    try:
        df = pd.read_excel(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Load the data
df = load_data("C:/Users/HP/Documents/dosages2.xlsx")

# Feature importance from XGBoost model (from notebook)
feature_importance = {
    'SEVERITY': 0.1497,
    'PREGNANCY_FIRST': 0.1075,
    'PREGNANCY_SECOND': 0.1921,
    'PREGNANCY_THIRD': 0.2073,
    'NOT_PREGNANT': 0.1524,
    'WEIGHT': 0.1910,
    'LIVER': 0.0,
    'LUNG': 0.0,
    'HEART': 0.0,
    'KIDNEY': 0.0,
    'HYPERTENSION': 0.0
}

def preprocess_input(data):
    """Preprocess user input for prediction"""
    try:
        # Map inputs to encoded values
        severity = 1 if data.get('severity', '').lower() == 'severe' else 0
        pregnancy = data.get('pregnancy', '').lower()
        trimester = data.get('trimester', '')
        
        pregnancy_first = 1 if pregnancy == 'yes' and trimester == '1' else 0
        pregnancy_second = 1 if pregnancy == 'yes' and trimester == '2' else 0
        pregnancy_third = 1 if pregnancy == 'yes' and trimester == '3' else 0
        not_pregnant = 1 if pregnancy == 'no' else 0
        
        # Clean weight
        weight = float(data.get('weight', 0))
        if weight <= 0:
            raise ValueError("Invalid weight")
        
        # Handle "none" checkbox for medical conditions
        none_checked = data.get('none', '').lower() == 'yes'
        
        # If "none" is checked, set all medical conditions to 0
        if none_checked:
            liver = lung = kidney = heart = hypertension = 0
        else:
            # Otherwise, encode based on individual checkboxes
            liver = 1 if data.get('liver', '').lower() == 'yes' else 0
            lung = 1 if data.get('lung', '').lower() == 'yes' else 0
            kidney = 1 if data.get('kidney', '').lower() == 'yes' else 0
            heart = 1 if data.get('heart', '').lower() == 'yes' else 0
            hypertension = 1 if data.get('hypertension', '').lower() == 'yes' else 0
        
        # Create feature vector
        features = [severity, pregnancy_first, pregnancy_second,
                    pregnancy_third, not_pregnant, weight,
                    liver, lung, heart, kidney, hypertension]
        
        return np.array([features]), {
            'severity': 'Severe' if severity == 1 else 'Mild',
            'pregnancy': pregnancy.capitalize(),
            'trimester': trimester if trimester else 'N/A',
            'weight': weight,
            'liver': 'Yes' if liver else 'No',
            'lung': 'Yes' if lung else 'No',
            'kidney': 'Yes' if kidney else 'No',
            'heart': 'Yes' if heart else 'No',
            'hypertension': 'Yes' if hypertension else 'No',
            'none': 'Yes' if none_checked else 'No'
        }
    
    except Exception as e:
        raise ValueError(f"Error processing input: {str(e)}")

def predict_drug(features, severity):
    """Make prediction and format results with severity filtering"""
    try:
        predicted_code = model.predict(features)[0]
        predicted_strength = strength_encoder.inverse_transform([predicted_code])[0]
        
        # Get all drugs matching predicted strength AND severity
        recommended_drugs = df[
            (df['STRENGTHS'] == predicted_strength) & 
            (df['SEVERITY'] == severity)
        ]
        
        # If no drugs found for severe, fallback to default severe treatment
        if severity == "severe" and recommended_drugs.empty:
            recommended_drugs = df[df['SEVERITY'] == "severe"]
            
        # Get top 10 unique drugs
        results = recommended_drugs[['NAME', 'STRENGTHS', 'FORM']]\
            .drop_duplicates()\
            .head(10)\
            .to_dict('records')
            
        return predicted_strength, results
    except Exception as e:
        raise ValueError(f"Error in prediction: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features, input_data = preprocess_input(data)
        severity = data.get('severity', 'mild')
        predicted_strength, drugs = predict_drug(features, severity)
        return jsonify({
            'status': 'success',
          
            'drugs': drugs,
            'input_data': input_data,
            'feature_importance': feature_importance
        })
    except Exception as e:
        print("Error during prediction:", str(e))
        return False
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/random_drugs', methods=['POST'])
def random_drugs():
    try:
        data = request.get_json()
        features, input_data = preprocess_input(data)
        severity = data.get('severity', 'mild')
        predicted_strength, drugs = predict_drug(features, severity)
        
        # Get previously shown indices from request
        shown_indices = set(data.get('shown_indices', []))
        
        # Get all possible indices
        all_indices = list(range(len(drugs)))
        
        # Get available indices (not shown before)
        available_indices = [i for i in all_indices if i not in shown_indices]
        
        if not available_indices:
            # If all have been shown, reset and show all again
            available_indices = all_indices
            shown_indices = set()
        
        # Select up to 5 random drugs from available options
        np.random.shuffle(available_indices)
        selected_indices = available_indices[:min(10, len(available_indices))]
        
        # Prepare response with new drugs and updated shown indices
        response_drugs = [drugs[i] for i in selected_indices]
        new_shown_indices = shown_indices.union(set(selected_indices))
        
        return jsonify({
            'status': 'success',
            'drugs': response_drugs,
            'shown_indices': list(new_shown_indices),
            'input_data': input_data,
            'feature_importance': feature_importance
        })
            
    except Exception as e:
        print("Error during random drug selection:", str(e))
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400
if __name__ == '__main__':
    if model is None or strength_encoder is None or df is None:
        print("Cannot start application: Model, encoder, or data not loaded properly")
    else:
        app.run(debug=True, port=5000)