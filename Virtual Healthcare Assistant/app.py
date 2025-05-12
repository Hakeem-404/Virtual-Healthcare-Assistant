# app.py
from flask import Flask, request, render_template, jsonify
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load models and tokenizer
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('./biobert_tokenizer_action')
    
    # Load models
    model_action = BertForSequenceClassification.from_pretrained('./biobert_model_action')
    model_condition = BertForSequenceClassification.from_pretrained('./biobert_model_condition')
    
    # Load label encoders
    with open('label_encoder_action.pkl', 'rb') as f:
        label_encoder_action = pickle.load(f)
    with open('label_encoder_condition.pkl', 'rb') as f:
        label_encoder_condition = pickle.load(f)
    
    # Move models to appropriate device
    model_action.to(device)
    model_condition.to(device)
    
    return tokenizer, model_action, model_condition, label_encoder_action, label_encoder_condition, device

# Load all required models and encoders
tokenizer, model_action, model_condition, label_encoder_action, label_encoder_condition, device = load_models()

def predict_medical_advice(patient_data):
    try:
        # Validate inputs
        required_fields = ['symptoms', 'age', 'gender', 'insurance_status', 'ethnicity',
                         'region', 'socioeconomic_status', 'severity', 'duration',
                         'chronic_condition', 'allergies', 'previous_visits']
        
        for field in required_fields:
            if field not in patient_data or not patient_data[field]:
                return {'error': f"Missing required field: {field}"}

        # Check for vague inputs
        vague_inputs = ['maybe', 'may', 'perhaps', 'sometimes', 'kind of', 'sort of', 
                       'not sure', 'might be', 'could be', 'possibly', 'i think', 'dunno', 'might', 'probably']
        
        for term in vague_inputs:
            if term in patient_data['symptoms'].lower():
                return {'warning': f"Please be more specific about your symptoms. Avoid terms like '{term}'."}

        # Prepare input text
        combined_text = f"""
            {patient_data['symptoms']} 
            age: {patient_data['age']}
            gender: {patient_data['gender']} 
            insurance: {patient_data['insurance_status']}
            ethnicity: {patient_data['ethnicity']}
            region: {patient_data['region']}
            ses: {patient_data['socioeconomic_status']} 
            severity: {patient_data['severity']} 
            duration: {patient_data['duration']} 
            condition: {patient_data['chronic_condition']}
            allergies: {patient_data['allergies']}
            visits: {patient_data['previous_visits']} 
        """

        def get_prediction_confidence(model_outputs):
            probabilities = torch.nn.functional.softmax(model_outputs.logits, dim=-1)
            confidence_score = torch.max(probabilities).item()
            return confidence_score
        
        # Tokenize input
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            action_outputs = model_action(**inputs)
            condition_outputs = model_condition(**inputs)
            

        action_pred = torch.argmax(action_outputs.logits, dim=-1)
        condition_pred = torch.argmax(condition_outputs.logits, dim=-1)
        action_confidence = get_prediction_confidence(action_outputs)
        condition_confidence = get_prediction_confidence(condition_outputs)

        # Convert predictions to labels
        predicted_action = label_encoder_action.inverse_transform(action_pred.cpu().numpy())[0]
        predicted_condition = label_encoder_condition.inverse_transform(condition_pred.cpu().numpy())[0]

        # Format predictions with confidence percentages
        action_with_confidence = f"{predicted_action} || Confidence level: {action_confidence * 100:.2f}%"
        condition_with_confidence = f"{predicted_condition} || Confidence level: {condition_confidence * 100:.2f}%"

        return {
            'suggested_action': action_with_confidence,
            'potential_condition': condition_with_confidence,
            'disclaimer': '''This is AI-generated advice and should not replace professional medical consultation. 
            If you're experiencing severe chest pain, difficulty breathing, or other life-threatening symptoms, please seek emergency medical care immediately.'''
        }

    except Exception as e:
        return {'error': f"An error occurred: {str(e)}"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = predict_medical_advice(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)