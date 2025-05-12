# Medical Symptom Analyzer

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.0+-yellow.svg)

A virtual healthcare assistant that uses Natural Language Processing (NLP) to interpret patient symptom descriptions and provide preliminary medical advice. This project is designed to assist patients in understanding their symptoms before deciding to visit a healthcare professional.

## 🌟 Features

- **Symptom Analysis**: Uses a fine-tuned BioBERT model to analyze patient-described symptoms
- **Dual Prediction**: Provides both a suggested action (Go to Emergency, Take Home Care Measures, Visit Doctor) and a potential condition
- **Confidence Scoring**: Shows prediction confidence levels for transparency
- **Comprehensive Input**: Considers demographic factors, symptom severity, medical history, and more
- **User-Friendly Interface**: Simple web interface designed for users of all technical abilities
- **Safety Mechanisms**: Includes disclaimers and emergency detection features

## 📋 Project Structure

```
/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
│   └── index.html         # Main UI template
├── biobert_model_action/  # Fine-tuned BioBERT model for action prediction
├── biobert_model_condition/ # Fine-tuned BioBERT model for condition prediction
├── biobert_tokenizer_action/ # BioBERT tokenizer
├── label_encoder_action.pkl  # Label encoder for action classes
└── label_encoder_condition.pkl # Label encoder for condition classes
```

## 🧠 Model Details

The system uses two fine-tuned BioBERT models:

1. **Action Prediction Model**: 
   - Base: `dmis-lab/biobert-base-cased-v1.1`
   - Classes: "Go to Emergency", "Take Home Care Measures", "Visit Doctor"
   - Accuracy: ~92%

2. **Condition Prediction Model**:
   - Base: `dmis-lab/biobert-base-cased-v1.1`
   - 18 Classes including: Allergic Reaction, Anxiety, Arthritis, Asthma, Common Cold, Covid-19, Dehydration, Diabetes, Flu, Food Poisoning, Gastroenteritis, Hypertension, Kidney Stones, Migraine, Muscle Strain, None, Sciatica, Sinus Infection
   - Accuracy: ~85%

## 🚀 Installation

### Prerequisites

- Python 3.7+
- PyTorch
- Flask
- Transformers library

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Hakeem-404/Virtual-Healthcare-Assistant.git
   cd Virtual-Healthcare-Assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the fine-tuned models or train your own:
   - If using pre-trained models, unzip them to the appropriate directories
   - If training your own, see the Training section below

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## 🔧 Training Your Own Models

We trained our models on a dataset of 3,600 medical records, including:
- 1,500 records from the primary dataset (data.csv)
- 2,100 additional records from synthetic datasets (synthetic.csv, synthetic1.csv, synthetic2.csv)

To train your own models:

1. Prepare your dataset with the following fields:
   - Demographics: Age, Gender, Ethnicity, Region, Socioeconomic Status
   - Medical information: Symptom Description, Symptom Severity, Duration, Additional Symptoms
   - Medical history: Chronic Condition, Allergies, Previous Visits, Insurance Status
   - Target variables: Potential Condition, Suggested Action

2. Run the training script:
   ```bash
   python train_models.py
   ```

The training process uses the following parameters:
- Learning rate: 2e-5
- Batch size: 4
- Epochs: 5
- Max sequence length: 128 tokens

## 📊 Sample Input/Output

### Input:
```json
{
  "symptoms": "severe headache, pain with nausea",
  "age": 84,
  "gender": "Female",
  "ethnicity": "Caucasian",
  "region": "Urban",
  "socioeconomic_status": "Middle",
  "severity": 10,
  "duration": 30,
  "insurance_status": "Yes",
  "chronic_condition": "None",
  "allergies": "Penicillin",
  "previous_visits": 5
}
```

### Output:
```json
{
  "suggested_action": "Visit Doctor || Confidence level: 87.52%",
  "potential_condition": "Migraine || Confidence level: 82.31%",
  "disclaimer": "This is AI-generated advice and should not replace professional medical consultation. If you're experiencing severe chest pain, difficulty breathing, or other life-threatening symptoms, please seek emergency medical care immediately."
}
```

## ⚠️ Disclaimer

This application is for educational and research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- BioBERT by DMIS Lab for the pre-trained medical language model
- [University of Greenwich](https://www.gre.ac.uk/) for the project guidance (COMP1921 Advanced Topics in Data Science and AI)
