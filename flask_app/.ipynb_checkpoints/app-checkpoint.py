import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

app = Flask(__name__)

# Load saved models
iso_forest = joblib.load('../models/iso_forest_model.pkl')
lof = joblib.load('../models/lof_model.pkl')
oc_svm = joblib.load('../models/oc_svm_model.pkl')
tfidf = joblib.load('../models/tfidf_vectorizer.pkl')


# Function to extract features from log data
def extract_features(log_data):
    # Initialize TF-IDF Vectorizer and LabelEncoder (using pre-trained ones)
    tfidf_features = tfidf.transform(log_data).toarray()
    # Assume 'Level_encoded' column is added in preprocessing before training
    level_encoded = np.array([0] * len(log_data))  # Placeholder for 'Level_encoded' (you can change as needed)
    features = np.hstack((level_encoded.reshape(-1, 1), tfidf_features))
    return features

# Prediction function using the sequential boosting model
def predict_anomalies(log_data):
    features = extract_features(log_data)
    
    # Step 1: Isolation Forest
    pred_iso = iso_forest.predict(features)
    X_refined_1 = features[pred_iso == 1]
    
    # Step 2: Local Outlier Factor
    pred_lof = lof.fit_predict(X_refined_1)
    X_refined_2 = X_refined_1[pred_lof == 1]
    
    # Step 3: One-Class SVM
    pred_svm = oc_svm.predict(X_refined_2)
    
    # Map predictions to labels (1 for normal, 0 for anomaly)
    labels = ['Normal' if p == 1 else 'Anomaly' for p in pred_svm]
    
    return labels

@app.route('/',methods=['GET'])
def index():
    return '''
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Anomaly Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background-color: #f4f4f9;
        }
        h1 {
            text-align: center;
        }
        .form-container {
            max-width: 500px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        input[type="file"] {
            width: 100%;
            padding: 10px;
            margin: 20px 0;
        }
        textarea {
            width: 100%;
            height: 200px;
            padding: 10px;
            margin: 20px 0;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            padding: 10px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background-color: #45a049;
        }
        .output-container {
            margin-top: 20px;
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .output-container p {
            margin: 5px 0;
        }
        .anomaly {
            color: red;
        }
        .normal {
            color: green;
        }
    </style>
</head>
<body>

    <h1>Log Anomaly Detection</h1>

    <div class="form-container">
        <h2>Upload or Paste your log file</h2>
        <form id="logForm" enctype="multipart/form-data">
            <label for="logfile">Upload Log File (Optional):</label>
            <input type="file" name="logfile" id="logfile">
            
            <label for="logcontent">Or Paste Log Content Below:</label>
            <textarea id="logcontent" name="logcontent" placeholder="Paste your log content here..." rows="10"></textarea>
            
            <button type="submit">Submit</button>
        </form>
    </div>

    <div id="outputContainer" class="output-container" style="display:none;">
        <h3>Prediction Results:</h3>
        <div id="predictions"></div>
    </div>

    <script>
        // Handle form submission
        document.getElementById('logForm').addEventListener('submit', function(event) {
            event.preventDefault();

            let formData = new FormData();
            let logFile = document.getElementById('logfile').files[0];
            let logContent = document.getElementById('logcontent').value.trim();

            if (logFile) {
                formData.append('logfile', logFile);
            } else if (logContent) {
                formData.append('logcontent', logContent);
            } else {
                alert('Please either upload a file or paste log content.');
                return;
            }

            // Make the POST request to the Flask server
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                let predictions = data.predictions;
                let outputContainer = document.getElementById('outputContainer');
                let predictionsDiv = document.getElementById('predictions');
                predictionsDiv.innerHTML = '';

                // Loop through predictions and display them
                predictions.forEach((prediction, index) => {
                    let predictionElement = document.createElement('p');
                    predictionElement.textContent = `Log Line ${index + 1}: ${prediction}`;
                    predictionElement.classList.add(prediction.toLowerCase());
                    predictionsDiv.appendChild(predictionElement);
                });

                // Show output container
                outputContainer.style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    </script>

</body>
</html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'logfile' not in request.files and 'logcontent' not in request.form:
        return jsonify({'error': 'No file or content provided'})
    
    # Handle log file upload or pasted content
    log_data = []
    
    if 'logfile' in request.files:
        log_file = request.files['logfile']
        log_data = log_file.read().decode('utf-8').splitlines()
    elif 'logcontent' in request.form:
        log_data = request.form['logcontent'].splitlines()
    
    # Predict anomalies for each log line
    predictions = predict_anomalies(log_data)
    
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
