from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load(r"D:\jupyter\liver_disease_model.pkl")  # Ensure this file is in the same directory

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    
    # Convert the data into DataFrame for prediction
    input_data = pd.DataFrame(data['data'])
    
    # Make predictions
    prediction = model.predict(input_data)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
