from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('titanic_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()

    # Convert JSON data into a DataFrame
    input_data = pd.DataFrame(data)

    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
