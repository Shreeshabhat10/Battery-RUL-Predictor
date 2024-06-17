from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask application
app = Flask(__name__, template_folder='templates')

# Load the pickled model
model = pickle.load(open('model.pkl', 'rb'))

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract inputs from request
    temperature = float(request.form['temperature'])
    voltage_load = float(request.form['voltage_measured'])
    voltage_discharge = float(request.form['voltage_load'])

    # Perform prediction using the loaded model
    prediction = model.predict(np.array([[temperature, voltage_load, voltage_discharge]]))

    # Format the prediction result
    result = {'prediction': prediction[0]}

    return render_template('result.html', prediction_text=f'Capacity of the Battery is : {result["prediction"]} Ah')

if __name__ == '__main__':
    app.run(debug=True)
