from flask import Flask, request, render_template
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
    try:
        # Extract inputs from request
        temperature = float(request.form['temperature'])
        voltage_measured = float(request.form['voltage_measured'])
        voltage_load = float(request.form['voltage_load'])

        # Perform prediction using the loaded model
        prediction = model.predict(np.array([[temperature, voltage_measured, voltage_load]]))

        # Format the prediction result
        result = {'prediction': prediction[0]}

        return render_template('result.html', prediction_text=f'Predicted Remaining Useful Life (RUL): {result["prediction"]} cycles')

    except ValueError:
        error_message = "Please enter valid numerical values for temperature, voltage measured, and voltage load."
        return render_template('error.html', error_message=error_message)

    except KeyError as e:
        error_message = f"Error: Missing form field - {str(e)}"
        return render_template('error.html', error_message=error_message)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
