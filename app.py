# app.py

from flask import Flask, render_template, request, jsonify
from src.predict import predict_new_data
from src.visualize_results import visualize_results
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the frontend (form submission)
        value = float(request.form['value'])
        gas = float(request.form['gas'])
        gasPrice = float(request.form['gasPrice'])
        from_address = request.form['from']
        to_address = request.form['to']

        # Prepare the input data for prediction
        new_data = pd.DataFrame({
            'value': [value],
            'gas': [gas],
            'gasPrice': [gasPrice],
            'from': [from_address],
            'to': [to_address]
        })

        # Predict fraudulent or non-fraudulent transactions
        result = predict_new_data(new_data)

        # Visualize the results (optional)
        visualize_results(result)

        # Return the result
        return render_template('index.html', prediction=result['is_fraud'].values[0])

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
