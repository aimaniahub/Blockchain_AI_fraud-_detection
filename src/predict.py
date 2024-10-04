# src/predict.py

import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.preprocess_data import preprocess_data


def predict_new_data(new_data):
    # Load the saved model
    with open('models/fraud_detection_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Preprocess the new data (same steps as before)
    new_data_cleaned = new_data.copy()
    
    # Convert 'value' from Wei to Ether
    new_data_cleaned['value_in_ether'] = new_data_cleaned['value'].astype(float) / 10**18

    # Label encoding for 'from' and 'to' addresses
    new_data_cleaned['from_encoded'] = new_data_cleaned['from'].astype('category').cat.codes
    new_data_cleaned['to_encoded'] = new_data_cleaned['to'].astype('category').cat.codes

    # Min-Max scaling for numerical features
    scaler = MinMaxScaler()
    new_data_cleaned[['value_in_ether', 'gas', 'gasPrice', 'from_encoded', 'to_encoded']] = scaler.fit_transform(
        new_data_cleaned[['value_in_ether', 'gas', 'gasPrice', 'from_encoded', 'to_encoded']]
    )

    # Features for prediction
    X_new = new_data_cleaned[['value_in_ether', 'gas', 'gasPrice', 'from_encoded', 'to_encoded']]

    # Make predictions
    predictions = model.predict(X_new)

    # Add predictions to the original data
    new_data['is_fraud'] = predictions

    # Return the dataframe with prediction results
    return new_data

if __name__ == "__main__":
    # Example new data
    new_data = pd.DataFrame({
        'value': [1234567890000000000, 300000000000000000],
        'gas': [21000, 21000],
        'gasPrice': [20000000000, 1500000000],
        'from': ['0xabc123...', '0xdef456...'],
        'to': ['0xghi789...', '0xjkl012...']
    })
    
    # Get prediction results
    results = predict_new_data(new_data)
    
    # Print the detailed results
    print("Prediction Results:\n", results)
