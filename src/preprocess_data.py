# src/preprocess_data.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def preprocess_data():
    # Load the data
    df = pd.read_csv('data/ethereum_transactions.csv')

    # Handle missing data by dropping rows with any NaN values
    df_cleaned = df.dropna()

    # Convert 'value' from Wei to Ether
    df_cleaned['value_in_ether'] = df_cleaned['value'].astype(float) / 10**18

    # Label encoding for 'from' and 'to' addresses
    df_cleaned['from_encoded'] = df_cleaned['from'].astype('category').cat.codes
    df_cleaned['to_encoded'] = df_cleaned['to'].astype('category').cat.codes

    # Min-Max scaling for numerical features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_cleaned[['value_in_ether', 'gas', 'gasPrice', 'from_encoded', 'to_encoded']])
    df_cleaned[['value_in_ether', 'gas', 'gasPrice', 'from_encoded', 'to_encoded']] = scaled_features

    # Modify the fraud detection condition to lower the threshold or create a better condition
    df_cleaned['is_fraud'] = df_cleaned['gasPrice'].astype(float) > 500000000  # Lowered threshold

    # Optionally, introduce some fake fraudulent samples for testing (remove this for real use)
    df_cleaned.loc[0:5, 'is_fraud'] = True  # Manually mark first 5 as fraud for testing

    # Features (X) and Target (y)
    X = df_cleaned[['value_in_ether', 'gas', 'gasPrice', 'from_encoded', 'to_encoded']]
    y = df_cleaned['is_fraud']

    # Check class distribution
    print("Class distribution before oversampling:\n", y.value_counts())

    # Apply RandomOverSampler to handle class imbalance
    oversample = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversample.fit_resample(X, y)

    # Split the resampled data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print("Class distribution after oversampling:\n", pd.Series(y_train).value_counts())

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
