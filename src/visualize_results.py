# src/visualize_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.predict import predict_new_data


def visualize_results(data):
    # Predict using the new data
    results = predict_new_data(data)

    # Separate fraudulent and non-fraudulent transactions
    fraud_data = results[results['is_fraud'] == True]
    non_fraud_data = results[results['is_fraud'] == False]

    # Line plot for Fraudulent transactions
    plt.figure(figsize=(10, 6))

    # Plot fraudulent transactions with a solid red line
    plt.plot(fraud_data['value'], fraud_data['gasPrice'], label='Fraudulent Transactions', color='red', linestyle='-', marker='o')

    # Plot non-fraudulent transactions with a dashed blue line
    plt.plot(non_fraud_data['value'], non_fraud_data['gasPrice'], label='Non-Fraudulent Transactions', color='blue', linestyle='--', marker='x')

    # Add labels and title
    plt.title('Fraudulent vs Non-Fraudulent Transactions (Line Plot)')
    plt.xlabel('Transaction Value (Wei)')
    plt.ylabel('Gas Price')

    # Add legend to identify the lines
    plt.legend(title='Transaction Type')

    # Show grid for better readability
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example data (can be replaced with more transactions)
    new_data = pd.DataFrame({
        'value': [1234567890000000000, 300000000000000000, 450000000000000000, 987654321000000000],
        'gas': [21000, 21000, 25000, 30000],
        'gasPrice': [20000000000, 1500000000, 30000000000, 100000000000],
        'from': ['0xabc123...', '0xdef456...', '0xghi789...', '0xjkl012...'],
        'to': ['0xghi789...', '0xjkl012...', '0xabc123...', '0xdef456...']
    })
    
    # Visualize the results with lines
    visualize_results(new_data)
