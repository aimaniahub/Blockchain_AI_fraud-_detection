# src/fetch_data.py

import requests
import pandas as pd

API_KEY = "Z4BHF4KGD39JDQ4CXRCQ4S1YT6PKQGT14M"  # Replace with your actual API key
address = "0x401230D64a5d996697cbB7919dd87103B730A693"  # Replace with the Ethereum address you want to analyze

# Fetch transactions
def fetch_transactions():
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data["status"] == "1":
        transactions = data["result"]
        df = pd.DataFrame(transactions)

        # Select relevant columns and clean up
        df = df[['blockNumber', 'timeStamp', 'hash', 'from', 'to', 'value', 'gas', 'gasPrice', 'isError']]
        
        # Explicitly convert the 'timeStamp' column to integers before converting to datetime
        df['timeStamp'] = pd.to_datetime(pd.to_numeric(df['timeStamp']), unit='s')

        # Save to CSV
        df.to_csv('data/ethereum_transactions.csv', index=False)
        print("Data fetched and saved successfully.")
    else:
        print(f"Error: {data['message']}")

if __name__ == "__main__":
    fetch_transactions()
