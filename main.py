# main.py

from src.fetch_data import fetch_transactions
from src.preprocess_data import preprocess_data

if __name__ == "__main__":
    # Fetch the data
    fetch_transactions()

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data()
