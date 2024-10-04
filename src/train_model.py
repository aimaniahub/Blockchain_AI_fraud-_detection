# src/train_model.py

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from preprocess_data import preprocess_data

# Preprocess the data and get train/test splits
X_train, X_test, y_train, y_test = preprocess_data()

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to a file using pickle
with open('models/fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
