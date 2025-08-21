import numpy as np
import pickle
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Define the LogisticRegression class as it was used during model training
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        # Add a column of ones for the bias term
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize weights (including bias)
        self.weights = np.zeros(X.shape[1])
        
        # Gradient Descent
        for epoch in range(self.epochs):
            predictions = self.sigmoid(np.dot(X, self.weights))
            errors = y - predictions
            gradient = -np.dot(X.T, errors) / X.shape[0]
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias column
        predictions = self.sigmoid(np.dot(X, self.weights))
        return (predictions >= 0.5).astype(int)  # Return 0 or 1 based on threshold
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# Load the trained logistic regression model
with open('osteoporosis_model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Route for home page (display the input form)
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    features = []
    try:
        # Collect data for all 14 features
        for i in range(1, 15):  # Adjust the range for all 14 features
            feature_value = float(request.form[f'feature{i}'])
            features.append(feature_value)

        # Convert to NumPy array
        X = np.array(features, dtype=np.float64).reshape(1, -1)

        # Predict using the logistic regression model
        prediction = model.predict(X)

        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
