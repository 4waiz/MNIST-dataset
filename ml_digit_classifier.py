from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (8x8 images of digits)
digits = load_digits()
X, y = digits.data, digits.target  # X: pixel values, y: labels (0–9)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression (Machine Learning)
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("ML Accuracy (Logistic Regression):", accuracy)
