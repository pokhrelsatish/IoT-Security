from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def train_mlp(x_train_selected, y_train, x_test_selected, y_test):

    # Scaling is REQUIRED for MLP
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_selected)
    x_test_scaled = scaler.transform(x_test_selected)

    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # 2 hidden layers
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    )

    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    print("Output with MLP Classifier Model (Neural Network)")

    # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler