from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def train_lf(x_train_selected, y_train, x_test_selected, y_test):

    # Scale data (important for Logistic Regression)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_selected)
    x_test_scaled = scaler.transform(x_test_selected)

    # Model
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs"
    )

    # Train
    model.fit(x_train_scaled, y_train)

    # Predict
    y_pred = model.predict(x_test_scaled)

    print("Output with Logistic Regression Model")

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler