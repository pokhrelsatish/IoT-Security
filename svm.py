from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def train_svm(x_train_selected, y_train, x_test_selected, y_test):

    # Scaling is REQUIRED for SVM
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_selected)
    x_test_scaled = scaler.transform(x_test_selected)

    model = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42
    )

    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    print("Output with SVM Model")

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler