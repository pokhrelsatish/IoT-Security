from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def train_knn(x_train_selected, y_train, x_test_selected, y_test):

    # Scaling is VERY important for KNN
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_selected)
    x_test_scaled = scaler.transform(x_test_selected)

    model = KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )

    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    print("Output with KNN Model")

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler