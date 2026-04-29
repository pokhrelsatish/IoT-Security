from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_rf(x_train_selected, y_train, x_test_selected, y_test):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1 #use all CPU Cores
    )

    model.fit(x_train_selected, y_train)
    y_pred = model.predict(x_test_selected)
    print("Output with Random Forest Classifier Model")
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model
