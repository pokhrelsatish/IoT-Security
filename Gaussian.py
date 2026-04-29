from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_gaussian_model(x_train_selected, y_train, x_test_selected, y_test):
    
    # Create model
    model = GaussianNB()
    
    # Train model
    model.fit(x_train_selected, y_train)
    
    # Predict on test data
    y_pred = model.predict(x_test_selected)
    print("Output with GaussianNB ML Model")
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