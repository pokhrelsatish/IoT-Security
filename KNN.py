from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, learning_curve
import matplotlib.pyplot as plt
import numpy as np


def train_knn(x_train_selected, y_train, x_test_selected, y_test):

    # Scaling
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train_selected)
    x_test_scaled = scaler.transform(x_test_selected)

    # Model
    model = KNeighborsClassifier(
        n_neighbors=5
    )

    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    y_prob = model.predict_proba(x_test_scaled)[:, 1]

    print("Output with KNN Model")

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC Score:", roc_auc)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"KNN (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # Learning curve
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        x_train_scaled,
        y_train,
        cv=cv,
        scoring='accuracy'
    )

    plt.figure()

    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train Score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="CV Score")

    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()

    plt.show()

    return roc_auc, acc