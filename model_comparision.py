import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def model_comparison_dashboard(results):

    # -----------------------------
    # Create Summary Table
    # -----------------------------
    df = pd.DataFrame(
        results,
        index=["ROC-AUC", "Accuracy"]
    ).T

    print("\nMODEL PERFORMANCE SUMMARY")
    print(df)

    # -----------------------------
    # Prepare data for plotting
    # -----------------------------
    models = df.index.tolist()
    roc = df["ROC-AUC"].values
    accuracy = df["Accuracy"].values

    x = np.arange(len(models))
    width = 0.35

    # -----------------------------
    # Plot Comparison Bar Chart
    # -----------------------------
    plt.figure(figsize=(10, 6))

    plt.bar(x - width / 2, accuracy, width, label="Accuracy")
    plt.bar(x + width / 2, roc, width, label="ROC-AUC")

    plt.xticks(x, models, rotation=30)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return df