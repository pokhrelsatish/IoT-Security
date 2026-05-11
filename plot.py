
import matplotlib.pyplot as plt


def plot20features(feature_importance):

    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_importance['Feature'].head(20),
        feature_importance['Importance'].head(20)
    )

    plt.gca().invert_yaxis()

    plt.title("Top 20 Important Features")
    plt.xlabel("Importance")

    plt.show()