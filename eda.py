import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Generate statistics
def generate_statistics(df):

    clean = df[df["attack"] == 0]
    infected = df[df["attack"] == 1]

    # Summary statistics
    clean_stats = clean.describe()
    infected_stats = infected.describe()

    print("Clean Packet:\n", clean_stats)
    print("\nInfected Packet:\n", infected_stats)

    return


# Full Correlation Heatmap
def plot_heatmap(
    x_train,
    save_path=r"F:\E drive Old computer\D-drive\Project\Python files\heatmap.png"
):

    corr = x_train.corr()

    # Hide upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=False
    )

    plt.title("Feature Correlation Heatmap", fontsize=16)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


# Filtered Strong Correlation Heatmap
def plot_filtered_heatmap(x_train, threshold=0.7):

    corr = x_train.corr()

    # Keep strong correlations only
    corr_filtered = corr.where(abs(corr) >= threshold)

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        corr_filtered,
        cmap="coolwarm",
        annot=False,
        linewidths=0.5
    )

    plt.title(f"Correlation Heatmap (|corr| ≥ {threshold})")

    plt.tight_layout()

    plt.show()


# Clustered Heatmap
def plot_clustermap(x_train):

    corr = x_train.corr()

    sns.clustermap(
        corr,
        cmap="coolwarm",
        figsize=(12, 10),
        linewidths=0.5
    )

    plt.show()


# Top Features Heatmap
def plot_top_features_heatmap(x_train, top_n=20):

    top_features = x_train.columns[:top_n]

    plot_heatmap(x_train[top_features])