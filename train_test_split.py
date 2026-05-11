import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def Split_Data(ds):

    # Features and target
    x = ds.drop(['attack'], axis=1)
    y = ds['attack']

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y
    )

    print("X_train shape:", x_train.shape)
    print("X_test shape :", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)

    # Class distribution
    print("Size of class in dataset")
    print(y_train.value_counts())

    # Visualize class distribution
    plt.figure(figsize=(8, 5))

    sns.countplot(x=y_train)

    plt.title("Count (Traffic)")
    plt.xlabel("Traffic (1: Botnet Traffic, 0: Normal Traffic)")
    plt.ylabel("Count")

    plt.show()

    return x_train, x_test, y_train, y_test