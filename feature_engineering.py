import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance

import shap


def feature_Selection(x_train, y_train, x_test, y_test):

    # Train Random Forest
    print("\nTraining Random Forest")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Permutation Importance
    print("\nPermutation Importance")

    perm = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "Feature": x_test.columns,
        "Importance": perm.importances_mean
    }).sort_values(by="Importance", ascending=False)

    print(perm_df)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=perm_df.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Permutation Importance")

    plt.tight_layout()

    plt.show()

    # SHAP Explainability
    print("\nSHAP Feature Explanation")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(x_test)

    # Global SHAP summary plot
    shap.summary_plot(
        shap_values.values,
        x_test,
        max_display=20
    )

    # Select one sample
    i = 0

    features = x_test.columns
    feature_values = x_test.iloc[i].values

    shap_array = shap_values.values

    print(shap_array.shape)

    # Handle binary vs multiclass SHAP output
    if shap_array.ndim == 2:

        shap_vals = shap_array[i]

    elif shap_array.ndim == 3:

        class_index = 1
        shap_vals = shap_array[i, :, class_index]

    else:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_array.shape}"
        )

    print(len(features), len(feature_values), len(shap_vals))

    # SHAP explanation table
    single_shap_df = pd.DataFrame({
        "Feature": features,
        "Feature Value": feature_values,
        "SHAP Value": shap_vals
    })

    print(single_shap_df.head(2))
    print(single_shap_df.iloc[:2])

    # Verify prediction breakdown
    base_value = shap_values.base_values[i]
    shap_sum = shap_vals.sum()

    print("\nBase value:", base_value)
    print("Sum of SHAP values:", shap_sum)

    print(
        "Final prediction (approx):",
        base_value + shap_sum
    )

    # Waterfall Plot
    if shap_array.ndim == 3:

        class_index = 1

        shap.plots.waterfall(
            shap_values[i, :, class_index]
        )

    else:

        shap.plots.waterfall(
            shap_values[i]
        )

    plt.show()

    return perm_df