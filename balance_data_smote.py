from imblearn.over_sampling import SMOTE


def smote(x_train, y_train):

    # Check for non-numeric columns
    print(
        "Object columns:",
        x_train.select_dtypes(include=['object']).columns
    )

    # SMOTE model
    smote_model = SMOTE(
        random_state=42,
        sampling_strategy='auto',
        k_neighbors=5
    )

    # Apply SMOTE
    x_train_resampled, y_train_resampled = smote_model.fit_resample(
        x_train,
        y_train
    )

    print("x_train after SMOTE:", x_train_resampled.shape)
    print("y_train after SMOTE:", y_train_resampled.shape)

    return x_train_resampled, y_train_resampled