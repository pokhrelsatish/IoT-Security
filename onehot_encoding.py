import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os


def Encoding_Data(x_train, x_test, y_train, y_test):

    print("Running one-hot encoding")

    # Find categorical columns
    obj_cols = x_train.select_dtypes(include=['object']).columns
    print("Categorical columns:", obj_cols)

    cat_cols = ['proto', 'flgs', 'state']

    # One-hot encoding
    encoder = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    # Fit and transform training data
    x_train_cat = encoder.fit_transform(x_train[cat_cols])

    # Transform test data
    x_test_cat = encoder.transform(x_test[cat_cols])

    # Convert numpy arrays into DataFrames
    x_train_cat = pd.DataFrame(
        x_train_cat,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    x_test_cat = pd.DataFrame(
        x_test_cat,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    # Remove original categorical columns
    x_train_num = x_train.drop(columns=cat_cols).reset_index(drop=True)
    x_test_num = x_test.drop(columns=cat_cols).reset_index(drop=True)

    # Combine numerical and encoded categorical features
    x_train = pd.concat([x_train_num, x_train_cat], axis=1)
    x_test = pd.concat([x_test_num, x_test_cat], axis=1)

    print("x_test after using OneHotEncoder:", x_test.shape)

    # Show current working directory
    print(os.getcwd())

    return x_train, x_test, y_train, y_test