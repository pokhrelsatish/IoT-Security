import sys
import matplotlib
import pandas as pd
import sklearn
import imblearn
import openpyxl

# To see version of the libraries used
def see_version():
    print('matplotlib: {}'.format(matplotlib.__version__))
    print('sys: {}'.format(sys.version))
    print('pandas: {}'.format(pd.__version__))
    print('sklearn: {}'.format(sklearn.__version__))
    print('imblearn: {}'.format(imblearn.__version__))
    print('openpyxl: {}'.format(openpyxl.__version__))
    return


def main():

    # Data Cleaning
    import Data_Cleaning
    ds = Data_Cleaning.Clean_dataset()

    # Train-test split
    import train_test_split as split
    x_train, x_test, y_train, y_test = split.Split_Data(ds)

    # Encoding
    import onehot_encoding
    x_train, x_test, y_train, y_test = onehot_encoding.Encoding_Data(
        x_train, x_test, y_train, y_test
    )

    # EDA: Exploratory Data Analysis
    import eda
    eda.generate_statistics(ds)
    eda.plot_heatmap(x_train)
    eda.plot_filtered_heatmap(x_train)
    eda.plot_clustermap(x_train)
    eda.plot_top_features_heatmap(x_train)

    # SMOTE balancing
    import balance_data_smote as smt
    x_train, y_train = smt.smote(x_train, y_train)

    # Feature Importance
    from feature_engineering import feature_Selection
    feature_importance = feature_Selection(
        x_train, y_train, x_test, y_test
    )

    # Visualization of feature impact
    import plot
    plot.plot20features(feature_importance)

    # Feature Selection
    import feature_Selection as fs
    x_train_selected, x_test_selected = fs.selected_feature(
        feature_importance, x_train, x_test
    )

    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # Class distribution
    print("Class distribution in training dataset")
    print(y_train.value_counts())

    print("Class distribution in test dataset")
    print(y_test.value_counts())

    # Store model results
    results = {}

    # Gaussian Naive Bayes
    import Gaussian
    roc, accuracy = Gaussian.train_gaussian_model(
        x_train_selected,
        y_train,
        x_test_selected,
        y_test
    )
    results["Gaussian"] = (roc, accuracy)

    #Random Forest
    import random_forest_model as rf
    roc, accuracy = rf.train_rf(
         x_train_selected,
         y_train,
         x_test_selected,
         y_test
     )
    results["Random Forest"] = (roc, accuracy)

    # Logistic Regression
    import Logistic_regression as lr
    roc, accuracy = lr.train_lf(
         x_train_selected,
         y_train,
         x_test_selected,
         y_test
     )
    results["Logistic Regression"] = (roc, accuracy)

    # KNN
    import KNN
    roc, accuracy = KNN.train_knn(
         x_train_selected,
         y_train,
         x_test_selected,
         y_test
     )
    results["KNN"] = (roc, accuracy)

    # MLP
    import MLP
    roc, accuracy = MLP.train_mlp(
         x_train_selected,
         y_train,
         x_test_selected,
         y_test
     )
    results["MLP"] = (roc, accuracy)

    # SVM
    import svm
    roc, accuracy = svm.train_svm(
        x_train_selected,
        y_train,
        x_test_selected,
        y_test
    )
    results["SVM"] = (roc, accuracy)

    # Result Table
    rslt = pd.DataFrame(
        results,
        index=["ROC-AUC", "Accuracy"]
    )

    print(rslt)
    
    import model_comparision
    model_comparision.model_comparison_dashboard(results)

see_version()
main()