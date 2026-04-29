#New Code
import os #a module (toolbox for working with files & folders
import sys #lets Python interact with the system (Not important)
import matplotlib #Main library for plotting 
from matplotlib import pyplot as plt #importing pyplot module from matplotlib and giving it a nickname called plot (plot is alias not variable) 
import pandas as pd # Main library for data handling (Data frames, CSV, tables)
import sklearn # Main library for Machine Learning models and tools
import imblearn
import openpyxl
from sklearn import model_selection #General module for splitting data and evaluating models
from sklearn.model_selection import KFold #Splits data into K parts (folds), E.g: train on 4 parts, test on 1 (repeat 5 times)
from sklearn import preprocessing #Library to normalize dataframe :Encoding, scaling, Normalization
from sklearn.preprocessing import MinMaxScaler #Transform features by scaling each feature to a given range.
from sklearn.datasets import make_classification # Creates classification data (for testing models and learning data)
from sklearn.metrics import classification_report # Gives Precision, Recall, F1-Score, Support all in one output
from sklearn.metrics import confusion_matrix #lays out predictions vs. actual
from sklearn.metrics import accuracy_score # % of correct predictions
from sklearn.metrics import f1_score # How well model identifies positive class
from sklearn.model_selection import cross_val_score #runs cross validation (returns multiple scores: Accuracy, AUC etc)
from sklearn.feature_selection import SelectKBest # Selects top K important features
from sklearn.feature_selection import chi2 # Statistical test: Higher value = more important feature
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE #Creates synthetic data for minority class: Generates new, artificial samples based on existing ones
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc


#To see version of the models used
def see_version():
  print('matplotlib: {}'.format(matplotlib.__version__))
  print('sys: {}'.format(sys.version))
  print('pandas: {}'.format(pd.__version__))
  print('sklearn: {}'.format(sklearn.__version__))
  print('imblearn: {}'.format(imblearn.__version__))
  print('openpyxl:{}'.format(openpyxl.__version__))
  return

def main():
    ds = Clean_dataset()
    x_train, x_test, y_train, y_test = Split_Data(ds)
    x_train, x_test, y_train, y_test = Encoding_Data(x_train, x_test, y_train, y_test)
    x_train, y_train = smote(x_train, y_train)
#Go to a file/module named random_forest.py and import the function (or object) called train_random_forest
    from random_forest import train_random_forest
    model, y_pred, feature_importance = train_random_forest(x_train, y_train, x_test, y_test)
    import plot
    plot.plot20features(feature_importance)
    #feature_interpretability()
    import feature_Selection as fs
    x_train_selected, x_test_selected = fs.selected_feature(feature_importance,x_train, x_test)
    print("x_train_selected:", x_train_selected.shape)
    print("x_test_selected :", x_test_selected.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)
    import Gaussian
    Gaussian.train_gaussian_model(x_train_selected, y_train, x_test_selected, y_test)
    import random_forest_model as rf
    rf.train_rf(x_train_selected, y_train, x_test_selected, y_test)
    import Logistic_regression as lr
    lr.train_lf(x_train_selected, y_train, x_test_selected, y_test)
    import KNN
    KNN.train_knn(x_train_selected, y_train, x_test_selected, y_test)
    import MLP
    MLP.train_mlp(x_train_selected, y_train, x_test_selected, y_test)
    #import svm
    #svm.train_svm(x_train_selected, y_train, x_test_selected, y_test)
    return

def Clean_dataset():
    ds = pd.read_csv(r'F:\E drive Old computer\D-drive\Project\Dataset\CSV_BoT-IoT\New Dataset\IoT_Botnet_Dataset_1.csv', encoding = 'utf-8', engine = 'python')
    MB = 1024*1024 #To convert bytes to MB (1MB = 1024kb, 1kb = 1024bytes)
    print("Pandas dataframe size %d MB " % (sys.getsizeof(ds)/MB))
    print(ds.shape)
    rows_before_drop = ds.shape[0]
    ds = ds.dropna()
    ds = ds.drop_duplicates()
    ds = ds.drop(columns=['saddr', 'daddr'])
    #print(ds.columns)
    rows_After_drop = ds.shape[0]
    print("Rows removed: %d" %(rows_before_drop - rows_After_drop))
    print("Clean Rows and Columns in dataset:", ds.shape) 
    #print(ds.columns)
   # ds.to_csv(r"F:\E drive Old computer\D-drive\Project\Dataset\CSV_BoT-IoT\New Dataset\clean_data.csv", index=False) #To extract clean dataset to csv
   #ds.to_excel(r"F:\E drive Old computer\D-drive\Project\Dataset\CSV_BoT-IoT\New Dataset\clean_data.xlsx", index=False, engine='openpyxl')
   #feature_interpretability()
    return ds

def Split_Data(ds):
    from sklearn.model_selection import train_test_split # splitting data and evaluating models (Training sets_Learn, Testing sets_evaluate)
    x = ds.drop(['attack', 'category', 'subcategory'], axis =1) #Features (everything except target)
    y = ds['attack'] # Target column
  #Split 20% test, 80% train, reproducible results, keep class balance
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size =0.2, random_state = 42, shuffle =True , stratify = y)  #20% test
    print("X_train shape:", x_train.shape)
    print("X_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    return x_train, x_test, y_train, y_test

def Encoding_Data(x_train, x_test, y_train, y_test):
    from sklearn.preprocessing import OneHotEncoder
    global x_train_cat, x_test_cat
    print(x_train.select_dtypes(include=['object']).columns)
    x_train['sport'] = pd.to_numeric(x_train['sport'], errors='coerce') #Convert invalid -> NAN
    x_train['dport'] = pd.to_numeric(x_train['dport'], errors='coerce') #Convert invalid -> NAN
    x_test['sport'] = pd.to_numeric(x_test['sport'], errors='coerce') #Convert invalid -> NAN
    x_test['dport'] = pd.to_numeric(x_test['dport'], errors='coerce') #Convert invalid -> NAN
    # Drop rows where conversion failed
    x_train = x_train.dropna(subset=['sport', 'dport'])
    x_test = x_test.dropna(subset=['sport','dport'])
    # 🔑 FIX: align y_train
    #.loc[] = select rows by index label, x_train.index = list of valid row indices
    y_train = y_train.loc[x_train.index]
    y_test = y_test.loc[x_test.index]
    #Reset index
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    cat_cols = ['proto', 'flgs', 'state']   #subcategory
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) #Sparse matrix = mostly zeros, stored efficiently to save memory
    x_train_cat = encoder.fit_transform(x_train[cat_cols]) #fit and transform training data
 #fit -> Learn unique value from trainning, transfer -> convert them into onehot encoded format, result: numeric array
    x_test_cat = encoder.transform(x_test[cat_cols]) # Only transforms test data using already learned categories (no fitting again).
    x_train_cat = pd.DataFrame(x_train_cat, columns=encoder.get_feature_names_out(cat_cols)) #Convert trained encoded data to dataframe
    x_test_cat = pd.DataFrame(x_test_cat, columns=encoder.get_feature_names_out(cat_cols))
    x_train_num = x_train.drop(columns=cat_cols).reset_index(drop=True) #
    x_test_num = x_test.drop(columns=cat_cols).reset_index(drop=True)
    x_train = pd.concat([x_train_num, x_train_cat], axis=1)
    x_test = pd.concat([x_test_num, x_test_cat], axis=1)  
    print("x_test after using OneHotEncoder :", x_test.shape)
    #x_train.to_csv(r"F:\E drive Old computer\D-drive\Project\Dataset\CSV_BoT-IoT\New Dataset\Train_Set.csv", index = False) #index false, only actual data saved no extra index column
    #x_test.to_csv(r"F:\E drive Old computer\D-drive\Project\Dataset\CSV_BoT-IoT\New Dataset\Test_Set.csv", index = False)
    print(os.getcwd()) #Shows the Current Working directory
    #print(x_train.dtypes)
    #print(x_test.dtypes)
   # x_train = x_train.fillna(0)
    return x_train, x_test, y_train, y_test

    
def smote(x_train, y_train):
    print(x_train.select_dtypes(include=['object']).columns) # To check which columns are still non-numeric (object type)
    smote_model = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors = 5)
    # Apply SMOTE
    x_train_resampled, y_train_resampled = smote_model.fit_resample(x_train, y_train)
#Here, fit : SMOTE Studies trainning data, resample: generate new data to balance classes
    # Combine into one DataFrame (Optional)
    df_resampled = pd.concat([pd.DataFrame(x_train_resampled), pd.DataFrame(y_train_resampled)], axis=1)
    #df_resampled.to_csv(r"F:\E drive Old computer\D-drive\Project\Dataset\CSV_BoT-IoT\New Dataset\train_set_smote.csv", index = False)
    print ("Final dataframe after SMOTE Size = ",df_resampled.shape)
    return x_train_resampled, y_train_resampled


#def correlations():
 #   ds.corr()['attack']
  #  ds.corr()['attack'].sort_values(ascending=False) #Sort to see strongest feature

#Feature importance / interpretability
#Best approach: Random Forest, Correlation, Groupby Comparision
#def feature_interpretability():
 #   global ds
  #  x = ds.drop('attack', axis =1)
   # y = ds.drop['attack']
#Train a model (Random Forest)
    #model = RandomForestClassifier()
    #model.fit(x, y)
#Get feature importance
    #importance = model.feature_importances_
   # for i, col in enumerate(x.columns):
    #  print(col, ":", importance[i])
#Compare feature means per class
    #ds.groupby('attack').mean()
    #return

#def export():
 #     ds.to_csv("clean_data.csv", index=False)
  #    df.to_excel("clean_data.xlsx", index=False)
   #   return

#Heatmap for visual understanding
#def heatmap():
 # import seaborn as sns
  #import matplotlib.pyplot as plt
  #sns.heatmap(ds.corr(), cmap='coolwarm')
  #plt.show()
  #return

see_version()
main()