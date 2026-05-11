import pandas as pd   # Main library for data handling (DataFrames, CSV, tables)
import sys            # Python interacting with the system


def Clean_dataset():

    # Load Dataset
    ds = pd.read_csv(
        r'F:\E drive Old computer\D-drive\Project\Dataset\CSV_BoT-IoT\New Dataset\IoT_Botnet_Dataset_1.csv',
        encoding='utf-8',
        engine='python'
    )

    MB = 1024 * 1024
    print("Pandas dataframe size %d MB " % (sys.getsizeof(ds) / MB))

    # Print dataset shape
    print(ds.shape)

    rows_before_drop = ds.shape[0]

    # Remove missing values and duplicates
    ds = ds.dropna()
    ds = ds.drop_duplicates()

    # Removing labels, identifiers, and leakage columns
    ds = ds.drop(columns=[
        'saddr', 'daddr', 'category', 'subcategory',
        'pkSeqID', 'stime', 'sport', 'dport', 'seq'
    ])

    rows_after_drop = ds.shape[0]

    print("Rows removed: %d" % (rows_before_drop - rows_after_drop))
    print("Clean Rows and Columns in dataset:", ds.shape)

    return ds