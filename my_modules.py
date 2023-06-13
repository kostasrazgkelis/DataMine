import numpy as np
import pandas as pd


def count_same_values(df1, df2):
    arr1 = df1.to_numpy()
    arr2 = df2.to_numpy()

    count_same = np.sum(arr1 == arr2, axis=1)

    return count_same


def custom_f1_score(y_test, y_pred):
    # cast to dataframe if it's not already
    if isinstance(y_test, list):
        y_test = pd.DataFrame(y_test)

    if isinstance(y_pred, list):
        y_pred = pd.DataFrame(y_pred)

    assert y_pred.shape == y_test.shape, 'the two dataframes should have the same shape'

    dividend = 2 * count_same_values(y_pred,y_test)
    divisor = y_pred.shape[1] + y_test.shape[1]

    f1score = np.sum(dividend / divisor) / y_pred.shape[0]
    
    return f1score