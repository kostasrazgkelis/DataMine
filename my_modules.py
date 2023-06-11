import numpy as np
import pandas as pd


def count_same_values(df1, df2):
    arr1 = df1
    arr2 = df2

    if type(df1) == pd.core.frame.DataFrame:
        arr1 = df1.to_numpy()
    if type(df2) == pd.core.frame.DataFrame:
        arr2 = df2.to_numpy()

    count_same = np.sum(arr1 == arr2, axis=1)

    return count_same

def custom_f1_score(y_test, y_pred):
    assert y_pred.shape == y_test.shape, 'the two dataframes should have the same shape'

    dividend = 2 * count_same_values(y_pred, y_test)
    divisor = y_pred.shape[1] + y_test.shape[1]

    f1score = np.sum(dividend / divisor) / y_pred.shape[0]

    return f1score