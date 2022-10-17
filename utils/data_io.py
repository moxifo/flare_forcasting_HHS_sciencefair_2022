import pandas as pd


def dump_data(df: pd.DataFrame, path: str):
    """
    stores a given dataframe as a CSV file, with the given filename.
    
    :param df: the pandas dataframe to be stored as a CSV file
    :param path: the path where the data should be stored at
    
    :return: the stored csv file
    """
    # converting the new pd.dataframe into a readable csv file
    new_csv_file = df.to_csv(path, index=False)
    return new_csv_file


def load_data(arg1: pd.DataFrame):
    """
    Separates the features of the data with the outcome and creates a separate dataframe

    :param arg1: the data frame that is going to have its last column removed
    
    :return:
    """
    # deciding which collumns to add and remove
    X = arg1.iloc[:, :-1].values
    # converting the array into a dataset
    Xdataframe = pd.DataFrame(X)
    return Xdataframe

def cut_data(arg1):
    """
    seperates the last two columns from the data
    :param arg1: the data frame that is going to have its last two columns removed
    :return: the circumcised dataframe
    """
    # deciding which collumns to add and remove
    X = arg1.iloc[:, :-3].values
    # converting the array into a dataset
    Xdataframe = pd.DataFrame(X)
    Xdataframe.columns = ['TOTUSJH_median', 'TOTBSQ_median', 'TOTPOT_median', 'TOTUSJZ_median', 'ABSNJZH_median',
                           'TOTUSJH_mean',
                           'TOTBSQ_mean', 'TOTPOT_mean', 'TOTUSJZ_mean', 'ABSNJZH_mean', 'TOTUSJH_min', 'TOTBSQ_min',
                           'TOTPOT_min',
                           'TOTUSJZ_min', 'ABSNJZH_min', 'TOTUSJH_max', 'TOTBSQ_max', 'TOTPOT_max', 'TOTUSJZ_max',
                           'ABSNJZH_max',
                           'TOTUSJH_std', 'TOTBSQ_std', 'TOTPOT_std', 'TOTUSJZ_std', 'ABSNJZH_std', 'TOTUSJH_skewness',
                           'TOTBSQ_skewness', 'TOTPOT_skewness', 'TOTUSJZ_skewness', 'ABSNJZH_skewness',
                           'TOTUSJH_kurtosis',
                           'TOTBSQ_kurtosis', 'TOTPOT_kurtosis', 'TOTUSJZ_kurtosis', 'ABSNJZH_kurtosis',
                           'TOTUSJH_last_value',
                           'TOTBSQ_last_value', 'TOTPOT_last_value', 'TOTUSJZ_last_value', 'ABSNJZH_last_value',
                           'label']
    return Xdataframe


