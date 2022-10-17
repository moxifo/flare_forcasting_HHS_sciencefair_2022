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


# @Eugene, please read these comments and remove them afterwards.
# 1. Stick to one naming convention. E.g., instead of "newcsvfile" use
# "new_csv_file", and instead of "Xdataframe" use "x_dataframe". This
# convention is called Snake Case.
#
# 2. As you can see in this scrip, PyCharm is not underlining my
# code/comments. Make sure you follow all good practices that IDE
# suggests to avoid those underlined pieces. See the yellow bulb next to each
# underlined piece and follow
# the suggestions.
#
# 3. Whenever you need to pass a file path to a method, note that you CANNOT
# use your local machine's paths. For example, you cannot use
# 'C:/Users/eugen/Downloads/test111'. Instead, you should use a relative path
# so that it works on anyone's computer, who has this project. So,
# maybe something like this: '../data/output/test111.csv'.
