import pandas as pd


def normalize_data(data: pd.DataFrame, a, b):
    """
    runs a zero-one normalizer on the data. Each column of the data will be
    normalized separately. To do this, we use the MinMaxScaler function of
    the sklearn.preprocessing package.

    :param data: the dataframe to be normalized
    :param a:min default=0
    :param b:max default=1
    :return:
    """
    from sklearn.preprocessing import MinMaxScaler

    # fitting the scalar function into my data
    data_normalized = MinMaxScaler(feature_range=(a, b)).fit_transform(data)

    # converting the scaled dataset back into the dataset variable type
    data_normalized_df: pd.DataFrame = pd.DataFrame(data_normalized)

    # reading columns back to the new dataframe type variable
    data_normalized_df.columns = ['TOTUSJH_median', 'TOTBSQ_median',
                                  'TOTPOT_median', 'TOTUSJZ_median',
                                  'ABSNJZH_median', 'TOTUSJH_mean',
                                  'TOTBSQ_mean', 'TOTPOT_mean',
                                  'TOTUSJZ_mean', 'ABSNJZH_mean',
                                  'TOTUSJH_min', 'TOTBSQ_min',
                                  'TOTPOT_min',
                                  'TOTUSJZ_min', 'ABSNJZH_min',
                                  'TOTUSJH_max', 'TOTBSQ_max',
                                  'TOTPOT_max', 'TOTUSJZ_max',
                                  'ABSNJZH_max',
                                  'TOTUSJH_std', 'TOTBSQ_std',
                                  'TOTPOT_std', 'TOTUSJZ_std',
                                  'ABSNJZH_std', 'TOTUSJH_skewness',
                                  'TOTBSQ_skewness', 'TOTPOT_skewness',
                                  'TOTUSJZ_skewness',
                                  'ABSNJZH_skewness',
                                  'TOTUSJH_kurtosis',
                                  'TOTBSQ_kurtosis', 'TOTPOT_kurtosis',
                                  'TOTUSJZ_kurtosis',
                                  'ABSNJZH_kurtosis',
                                  'TOTUSJH_last_value',
                                  'TOTBSQ_last_value',
                                  'TOTPOT_last_value',
                                  'TOTUSJZ_last_value',
                                  'ABSNJZH_last_value', 'label']
    return data_normalized_df


def centralize_data(data: pd.DataFrame):
    """
    scales all the data to have a mean of 0. Each collum of the data will be centralized all together.
    To do this, we use the 'StandardScaler' function of the 'sklearn.preprocessing' package.

    :param data:the data frame to be centralized
    :return: the centralized data with the same column names
    """
    from sklearn.preprocessing import StandardScaler

    # fitting the scalar function into my data
    data_centralized = StandardScaler().fit_transform(data)

    # covnerting the scaled dataset back into the dataset variable type
    data_centralized_df: pd.DataFrame = pd.DataFrame(data_centralized)

    # readding collums back to the new dataframe type variable
    data_centralized_df.columns = ['TOTUSJH_median', 'TOTBSQ_median',
                                   'TOTPOT_median', 'TOTUSJZ_median',
                                   'ABSNJZH_median', 'TOTUSJH_mean',
                                   'TOTBSQ_mean', 'TOTPOT_mean',
                                   'TOTUSJZ_mean', 'ABSNJZH_mean',
                                   'TOTUSJH_min',
                                   'TOTBSQ_min', 'TOTPOT_min',
                                   'TOTUSJZ_min', 'ABSNJZH_min',
                                   'TOTUSJH_max', 'TOTBSQ_max',
                                   'TOTPOT_max',
                                   'TOTUSJZ_max', 'ABSNJZH_max',
                                   'TOTUSJH_std', 'TOTBSQ_std',
                                   'TOTPOT_std', 'TOTUSJZ_std',
                                   'ABSNJZH_std',
                                   'TOTUSJH_skewness',
                                   'TOTBSQ_skewness', 'TOTPOT_skewness',
                                   'TOTUSJZ_skewness', 'ABSNJZH_skewness',
                                   'TOTUSJH_kurtosis',
                                   'TOTBSQ_kurtosis', 'TOTPOT_kurtosis',
                                   'TOTUSJZ_kurtosis', 'ABSNJZH_kurtosis',
                                   'TOTUSJH_last_value',
                                   'TOTBSQ_last_value',
                                   'TOTPOT_last_value',
                                   'TOTUSJZ_last_value',
                                   'ABSNJZH_last_value', 'label']
    return data_centralized_df
