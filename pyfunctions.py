#import datasets
import pandas as pd

#the classifcation function - used for classigying data
def centralize_data(arg1):
#import the MinMaxScalar from sklearn
    from sklearn.preprocessing import MinMaxScaler
#creating the object for the scalar
    scaling1 = MinMaxScaler()
#fitting the scalar function into my data
    p1_centeralized = scaling1.fit_transform(arg1)
#covnerting the scaled dataset back into the dataset variable type
    p1_normalization_dataframe = pd.DataFrame(p1_centeralized)
#readding collums back to the new dataframe type variable
    p1_normalization_dataframe.columns = ['TOTUSJH_median','TOTBSQ_median','TOTPOT_median','TOTUSJZ_median','ABSNJZH_median','TOTUSJH_mean',
                'TOTBSQ_mean', 'TOTPOT_mean','TOTUSJZ_mean','ABSNJZH_mean','TOTUSJH_min', 'TOTBSQ_min','TOTPOT_min',
                'TOTUSJZ_min', 'ABSNJZH_min', 'TOTUSJH_max', 'TOTBSQ_max', 'TOTPOT_max', 'TOTUSJZ_max', 'ABSNJZH_max',
                'TOTUSJH_std', 'TOTBSQ_std', 'TOTPOT_std', 'TOTUSJZ_std', 'ABSNJZH_std', 'TOTUSJH_skewness',
                'TOTBSQ_skewness', 'TOTPOT_skewness', 'TOTUSJZ_skewness', 'ABSNJZH_skewness', 'TOTUSJH_kurtosis',
                'TOTBSQ_kurtosis', 'TOTPOT_kurtosis','TOTUSJZ_kurtosis', 'ABSNJZH_kurtosis', 'TOTUSJH_last_value',
                'TOTBSQ_last_value', 'TOTPOT_last_value', 'TOTUSJZ_last_value', 'ABSNJZH_last_value', 'label']
    return p1_normalization_dataframe


#create the normalize data set
def normalize_data(arg1):
# import the StandardScalar from sklearn
    from sklearn.preprocessing import StandardScaler
# creating the object for the scalar
    scaling2 = StandardScaler()
# fitting the scalar function into my data
    p1_centralized = scaling2.fit_transform(arg1)
# covnerting the scaled dataset back into the dataset variable type
    p1_centralized_dataframe = pd.DataFrame(p1_centralized)
# readding collums back to the new dataframe type variable
    p1_centralized_dataframe.columns = ['TOTUSJH_median', 'TOTBSQ_median', 'TOTPOT_median', 'TOTUSJZ_median',
                                        'ABSNJZH_median', 'TOTUSJH_mean',
                                        'TOTBSQ_mean', 'TOTPOT_mean', 'TOTUSJZ_mean', 'ABSNJZH_mean', 'TOTUSJH_min',
                                        'TOTBSQ_min', 'TOTPOT_min',
                                        'TOTUSJZ_min', 'ABSNJZH_min', 'TOTUSJH_max', 'TOTBSQ_max', 'TOTPOT_max',
                                        'TOTUSJZ_max', 'ABSNJZH_max',
                                        'TOTUSJH_std', 'TOTBSQ_std', 'TOTPOT_std', 'TOTUSJZ_std', 'ABSNJZH_std',
                                        'TOTUSJH_skewness',
                                        'TOTBSQ_skewness', 'TOTPOT_skewness', 'TOTUSJZ_skewness', 'ABSNJZH_skewness',
                                        'TOTUSJH_kurtosis',
                                        'TOTBSQ_kurtosis', 'TOTPOT_kurtosis', 'TOTUSJZ_kurtosis', 'ABSNJZH_kurtosis',
                                        'TOTUSJH_last_value',
                                        'TOTBSQ_last_value', 'TOTPOT_last_value', 'TOTUSJZ_last_value',
                                        'ABSNJZH_last_value', 'label']
    return p1_centralized_dataframe