### Import the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pyfunctions import normalize_data
from pyfunctions import load_data
from pyfunctions import dump_data

### Importing the csv file into python (normalized data)
df1 = pd.read_csv(r"C:\Users\eugen\Downloads\p1_normalization_dataframe.csv")
nd = pd.read_csv(r'C:\Users\eugen\Downloads\p2_40sf.csv')

### removing the last two collumns of partition 2 of data
X1 = nd.iloc[:, :-3].values
X1dataframe = pd.DataFrame(X1)
X1dataframe.columns = ['TOTUSJH_median','TOTBSQ_median','TOTPOT_median','TOTUSJZ_median','ABSNJZH_median','TOTUSJH_mean',
                      'TOTBSQ_mean', 'TOTPOT_mean','TOTUSJZ_mean','ABSNJZH_mean','TOTUSJH_min', 'TOTBSQ_min','TOTPOT_min',
                      'TOTUSJZ_min', 'ABSNJZH_min', 'TOTUSJH_max', 'TOTBSQ_max', 'TOTPOT_max', 'TOTUSJZ_max', 'ABSNJZH_max',
                      'TOTUSJH_std', 'TOTBSQ_std', 'TOTPOT_std', 'TOTUSJZ_std', 'ABSNJZH_std', 'TOTUSJH_skewness',
                      'TOTBSQ_skewness', 'TOTPOT_skewness', 'TOTUSJZ_skewness', 'ABSNJZH_skewness', 'TOTUSJH_kurtosis',
                      'TOTBSQ_kurtosis', 'TOTPOT_kurtosis','TOTUSJZ_kurtosis', 'ABSNJZH_kurtosis', 'TOTUSJH_last_value',
                      'TOTBSQ_last_value', 'TOTPOT_last_value', 'TOTUSJZ_last_value', 'ABSNJZH_last_value', 'label']

### normalize the data
X1_normalized = normalize_data(X1dataframe)
#dump_data(X1_normalized)

### splitting the data into 2 partitions (train and test)
X = np.array(df1.iloc[:,1:-1].values)
X = X.reshape(len(X),39)
y = df1.iloc[:,-1].values
y = y.reshape(len(y),1)

### splitting the p2 dataset into two partitions
Xn = np.array(X1_normalized.iloc[:-15065,1:-1].values)
Xn = Xn.reshape(len(Xn), 39)
yn = X1_normalized.iloc[:-15065,-1].values
yn = yn.reshape(len(yn), 1)

### training the model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,np.ravel(y, order='C'))

asd = regressor.predict(Xn)
#print(np.concatenate((y.reshape(len(y),1), asd.reshape(len(asd),1)),1))

print(X)