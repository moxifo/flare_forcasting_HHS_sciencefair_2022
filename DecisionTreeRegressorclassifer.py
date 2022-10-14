# TODO: Whatever is happening in this script should also be broken down into
#  several methods. Do NOT let snippets of code wander around. So, please
#  redo this script (learning from those scripts that I re-did for you). The
#  parts where the actual classification takes place (training + testing)
#  should go to "classification > flare_forecasting_dt.py".

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
Xn = np.array(Xn.reshape(len(Xn), 39))
yn = X1_normalized.iloc[:-15065,-1].values
yn = yn.reshape(len(yn), 1)

### training the model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,np.ravel(y, order='C'))

asd = regressor.predict(Xn)
asd1 = np.concatenate((y.reshape(len(y),1), asd.reshape(len(asd),1)),1)

### getting the metrics from confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,asd)
print(cm)

from sklearn.metrics import accuracy_score
Ascore = accuracy_score(y,asd)
print(Ascore)
print(asd)

from sklearn.metrics import precision_score
Pscore = precision_score(y,asd)
print(Pscore)

from sklearn.metrics import recall_score
Rscore = recall_score(y,asd)
print(Rscore)

from sklearn.metrics import f1_score
F1score = f1_score(y,asd)
print(F1score)

### NOTES !!!
#precision score is for all the samples of the positive class out of all predicted samples
#recall score is a 'harmonic' mean of precision
# 'harmonic mean of the precision and recall scores obtained from the positve class

#accuarcy was almost 94%, I dont know if I did anything wrong but that does not seem right
#additional when I used the parition 2 of the data, I removed some of the segments so they would have the
#same amount of rows