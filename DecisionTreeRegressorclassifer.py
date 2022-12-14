### Import the libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from utils import normalizer as nm
from utils import data_io
from classification import flare_forecasting_dt as ffdt
from visualisation import visualize_data_stats as vds

### Importing the csv file into python (normalized data)
df1 = vds.import_csv(r"C:\Users\eugen\Downloads\p1_normalization_dataframe.csv")
nd = vds.import_csv(r'C:\Users\eugen\Downloads\p2_40sf.csv')

### removing the last two collumns of partition 2 of data
X1 = data_io.cut_data(nd)

### normalize the data
X1_normalized = nm.normalize_data(X1, 0, 1)


### splitting the data into 2 partitions (train and test)
X, y = ffdt.train_test_split(df1, None, None)

### splitting the p2 dataset into two partitions
Xn, yn = ffdt.train_test_split(X1_normalized, None, None)

### training the model
regressor = ffdt.train_decision_tree(X, y)
asd = regressor.predict(Xn)

### getting the metrics from confusion matrix
cm = ffdt.create_confusion_matrix(yn, asd)

A_score = ffdt.return_accuracy_score(yn, asd)

P_score = ffdt.return_precision_score(yn, asd)

R_score = ffdt.return_recall_score(yn, asd)

F1_score = ffdt.return_f1_score(yn, asd)

print(cm, A_score, P_score, R_score, F1_score)

### NOTES !!!
# precision score is for all the samples of the positive class out of all predicted samples
# recall score is a 'harmonic' mean of precision
# 'harmonic mean of the precision and recall scores obtained from the positve class

# accuarcy was almost 94%, I dont know if I did anything wrong but that does not seem right
# additional when I used the parition 2 of the data, I removed some of the segments so they would have the
# same amount of rows
