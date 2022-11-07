import numpy as npd
from matplotlib import pyplot as plt
import seaborn as sns
from utils import normalizer as nm
from utils import data_io
from classification import flare_forecasting_dt as ffdt
from visualisation import visualize_data_stats as vds
import pandas as pd
from sklearn.model_selection import train_test_split

### Import p1 and p2 of SWAN-SF dataset
p1 = vds.import_csv(r"C:\Users\eugen\Downloads\p1_40sf (1).csv")
p2 = vds.import_csv(r"C:\Users\eugen\Downloads\p2_40sf (1).csv")

### Removing the last 2 column of the data
p1_modified = data_io.cut_data(p1)
p2_modified = data_io.cut_data(p2)

### Finding Global extrema
p1_modified.max().max()
p2_modified.min().min()

### Normalizing the data via MinMax (Global extrema {min:-5.723171611452671, max:2.820688751045556e+25} )
p1_normalized = nm.normalize_data(p1_modified, 0, 1)
p2_normalized = nm.normalize_data(p2_modified, 0, 1)

### store the data into a new csv file
# data_io.dump_data(p1_normalized,r"C:\Users\eugen\Downloads\p1_minmaxnormalized.csv")
#data_io.dump_data(p2_normalized, r"C:\Users\eugen\Downloads\p2_minmaxnormalized.csv")

### undersample data (10 times)
df1_undersampled_01 = ffdt.undersample(p1_normalized, 72238, 0.0173592846)
df1_undersampled_02 = ffdt.undersample(p1_normalized, 72238, 0.0173592846)
df1_undersampled_03 = ffdt.undersample(p1_normalized, 72238, 0.0173592846)
df1_undersampled_04 = ffdt.undersample(p1_normalized, 72238, 0.0173592846)
df1_undersampled_05 = ffdt.undersample(p1_normalized, 72238, 0.0173592846)
df1_undersampled_06 = ffdt.undersample(p1_normalized, 72238, 0.0173592846)
df1_undersampled_07 = ffdt.undersample(p1_normalized, 72238, 0.0173592846)
df1_undersampled_08 = ffdt.undersample(p1_normalized, 72238, 0.0173592846)
df1_undersampled_09 = ffdt.undersample(p1_normalized, 72238, 0.0173592846)
df1_undersampled_10 = ffdt.undersample(p1_normalized, 72238, 0.0173592846)

df2_undersampled_01 = ffdt.undersample(p2_normalized, 87156, 0.0158316113)

### Test train split (70/30)
X1_train, X1_test, y1_train, y1_test = ffdt.modified_sklearn_testrainsplit(df1_undersampled_01)
X2_train, X2_test, y2_train, y2_test = ffdt.modified_sklearn_testrainsplit(df1_undersampled_02)
X3_train, X3_test, y3_train, y3_test = ffdt.modified_sklearn_testrainsplit(df1_undersampled_03)
X4_train, X4_test, y4_train, y4_test = ffdt.modified_sklearn_testrainsplit(df1_undersampled_04)
X5_train, X5_test, y5_train, y5_test = ffdt.modified_sklearn_testrainsplit(df1_undersampled_05)
X6_train, X6_test, y6_train, y6_test = ffdt.modified_sklearn_testrainsplit(df1_undersampled_06)
X7_train, X7_test, y7_train, y7_test = ffdt.modified_sklearn_testrainsplit(df1_undersampled_07)
X8_train, X8_test, y8_train, y8_test = ffdt.modified_sklearn_testrainsplit(df1_undersampled_01)
X9_train, X9_test, y9_train, y9_test = ffdt.modified_sklearn_testrainsplit(df1_undersampled_09)
X10_train, X10_test, y10_train, y10_test = ffdt.modified_sklearn_testrainsplit(df1_undersampled_10)

### removing the input and output values of p2 dataset
Xn, yn = ffdt.train_test_split(df2_undersampled_01, None, None)

### Train the model
regressor_01 = ffdt.train_decision_tree(X1_train, y1_train)
regressor_02 = ffdt.train_decision_tree(X2_train, y2_train)
regressor_03 = ffdt.train_decision_tree(X3_train, y3_train)
regressor_04 = ffdt.train_decision_tree(X4_train, y4_train)
regressor_05 = ffdt.train_decision_tree(X5_train, y5_train)
regressor_06 = ffdt.train_decision_tree(X6_train, y6_train)
regressor_07 = ffdt.train_decision_tree(X7_train, y7_train)
regressor_08 = ffdt.train_decision_tree(X8_train, y8_train)
regressor_09 = ffdt.train_decision_tree(X9_train, y9_train)
regressor_10 = ffdt.train_decision_tree(X10_train, y10_train)

### generate predicted models
pred_01 = regressor_01.predict(Xn)
pred_02 = regressor_02.predict(Xn)
pred_03 = regressor_03.predict(Xn)
pred_04 = regressor_04.predict(Xn)
pred_05 = regressor_05.predict(Xn)
pred_06 = regressor_06.predict(Xn)
pred_07 = regressor_07.predict(Xn)
pred_08 = regressor_08.predict(Xn)
pred_09 = regressor_09.predict(Xn)
pred_10 = regressor_10.predict(Xn)

### get confusion metrics
F1_score_01 = ffdt.return_f1_score(yn, pred_01)
F1_score_02 = ffdt.return_f1_score(yn, pred_02)
F1_score_03 = ffdt.return_f1_score(yn, pred_03)
F1_score_04 = ffdt.return_f1_score(yn, pred_04)
F1_score_05 = ffdt.return_f1_score(yn, pred_05)
F1_score_06 = ffdt.return_f1_score(yn, pred_06)
F1_score_07 = ffdt.return_f1_score(yn, pred_07)
F1_score_08 = ffdt.return_f1_score(yn, pred_08)
F1_score_09 = ffdt.return_f1_score(yn, pred_09)
F1_score_10 = ffdt.return_f1_score(yn, pred_10)

Final_F1 = (F1_score_01 + F1_score_02 + F1_score_03 + F1_score_04 + F1_score_05 + F1_score_06 +
            F1_score_07 + F1_score_08 + F1_score_09 + F1_score_10) / 10
