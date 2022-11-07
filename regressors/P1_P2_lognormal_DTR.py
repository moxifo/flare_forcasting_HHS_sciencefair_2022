import numpy as np
import numpy as npd
from matplotlib import pyplot as plt
import seaborn as sns
from utils import normalizer as nm
from utils import data_io
from classification import flare_forecasting_dt as ffdt
from visualisation import visualize_data_stats as vds
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

### Import p1 and p2 of SWAN-SF dataset
p1 = vds.import_csv(r"C:\Users\eugen\Downloads\p1_40sf (1).csv")
p2 = vds.import_csv(r"C:\Users\eugen\Downloads\p2_40sf (1).csv")

### Removing the last 2 column of the data
p1_modified = data_io.cut_data(p1)
p2_modified = data_io.cut_data(p2)

### Normalizing the data via log
p1_log = ffdt.log_normalization(p1_modified)
p2_log = ffdt.log_normalization(p2_modified)

### undersample data (10 times)
df1_undersampled_01 = ffdt.undersample(p1_log, 72238, 0.0173592846)
df1_undersampled_02 = ffdt.undersample(p1_log, 72238, 0.0173592846)
df1_undersampled_03 = ffdt.undersample(p1_log, 72238, 0.0173592846)
df1_undersampled_04 = ffdt.undersample(p1_log, 72238, 0.0173592846)
df1_undersampled_05 = ffdt.undersample(p1_log, 72238, 0.0173592846)
df1_undersampled_06 = ffdt.undersample(p1_log, 72238, 0.0173592846)
df1_undersampled_07 = ffdt.undersample(p1_log, 72238, 0.0173592846)
df1_undersampled_08 = ffdt.undersample(p1_log, 72238, 0.0173592846)
df1_undersampled_09 = ffdt.undersample(p1_log, 72238, 0.0173592846)
df1_undersampled_10 = ffdt.undersample(p1_log, 72238, 0.0173592846)

df2_undersampled_01 = ffdt.undersample(p2_log, 87156, 0.0158316113)

### Test train split
X_01, y_01 = ffdt.train_test_split(df1_undersampled_01, None, None)
X_02, y_02 = ffdt.train_test_split(df1_undersampled_01, None, None)
X_03, y_03 = ffdt.train_test_split(df1_undersampled_01, None, None)
X_04, y_04 = ffdt.train_test_split(df1_undersampled_01, None, None)
X_05, y_05 = ffdt.train_test_split(df1_undersampled_01, None, None)
X_06, y_06 = ffdt.train_test_split(df1_undersampled_01, None, None)
X_07, y_07 = ffdt.train_test_split(df1_undersampled_01, None, None)
X_08, y_08 = ffdt.train_test_split(df1_undersampled_01, None, None)
X_09, y_09 = ffdt.train_test_split(df1_undersampled_01, None, None)
X_10, y_10 = ffdt.train_test_split(df1_undersampled_01, None, None)

Xn, yn = ffdt.train_test_split(df2_undersampled_01, None, None)

### Train the model
regressor_01 = ffdt.train_decision_tree(X_01, y_01)
regressor_02 = ffdt.train_decision_tree(X_02, y_02)
regressor_03 = ffdt.train_decision_tree(X_03, y_03)
regressor_04 = ffdt.train_decision_tree(X_04, y_04)
regressor_05 = ffdt.train_decision_tree(X_05, y_05)
regressor_06 = ffdt.train_decision_tree(X_06, y_06)
regressor_07 = ffdt.train_decision_tree(X_07, y_07)
regressor_08 = ffdt.train_decision_tree(X_08, y_08)
regressor_09 = ffdt.train_decision_tree(X_09, y_09)
regressor_10 = ffdt.train_decision_tree(X_10, y_10)

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

print('hoi')