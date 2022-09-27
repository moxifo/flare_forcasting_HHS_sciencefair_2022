import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
dataset = pd.read_csv(r"C:\Users\eugen\Downloads\p1_40sf.csv")
#removing last 3 columns
X = dataset.iloc[:, :-3].values
#converting the array into a dataset
Xdataframe = pd.DataFrame(X)
#manually adding back the header titles to the modified dataframe
Xdataframe.columns = ['TOTUSJH_median','TOTBSQ_median','TOTPOT_median','TOTUSJZ_median','ABSNJZH_median','TOTUSJH_mean',
                'TOTBSQ_mean', 'TOTPOT_mean','TOTUSJZ_mean','ABSNJZH_mean','TOTUSJH_min', 'TOTBSQ_min','TOTPOT_min',
                'TOTUSJZ_min', 'ABSNJZH_min', 'TOTUSJH_max', 'TOTBSQ_max', 'TOTPOT_max', 'TOTUSJZ_max', 'ABSNJZH_max',
                'TOTUSJH_std', 'TOTBSQ_std', 'TOTPOT_std', 'TOTUSJZ_std', 'ABSNJZH_std', 'TOTUSJH_skewness',
                'TOTBSQ_skewness', 'TOTPOT_skewness', 'TOTUSJZ_skewness', 'ABSNJZH_skewness', 'TOTUSJH_kurtosis',
                'TOTBSQ_kurtosis', 'TOTPOT_kurtosis','TOTUSJZ_kurtosis', 'ABSNJZH_kurtosis', 'TOTUSJH_last_value',
                'TOTBSQ_last_value', 'TOTPOT_last_value', 'TOTUSJZ_last_value', 'ABSNJZH_last_value', 'label']
#saving the new dataframe to computer
#Xdataframe.to_csv(r'C:/Users/eugen/Downloads/Xdataframe2.csv', index = False) # it has already been done Eugene
#creating the variables
names = ["nonflare","flare"]
xaxis_one = [0, 1]
yaxis_one = [72238, 1254]
#creating the bar plot via matplotlib
#plt.bar(xaxis_one, yaxis_one, width=0.5, )
#plt.xticks(xaxis_one, names)
#plt.title('amount of nonflare vs flare')
#plt.show()
#importing new dataframe into pandas
dataset2 = pd.read_csv(r"C:\Users\eugen\Downloads\Xdataframe2.csv")

#BOXPLOT!!!
#setting the style to whitegrid
#sns.set(style='whitegrid')
#setting the figure and axis
#fig, ax = plt.subplots(figsize=(8,6))
#setting up the boxplot(s)
#g = sns.boxplot(data=dataset2[['TOTUSJH_last_value', 'TOTBSQ_last_value', 'TOTPOT_last_value',
                               #'TOTUSJZ_last_value', 'ABSNJZH_last_value']], width=0.7)

#setting up titles and lables
#plt.title("distribution between all last value independent variables")
#plt.xlabel("last value variables", fontsize=14)
#plt.ylabel('values numeric(range is very big)', fontsize=14)

#x tick-lables for full variable name
#xvalues = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH'] #RED FLAG (THIS CODE MAY NOT BE NEEDED)

#set xvalues as xtick values
#plt.xticks(np.arange(5), xvalues)

#set y-axis values only whole numbers
#plt.yticks(np.arange(1,10))

#remove all borders except bottom
#sns.despine(top=True,
            #right=True,
            #left=True,
            #bottom=False)


#plt.tight_layout()

#plt.show()

#scatter plot

#dataset2.plot.scatter(x='TOTPOT_mean',
                     # y='label',
                     # )
#plt.show()![](../../Downloads/scatter plot TOTPOT mean vs label.png)
#![](../../Downloads/scatter polot TOTPOT median vs label.png)



#import the MinMaxScalar from sklearn
from sklearn.preprocessing import MinMaxScaler
#creating the object for the scalar
scaling1 = MinMaxScaler()
#fitting the scalar function into my data
p1_centeralized = scaling1.fit_transform(dataset2)
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
#converting the new pd.dataframe into a readable csv file
#p1_normalization_dataframe.to_csv(r'C:/Users/eugen/Downloads/p1_normalization_dataframe.csv', index = False)
#reading the csv file back into java via pandas
normalized_dataset = pd.read_csv(r"C:\Users\eugen\Downloads\p1_normalization_dataframe.csv")

#SETTING UP THE NORMALIZED BOXPLOT!!!

#setting the style to whitegrid
#sns.set(style='whitegrid')
#setting the figure and axis
#fig, ax = plt.subplots(figsize=(8,6))
#setting up the boxplot(s)
#loop = sns.boxplot(data=normalized_dataset[['TOTUSJH_median', 'TOTBSQ_median', 'TOTPOT_median',
                                        # 'TOTUSJZ_median', 'ABSNJZH_median']], width=0.7)

#setting up titles and lables
#plt.title("distribution between all median independent variables CENTRALIZED DATA SET")
#plt.xlabel("kurtosis variables CENTRALIZED", fontsize=14)
#plt.ylabel('values numeric(range is very big)', fontsize=14)

#x tick-lables for full variable name
#xvalues = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH']

#set xvalues as xtick values
#plt.xticks(np.arange(5), xvalues)

#set y-axis values only whole numbers
#plt.yticks(np.arange(1,10))

#remove all borders except bottom
#sns.despine(top=True,
           # right=True,
           # left=True,
           # bottom=False)


#plt.tight_layout()

#plt.show()

#import the StandardScalar from sklearn
from sklearn.preprocessing import StandardScaler
#creating the object for the scalar
scaling2 = StandardScaler()
#fitting the scalar function into my data
p1_centralized = scaling2.fit_transform(dataset2)
#covnerting the scaled dataset back into the dataset variable type
p1_centralized_dataframe = pd.DataFrame(p1_centralized)
#readding collums back to the new dataframe type variable
p1_centralized_dataframe.columns = ['TOTUSJH_median','TOTBSQ_median','TOTPOT_median','TOTUSJZ_median','ABSNJZH_median','TOTUSJH_mean',
                'TOTBSQ_mean', 'TOTPOT_mean','TOTUSJZ_mean','ABSNJZH_mean','TOTUSJH_min', 'TOTBSQ_min','TOTPOT_min',
                'TOTUSJZ_min', 'ABSNJZH_min', 'TOTUSJH_max', 'TOTBSQ_max', 'TOTPOT_max', 'TOTUSJZ_max', 'ABSNJZH_max',
                'TOTUSJH_std', 'TOTBSQ_std', 'TOTPOT_std', 'TOTUSJZ_std', 'ABSNJZH_std', 'TOTUSJH_skewness',
                'TOTBSQ_skewness', 'TOTPOT_skewness', 'TOTUSJZ_skewness', 'ABSNJZH_skewness', 'TOTUSJH_kurtosis',
                'TOTBSQ_kurtosis', 'TOTPOT_kurtosis','TOTUSJZ_kurtosis', 'ABSNJZH_kurtosis', 'TOTUSJH_last_value',
                'TOTBSQ_last_value', 'TOTPOT_last_value', 'TOTUSJZ_last_value', 'ABSNJZH_last_value', 'label']
#converting the new pd.dataframe into a readable csv file
p1_centralized_dataframe.to_csv(r'C:/Users/eugen/Downloads/p1_centralized_dataframe.csv', index = False)
#reading the csv file back into java via pandas
centralized_dataset = pd.read_csv(r"C:\Users\eugen\Downloads\p1_centralized_dataframe.csv")

#Creating the CENTRALIZED boxplot

#setting the style to whitegrid
sns.set(style='whitegrid')
#setting the figure and axis
fig, ax = plt.subplots(figsize=(8,6))
#setting up the boxplot(s)
loop1 = sns.boxplot(data=centralized_dataset[['TOTUSJH_last_value', 'TOTBSQ_last_value', 'TOTPOT_last_value',
                                              'TOTUSJZ_last_value', 'ABSNJZH_last_value']], width=0.7)

#setting up titles and lables
plt.title("distribution between all last_value independent variables CENTRALIZED DATA SET")
plt.xlabel("last_value variables CENTRALIZED", fontsize=14)
plt.ylabel('values numeric(range is very big)', fontsize=14)

#x tick-lables for full variable name
xvalues = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH']

#set xvalues as xtick values
plt.xticks(np.arange(5), xvalues)

#set y-axis values only whole numbers
plt.yticks(np.arange(0,125))

#remove all borders except bottom
sns.despine(top=True,
            right=True,
            left=True,
            bottom=False)

plt.tight_layout()

plt.show()

#change 1

from pyfunctions import dump_data








