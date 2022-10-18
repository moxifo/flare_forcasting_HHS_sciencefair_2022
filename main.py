import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from visualisation import visualize_data_stats as vds
from utils import data_io
from utils import normalizer as nm


dataset = vds.import_csv(r"C:\Users\eugen\Downloads\p1_40sf.csv")
X = data_io.cut_data(dataset)

# saving the new dataframe to computer
# Xdataframe.to_csv(r'C:/Users/eugen/Downloads/Xdataframe2.csv', index = False) # it has already been done Eugene

# creating the variables
plot_1 = vds.data_feature_distribution_chart(72238, 1254)

# importing new dataframe into pandas
dataset2 = pd.read_csv(r"C:\Users\eugen\Downloads\Xdataframe2.csv")

# BOXPLOT!!!

boxplot_1 = vds.data_boxplot(dataset2, 'TOTUSJH_last_value', 'TOTBSQ_last_value', 'TOTPOT_last_value',
                             'TOTUSJZ_last_value', 'ABSNJZH_last_value')

# import the MinMaxScalar from sklearn

nm.normalize_data(dataset2)

normalized_dataset = vds.import_csv(r"C:\Users\eugen\Downloads\p1_normalization_dataframe.csv")

# SETTING UP THE NORMALIZED BOXPLOT!!!

boxplot_2 = vds.data_boxplot(normalized_dataset, 'TOTUSJH_median', 'TOTBSQ_median', 'TOTPOT_median',
                             'TOTUSJZ_median', 'ABSNJZH_median')


p1_centralized_dataframe = nm.centralize_data(dataset2)

# creating the object for the scalar

centralized_dataset = vds.import_csv(r"C:\Users\eugen\Downloads\p1_centralized_dataframe.csv")

# Creating the CENTRALIZED boxplot

boxplot_3 = vds.data_boxplot(centralized_dataset, 'TOTUSJH_last_value', 'TOTBSQ_last_value', 'TOTPOT_last_value',
                             'TOTUSJZ_last_value', 'ABSNJZH_last_value')



