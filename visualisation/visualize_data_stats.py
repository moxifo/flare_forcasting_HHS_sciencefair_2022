# This script is a good place for your visualization methods.
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd


def import_csv(path: str):
    """
    Imports a csv file into a pd.dataframe

    :param path: the csv file that is going to be read by pandas

    :return:the read csv file
    """
    dataset = pd.read_csv(path)
    return dataset


def data_feature_distribution_chart(a: int, b: int):
    """
    Creates a boxplot of the occurences in which a solar flare did or did not appear.

    :param a: occurrences were solar flares did not appear
    :param b: occurrences were solar flares did appear
    :return: finished boxplot

    """
    names = ["nonflare", "flare"]
    xaxis_one = [0, 1]
    yaxis_one = [a, b]
    # creating the bar plot via matplotlib
    plt.bar(xaxis_one, yaxis_one, width=0.5, )
    plt.xticks(xaxis_one, names)
    plt.title('amount of nonflare vs flare')
    plt.show()


def data_boxplot(arg1,a: str, b: str, c: str, d: str, e: str):
    """

    :param arg1: the pandas dataframe in which the boxplot will be created from
    :param a: the value of TOTUSJH you want to include
    :param b: the value of TOTBSQ you want to include
    :param c: the value of TOTPOT you want to include
    :param d: the value of TOTUSJZ you want to include
    :param e: the value of ABSNJZH you want to include
    :return: finished boxplot
    """
    # setting the style to whitegrid
    sns.set(style='whitegrid')
    # setting the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    # setting up the boxplot(s)
    g = sns.boxplot(data=arg1[[a, b, c,
                               d, e]], width=0.7)

    # setting up titles and labels
    plt.title("distribution between all last value independent variables")
    plt.xlabel("last value variables", fontsize=14)
    plt.ylabel('values numeric(range is very big)', fontsize=14)

    # x tick-lables for full variable name
    xvalues = ['TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH']
    # set xvalues as xtick values
    plt.xticks(np.arange(5), xvalues)

    # set y-axis values only whole numbers
    plt.yticks(np.arange(1, 10))

    # remove all borders except bottom
    sns.despine(top=True,
                right=True,
                left=True,
                bottom=False)

    plt.tight_layout()

    plt.show()
