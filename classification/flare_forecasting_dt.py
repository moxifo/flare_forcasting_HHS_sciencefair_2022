import numpy as np
import pandas
print('slay')

def train_test_split(dataframe, a, b):
    """
    splits the data into a train and test split

    :param dataframe: the pandas data frame in which you want to split
    :param a: the integer in which you want the last row to be counted
    :return: the split data set
    """
    ### splitting the data into 2 partitions (train and test)
    X = np.array(dataframe.iloc[:a, :-1].values)
    X1 = X.reshape(len(X), 40)
    y = dataframe.iloc[:b, -1].values
    y1 = y.reshape(len(y), 1)
    return X1, y1


def modified_sklearn_testrainsplit(df):
    """

    :param df: the pandas data frane yoy want to split
    :return: the 4 dataframes
    """
    from classification import flare_forecasting_dt as ffdt
    from sklearn.model_selection import train_test_split
    X, y = ffdt.train_test_split(df, None, None)
    ashdajsd = df.iloc[:, -1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=ashdajsd)
    return X_train, X_test, y_train, y_test


def train_decision_tree(X, y):
    """
    uses decision tree regression to train a model via an inputted dataset

    :param X: The given parameters (features) of the dataset
    :param y: The outcome
    :return: The trained ML model
    """
    import sklearn.tree
    regressor = sklearn.tree.DecisionTreeRegressor(random_state=0)
    regressor.fit(X, np.ravel(y, order='C'))
    return regressor


def create_confusion_matrix(y, Y):
    """

    :param y: the real output values
    :param Y: the predicted output values
    :return: the confusion matrix
    """
    import sklearn.metrics
    cm = sklearn.metrics.confusion_matrix(y, Y)
    return cm


def return_accuracy_score(y, Y):
    """
    returns the accuracy of an ML model via the actual and predicted output values

    :param y: real output values
    :param Y: predicted output values
    :return: the accuracy score
    """
    import sklearn.metrics
    a_score = sklearn.metrics.accuracy_score(y, Y)
    return a_score


def return_precision_score(y, Y):
    """
    returns the precision score of an ML model via the actual and predicted output values
    :param y: real output values
    :param Y: predicted output values
    :return: the precision score
    """
    import sklearn.metrics
    p_score = sklearn.metrics.precision_score(y, Y)
    return p_score


def return_recall_score(y, Y):
    """
    returns the recall score of an ML model via the actual and predicted output values
    :param y: the real output values
    :param Y: predicted output values
    :return: the recall score
    """
    import sklearn.metrics
    r_score = sklearn.metrics.recall_score(y, Y)
    return r_score


def return_f1_score(y, Y):
    """
    returns the recall score of an ML model via the actual and predicted output values
    :param y: the real output values
    :param Y: predicted output values
    :return: f1 score
    """
    import sklearn.metrics
    f1_score = sklearn.metrics.f1_score(y, Y)
    return f1_score


def feature_split(df, v):
    """
    splits a pandas dataframe by rows by the select integer the user wants to separate the dataframe by.
    :param df: the pandas dataframe to be separated
    :param v: the integer that you want the row to be separated by
    :return: two dataframes separated by class
    """
    df1 = df.iloc[:v, :]
    # df1 = df1.reshape(len(df1), 41)
    df2 = df.iloc[v:, :]
    # df2 = df2.reshape(len(df2), 41)
    return df1, df2


def undersample(df, v, b: int):
    """
    undersample the data to manage class imbalance

    :param df: the pandas dataframe to be under-sampled
    :param v: the integer that represents the row where to positive and negative class are separated
    :param b: the decimal that represents the ratio of negative to positive class
    :return: the under-sampled dataframe
    """
    import pandas as pd
    from classification import flare_forecasting_dt as ffdt
    df1, df2 = ffdt.feature_split(df, v)
    df1_undersampled_01 = df1.sample(frac=b)
    frames = [df1_undersampled_01, df2]
    result = pd.concat(frames)
    return result


def log_normalization(df: pandas.DataFrame):
    """
    runs a log normalization on a pandas dataframe.
    All values must be integers
    The method will automatically calculate the absolute value of the dataframe.

    :param df: the pandas dataframe that log normalization will be run on
    :return: the normalized dataset
    """
    from sklearn.preprocessing import FunctionTransformer
    import pandas as pd
    import numpy as np
    from classification import flare_forecasting_dt as ffdt
    transformer = FunctionTransformer(np.log1p)
    p1_abs = df.abs()
    X, y = ffdt.train_test_split(p1_abs, None, None)
    p1_log = transformer.transform(X)
    result = pd.concat([pd.DataFrame(p1_log),pd.DataFrame(y)], axis=1)
    return result
