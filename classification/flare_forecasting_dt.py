import numpy as np


def train_test_split(dataframe, a, b):
    """
    splits the data into a train and test split

    :param dataframe: the pandas data frame in which you want to split
    :param a: the integer in which you want the last row to be counted
    :return: the split data set
    """
    ### splitting the data into 2 partitions (train and test)
    X = np.array(dataframe.iloc[:a, 1:-1].values)
    X = X.reshape(len(X), 39)
    y = dataframe.iloc[:b, -1].values
    y = y.reshape(len(y), 1)
    return X, y


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
    f1_score  = sklearn.metrics.f1_score(y, Y)
    return f1_score
