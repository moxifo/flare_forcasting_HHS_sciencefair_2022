import pandas as pd


def dump_data(df: pd.DataFrame, path: str):
    """
    stores a given dataframe as a CSV file, with the given filename.
    
    :param df: the pandas dataframe to be stored as a CSV file
    :param path: the path where the data should be stored at
    
    :return:
    """
    # converting the new pd.dataframe into a readable csv file
    new_csv_file = df.to_csv(path, index=False)
    return new_csv_file


def load_data(arg1):
    """
    @TODO: Please complete this similar to the example above.
    :param arg1:
    :return:
    """
    # deciding which collumns to add and remove
    X = arg1.iloc[:, :-1].values
    # converting the array into a dataset
    Xdataframe = pd.DataFrame(X)
    return Xdataframe


# @Eugene, please read these comments and remove them afterwards.
# 1. Stick to one naming convention. E.g., instead of "newcsvfile" use
# "new_csv_file", and instead of "Xdataframe" use "x_dataframe". This
# convention is called Snake Case.
#
# 2. As you can see in this scrip, PyCharm is not underlining my
# code/comments. Make sure you follow all good practices that IDE
# suggests to avoid those underlined pieces. See the yellow bulb next to each
# underlined piece and follow
# the suggestions.
#
# 3. Whenever you need to pass a file path to a method, note that you CANNOT
# use your local machine's paths. For example, you cannot use
# 'C:/Users/eugen/Downloads/test111'. Instead, you should use a relative path
# so that it works on anyone's computer, who has this project. So,
# maybe something like this: '../data/output/test111.csv'.
