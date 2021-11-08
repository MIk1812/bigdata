import pandas as pd


def load_data():
    # Load data into dataframe
    return pd.read_csv('Data/heart.csv', delimiter=';')
