import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    # Load data into dataframe
    return pd.read_csv('Data/heart.csv', delimiter=';')

def makeBoxPlot(dataframe):
    dataframe.boxplot()
    plt.show()

makeBoxPlot(load_data())

