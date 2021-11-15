import pandas as pd
import matplotlib.pyplot as plt
from LoadData import load_data

X = pd.read_csv("Data/feature_transform.csv", delimiter=',')
Xk = pd.read_csv("Data/one_out_of_k.csv", delimiter=',')

def makeBoxPlot(dataframe):
    dataframe_1 = dataframe.loc[:,'Age':'Oldpeak'].copy()
    dataframe_2 = dataframe.loc[:,'HeartDisease':'FastingBS'].copy()
    dataframe_1.boxplot(figsize=(5,5))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    dataframe_2.boxplot(figsize=(5,5))
    #dataframe.boxplot(column=['Age'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

makeBoxPlot(data)