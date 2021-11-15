import pandas as pd
from LoadData import load_data

X = pd.read_csv("Data/feature_transform.csv", delimiter=',')
Xk = pd.read_csv("Data/one_out_of_k.csv", delimiter=',')
