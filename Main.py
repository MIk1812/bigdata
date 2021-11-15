import pandas as pd
from Preprocessing import load_data

X = pd.read_csv("Data/feature_transform.csv", delimiter=',')
X_no_outliers = pd.read_csv("Data/feature_transform_outliers_removed.csv", delimiter=',')
Xk = pd.read_csv("Data/one_out_of_k.csv", delimiter=',')
Xk_no_outliers = pd.read_csv("Data/one_out_of_k_outliers_removed.csv", delimiter=',')


