import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def load_data(one_out_of_k: bool):
    out = pd.DataFrame()

    # Load data into dataframe
    data = pd.read_csv("Data/heart.csv", delimiter=';')

    # Add columns that do not need feature transformation
    out["Age"] = data["Age"]
    out["RestingBP"] = data["RestingBP"]
    out["Cholesterol"] = data["Cholesterol"]
    out["FastingBS"] = data["FastingBS"]
    out["MaxHR"] = data["MaxHR"]
    out["Oldpeak"] = data["Oldpeak"]
    out["HeartDisease"] = data["HeartDisease"]

    # Apply feature transformations
    if one_out_of_k:
        # Feature Transform Sex column to One-out-of-K coding
        out["Male"] = data["Sex"].eq("M").astype(int)
        out["Female"] = data["Sex"].eq("F").astype(int)

        # Feature Transform ChestPainType to One-out-of-K coding
        out["ATA"] = data["ChestPainType"].eq("ATA").astype(int)
        out["NAP"] = data["ChestPainType"].eq("NAP").astype(int)
        out["ASY"] = data["ChestPainType"].eq("ASY").astype(int)
        out["TA"] = data["ChestPainType"].eq("TA").astype(int)

        # Feature Transform RestingECG to One-out-of-K coding
        out["ECG_normal"] = data["RestingECG"].eq("Normal").astype(int)
        out["ECG_ST"] = data["RestingECG"].eq("ST").astype(int)
        out["ECG_LVH"] = data["RestingECG"].eq("LVH").astype(int)

        # Feature Transform ExcerciseAngina to One-out-of-K coding
        out["Exercise_Angina"] = data["ExerciseAngina"].eq("Y").astype(int)

        # Feature Transform ST_Slope to One-out-of-K coding
        out["ST_Slop_Up"] = data["ST_Slope"].eq("Up").astype(int)
        out["ST_Slop_Flat"] = data["ST_Slope"].eq("Flat").astype(int)
        out["ST_Slop_Down"] = data["ST_Slope"].eq("Down").astype(int)

    else:
        # Feature transform Sex (female = 0, male = 1)
        out["Sex"] = data["Sex"].eq("M").astype(int)

        # Feature transform ChestPainType
        out["ChestPainType"] = data["ChestPainType"].replace("ATA", 0)
        out["ChestPainType"] = out["ChestPainType"].replace("NAP", 1)
        out["ChestPainType"] = out["ChestPainType"].replace("ASY", 2)
        out["ChestPainType"] = out["ChestPainType"].replace("TA", 3)

        # Feature transform RestingECG
        out["RestingECG"] = data["RestingECG"].replace("Normal", 0)
        out["RestingECG"] = out["RestingECG"].replace("ST", 1)
        out["RestingECG"] = out["RestingECG"].replace("LVH", 2)

        # Feature transform ExcerciseAngina (no = 0, yes = 1)
        out["Exercise_Angina"] = data["ExerciseAngina"].eq("Y").astype(int)

        # Feature transform ST_Slope
        out["ST_Slope"] = data["ST_Slope"].replace("Up", 0)
        out["ST_Slope"] = out["ST_Slope"].replace("Flat", 1)
        out["ST_Slope"] = out["ST_Slope"].replace("Down", 2)

    return out

data = load_data(False)
print(data)

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

