import pandas as pd


def load_data(one_out_of_k: bool):
    out = pd.DataFrame()

    # Load data into dataframe
    data = pd.read_csv("Data/heart.csv", delimiter=';')

    # Add columns that do not need feature transformation
    out["Age"] = data["Age"]
    out["RestingBP"] = data["RestingBP"]
    out["Cholesterol"] = data["Cholesterol"]
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

    out["FastingBS"] = data["FastingBS"]

    return out


def remove_outliers():
    X = pd.read_csv("Data/feature_transform.csv", delimiter=',')
    Xk = pd.read_csv("Data/one_out_of_k.csv", delimiter=',')

    X.drop(X[X['RestingBP'] == 0].index, inplace = True)
    X.drop(X[X['Cholesterol'] == 0].index, inplace=True)
    X.to_csv("feature_transform_outliers_removed.csv", index=False)

    Xk.drop(Xk[Xk['RestingBP'] == 0].index, inplace=True)
    Xk.drop(Xk[Xk['Cholesterol'] == 0].index, inplace=True)
    Xk.to_csv("one_out_of_k_outliers_removed.csv", index=False)


load_data(False).to_csv('feature_transform.csv', index=False)
load_data(True).to_csv('one_out_of_k.csv', index=False)
remove_outliers()