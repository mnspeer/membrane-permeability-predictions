import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import load
from pathlib import Path

# best to use with scikit-learn version 0.24.1, in order to avoid problems loading the model.

def readFile():
    fpath = sys.argv[-1]
    df = pd.read_csv(fpath)
    manInput = False
    check = input("Do any parameters have different names as per readme file? (y/n)")

    # initial naming of columns as per our names in paper.
    columns = ['title', 'MLOGP', 'T(N..O)','T(N..N)','piPC10', 'piPC04','piPC02', 'nHDon','PSA_w','Es_w', '|dEs|']

    # prompt in case names of columns differ to names presented in paper.
    if check == "y":
        manInput = True
        print("Please press ENTER, when the name of the prompted field is correct, otherwise please enter the correct name and then press ENTER.\n")
    
    # prompt to adjust column names that differ from default names used in paper.
    if manInput == True:
        print(columns)
        for index, column in enumerate(columns):
            field = input(column+":")
            if not field == "":
                columns[index] = field
    
    print("The defined column names are set to the following:")
    print(columns)

    df = df[columns]

    df.to_csv("data_features_extracted.csv")

    return df, columns

def RandomForestPredictor(df, cols):
    print(len(sys.argv))
    if len(sys.argv) > 2:
        fmodel = sys.argv[-2] # useful if user is using their own model.
    else:
        # to load our best performing model, as per paper.
        fmodel = 'Set1_Set2_Set3_trained_model_3D_nHDon_new.joblib'
    regr = load(fmodel)
    title = df[cols[0]]
    df.drop(columns=cols[0], axis=1, inplace=True)
    testPred = regr.predict(df)

    Path('results/').mkdir(parents=True, exist_ok=True)
    with open('results/predictions.csv', 'w') as f:
        for item in testPred:
            f.write("%s\n" % item)

    results = pd.read_csv('results/predictions.csv', header=None, names=['LN_PERM_prediction'])
    print(results)
    output = pd.concat([title,df,results], axis=1)
    output.to_csv('results/predictions_data.csv')

if __name__ == "__main__":

    df, columns = readFile()
    RandomForestPredictor(df, columns)