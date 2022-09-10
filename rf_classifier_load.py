import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import load
from pathlib import Path

# best to use with scikit-learn version 0.24.1, in order to avoid problems loading the model.

def readFile():
    fpath = sys.argv[-1]
    df = pd.read_csv(fpath)
    manInput = False

    check_ground_truth = input("Do you have experimental ln_Pe included in your dataset? (y/n)")

    if check_ground_truth == "y":
        # initial naming of columns as per our names in paper including experimental ln_papp.
        columns = ['title','ALOGP', 'PSA_w', 'T(N..O)', 'nHDon', 'T(N..N)' , 'Es_w', 'piPC02','ln_Pe']
    else:
        # initial naming of columns as per our names in paper.
        columns = ['title','ALOGP', 'PSA_w', 'T(N..O)', 'nHDon', 'T(N..N)' , 'Es_w', 'piPC02',]

    check = input("Do any parameters have different names as per readme file? (y/n)")
  
    #columns = ['title','ALOGP', 'T(N..O)','nHDon', 'T(N..N)', 'piPC10', 'piPC04', 'piPC02', 'MAXDN', 'PSA_w','Es_w']

    # prompt in case names of columns differ to names presented in paper.
    if check == "y":
        manInput = True
        print("Please press ENTER, when the name of the prompted field is correct, otherwise please enter the correct name and then press ENTER.\n")
    
    # prompt to adjust column names that differ from default names used in paper.
    if manInput == True:
        #print(columns)
        for index, column in enumerate(columns):
            field = input(column+":")
            if not field == "":
                columns[index] = field
    
    print("\nThe defined column names are set to the following:")
    print(columns)


    df = df[columns]

    Path('results/').mkdir(parents=True, exist_ok=True)
    df.to_csv("results/data_features_extracted.csv", index=False)

    return df, columns

def RandomForestPredictor(df, cols):
    #print(len(sys.argv))
    if len(sys.argv) > 2:
        fmodel = sys.argv[-2] # useful if user is using their own model.
    else:
        # to load our best performing model, as per paper.
        fmodel = 'data/trained_model_7features.joblib'

    df_cpy = df.copy()

    regr = load(fmodel)
    # if ln_Pe not included.
    if len(cols) == 8:
        df.drop(columns=cols[0], axis=1, inplace=True)
    # if ln_Pe is included.
    else:
        df.drop(columns=[cols[0],cols[7]], axis=1, inplace=True)
    testPred = regr.predict(df)

    
    with open('results/predictions.csv', 'w') as f:
        for item in testPred:
            f.write("%s\n" % item)

    results = pd.read_csv('results/predictions.csv', header=None, names=['ln_Pe_prediction'])
    output = pd.concat([df_cpy,results], axis=1)
    output.to_csv('results/predictions_data.csv', index=False)
    os.remove('results/predictions.csv')
    os.remove('results/data_features_extracted.csv')
    print('\nAll predictions are completed. Please access the results directory to view the predictions.')


if __name__ == "__main__":

    df, columns = readFile()
    RandomForestPredictor(df, columns)