import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np
from joblib import dump
import sys


def readFile():
    fpath = sys.argv[-1]
    df = pd.read_csv(fpath)
    #df = pd.read_csv('Membrane_Permeability_Study/Set1_Set2_Set3/Set1_Set2_Set3_SMILE_3D.csv')
    labels = df['LN_PERM'].to_list()
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.drop(columns="LN_PERM", axis=1, inplace=True)

    df = df[['ALOGP', 'T(N..O)','nHDon', 'T(N..N)', 'piPC10', 'piPC04', 'piPC02', 'MAXDN', 'PSA_w','Es_w']]

    return df, labels

def RandomForestTraining(df, labels):
    if len(sys.argv) > 2:
        modelname = sys.argv[-2] # useful if user defines name of .joblib file
    else:
        # to use pre-defined "trained_model.joblib" as name.
        modelname = 'trained_model'
    regr = RandomForestRegressor(max_depth=17, random_state=0)
    regr.fit(df, labels)
    modelname = modelname + '.joblib'
    dump(regr, modelname)

if __name__ == "__main__":

    df, labels = readFile()
    RandomForestTraining(df, labels)