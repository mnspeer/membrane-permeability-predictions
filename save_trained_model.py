import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np
from joblib import dump


def readFile():
    # Set1_Set2_Set3_SMILE_noCorrel #Set1_Set2_Set3_SMILE_3D_noCorrel
    df = pd.read_csv('Membrane_Permeability_Study/Set1_Set2_Set3/Set1_Set2_Set3_SMILE_3D.csv')
    labels = df['LN_PERM'].to_list()
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.drop(columns="LN_PERM", axis=1, inplace=True)
    #df.drop(columns="MLOGP", axis=1, inplace=True)
    #df.drop(columns=["Es_o",'Es_w','dEs','PSAo','abs_dEs','scaffold'], axis=1, inplace=True)
    # , 'MSD', 'UNIP'
    #df = df[['ALOGP', 'T(N..O)','TPSA(Tot)', 'T(N..N)', 'piPC10', 'piPC04', 'piPC02', 'MAXDN']]

    #df = df[['ALOGP','T(N..O)', 'TPSA(Tot)', 'N-072', 'T(N..N)', 'H-052' ,'H-050', 'piPC10', 'piPC04', 'piPC02', 'MAXDN', 'Wap', 'ww',
    #'TI2', 'BLI', 'DELS', 'DECC', 'H-047', 'Ms', 'C-025', 'PW5', 'SPI', 'TIE', 'PW3', 'nRCONHR', 'AECC', 'Rww', 'PW4',
    #'D/Dr06', 'D/Dr05']]

    #df = df[['MLOGP','ALOGP', 'PSA_w','Es_w','TPSA(Tot)','T(N..O)','N-072','NumDon','nHDon','T(N..N)']]

    #df = df[['ALOGP', 'T(N..O)','|dEs|', 'T(N..N)', 'piPC10', 'piPC04', 'piPC02', 'MAXDN', 'PSA_w','Es_w']]
    # 'TPSA(Tot)'

    df = df[['ALOGP', 'T(N..O)','T(N..N)','piPC10', 'piPC04','piPC02', 'nHDon','PSA_w','Es_w', '|dEs|']]

    return df, labels

def RandomForestTraining(df, labels):
    regr = RandomForestRegressor(max_depth=17, random_state=0)
    regr.fit(df, labels)
    dump(regr, 'Set1_Set2_Set3_trained_model_3D_v4_new.joblib')

if __name__ == "__main__":

    df, labels = readFile()
    RandomForestTraining(df, labels)