#from numpy import random
import pandas as pd
import numpy as np
import sklearn
import os.path as path
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold
#from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
import statistics
import random
#from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from sklearn.feature_selection import RFE


def initiateFeatures():
    # Set1_Set2_Set3_SMILE_noCorrel #Set1_Set2_Set3_SMILE_3D_noCorrel_new.csv
    df = pd.read_csv('../data/Set1_Set2_Set3_SMILE_3D_noCorrel.csv')
    labels = df['ln_Pe'].to_list()
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.drop(columns="ln_Pe", axis=1, inplace=True)
    #df.drop(columns="MLOGP", axis=1, inplace=True)
    #df = df[['MLOGP', 'T(N..O)','T(N..N)','piPC10', 'piPC04','piPC02', 'nHDon','PSA_w','Es_w', '|dEs|']]

    # to get the feature importance ranking when only using the features we chose (for Figure 3 in Paper).
    df = df[['ALOGP', 'T(N..O)','nHDon', 'T(N..N)', 'piPC10', 'piPC04', 'piPC02', 'MAXDN', 'PSA_w','Es_w']]

    #df.to_csv("Set1_Set2_Set3_var4.csv")

    #df.drop(columns=["Es_o",'Es_w','dEs','PSAo','abs_dEs','scaffold'], axis=1, inplace=True)
    """
    (n, bins, patches) = plt.hist(trainLabels)
    print(bins)
    print(n)
    print(patches)
    plt.show()
    """

    #return dfTrain, dfTest, trainLabels, testLabels
    return df, labels

"""
def xgboost():
    df = pd.read_csv('Membrane_Permeability_Study/Set1_Set2_Set3/Set1_Set2_Set3_SMILE_noCorrel.csv')
    labels = df['ln_Pe'].to_list()
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.drop(columns="ln_Pe", axis=1, inplace=True)
    #df.drop(columns="MLOGP", axis=1, inplace=True)
    #df = df[['BLI', 'ALOGP','H-047', 'Hy', 'PSA_w']]
    #dfSel = importanceFeaturesv02(df, labels)
    dfSel = df

    regr = XGBRegressor(max_depth=8, random_state=0)

    shuffle = KFold(n_splits=6, shuffle=True, random_state=0)
    scores = cross_val_score(regr, dfSel, labels, cv=shuffle, scoring='r2')

    #scores = cross_val_score(regr, df, labels, cv=10, scoring='r2')
    print("Mean R Squared: {}".format(np.mean(scores)))
    print ('Cross-validated scores:', scores)
"""

def stratifiedSample():
    # Set1_Set2_Set3_SMILE_noCorrel #Set1_Set2_Set3_SMILE_3D_noCorrel
    df = pd.read_csv('Membrane_Permeability_Study/Set1_Set2_Set3/Set1_Set2_Set3_SMILE_noCorrel.csv')
    labels = df['ln_Pe'].to_list()

    lab = [1,2,3,4,5,6,7,8,9,10]
    df['binned'] = pd.cut(df['ln_Pe'], bins=10, labels=lab)


    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.drop(columns="ln_Pe", axis=1, inplace=True)
    df.drop(columns="MLOGP", axis=1, inplace=True)

    binList = df['binned'].to_list()
    df.drop(columns="binned", axis=1, inplace=True)

    regr = RandomForestRegressor(max_depth=17, random_state=0)
    sfm = SelectFromModel(regr, threshold=-np.inf, max_features=10)
    # Train the selector
    dfSel= sfm.fit_transform(df, labels)
    #dfSel = df


    results = []
    mse = []
    random.seed(123)
    rand = random.sample(range(9999999), 60)
    for i in range(len(rand)):
        dfTrain, dfTest, trainLabels, testLabels = train_test_split(dfSel, labels, test_size=0.2, random_state=rand[i], shuffle=True, stratify=binList) # 
        regr.fit(dfTrain, trainLabels)
        testPred = regr.predict(dfTest)
        results.append(regr.score(dfTest, testLabels))
        error = mean_squared_error(testLabels, testPred)
        mse.append(error)

    #print('cross-validated: ', results)
    r2 = statistics.mean(results)
    print("mean r2: ", r2)
    rmse = math.sqrt(statistics.mean(mse))
    print("mean RMSE: ",rmse)


def initiate():
    # Set1_Set2_Set3_SMILE_noCorrel  #Set1_Set2_Set3_SMILE_3D_noCorrel_new
    df = pd.read_csv('../data/Set1_Set2_Set3_SMILE_3D.csv')
    labels = df['ln_Pe'].to_list()
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.drop(columns="ln_Pe", axis=1, inplace=True)

    #dfSel = importanceFeaturesv02(df, labels)

    # our final combination of features.
    dfSel = df[['ALOGP', 'PSA_w', 'T(N..O)', 'nHDon', 'T(N..N)' , 'Es_w', 'piPC02']]

    #print(dfSel)
    #dfSel = df

    regr = RandomForestRegressor(max_depth=17, random_state=0)

    #dfSel.to_csv("Set1_Set2_Set3_4variants.csv")
    
    #sfm = SelectFromModel(regr, threshold=-np.inf, max_features=50)
    
    #rfe = RFE(regr, n_features_to_select=50, step=1)
    #dfSel = rfe.fit_transform(df, labels)

    # Train the selector
    #dfSel= sfm.fit_transform(df, labels)
    #saveSel = pd.DataFrame(dfSel, columns=[df.columns[i] for i in range(len(df.columns)) if sfm.get_support()[i]])
    #saveSel.to_csv("Set3_features_sel.csv")


    #dfTrain, dfTest, trainLabels, testLabels = train_test_split(dfSel, labels, test_size=0.2, random_state=3321, shuffle=True)
    #regr = RandomForestRegressor(max_depth=8, random_state=0) # show proof why chosen max_depth = 6.
    #regr.fit(dfTrain, trainLabels)
    

    shuffle = KFold(n_splits=10, shuffle=True, random_state=0)


    scores = cross_val_score(regr, dfSel, labels, cv=shuffle, scoring='r2')

    #scores = cross_val_score(regr, df, labels, cv=10, scoring='r2')
    print("Mean R Squared: {}".format(np.mean(scores)))
    print ('Cross-validated scores:', scores)
    predictions = cross_val_predict(regr, dfSel, labels, cv=shuffle)
    #print("predictions: ", predictions)

    """
    with open('results/Set1_Set2_Set3_preds.csv', 'w') as f:
        for item in predictions:
            f.write("%s\n" % item)
    """

    accuracy = metrics.r2_score(labels, predictions)
    print ('Cross-Predicted R2:', accuracy)

    rkf = RepeatedKFold(n_splits=10, n_repeats=6, random_state=2652124)
    # evaluate model
    scoresRKF = cross_val_score(regr, dfSel, labels, scoring='r2', cv=rkf, n_jobs=-1)
    # report performance
    #print ('Repeated Cross-validated scores:', scoresRKF)
    print("Mean R Squared of repeated cross-validated scores: {}".format(np.mean(scoresRKF)))

    scoresMSE = cross_val_score(regr, dfSel, labels, scoring='neg_mean_squared_error', cv=rkf, n_jobs=-1)
    scoresRMSE = cross_val_score(regr, dfSel, labels, scoring='neg_root_mean_squared_error', cv=rkf, n_jobs=-1)
    #print ('Repeated Cross-validated scores MSE:', scoresMSE)
    print("mean MSE: {}".format(np.mean(scoresMSE)))
    #print ('Repeated Cross-validated scores root of MSE:', scoresRMSE)
    print("mean root of MSE (average residual): {}".format(np.mean(scoresRMSE)))



def RandomForest(dfTrain, dfTest, trainLabels, testLabels):
    regr = RandomForestRegressor(max_depth=8, random_state=0) # show proof why chosen max_depth = 6.
    regr.fit(dfTrain, trainLabels)
    testPred = regr.predict(dfTest)
    if not path.exists('predictions/testpred_Set1_Set2_Set3_SMILE.csv'):
        with open('predictions/testpred_Set1_Set2_Set3_SMILE.csv', 'w') as f:
            for item in testPred:
                f.write("%s\n" % item)
    trainPred = regr.predict(dfTrain)
    if not path.exists('predictions/trainpred_Set1_Set2_Set3_SMILE.csv'):
        with open('predictions/trainpred_Set1_Set2_Set3_SMILE.csv', 'w') as f:
            for item in trainPred:
                f.write("%s\n" % item)
    print("Random Forest:")
    print(regr.score(dfTest, testLabels))

def correlation(dfTrain):
    if not path.exists("correlations/featuresCorrelationSet3_2_SMILE_selFeat.csv"):
        df = dfTrain.corr() # to find if featuers are correlated and thus can be dropped without drop in accuracy.
        df.to_csv("correlations/featuresCorrelationSet3_2_SMILE_selFeat.csv")

def importanceFeatures(dfTrain, trainLabels):
    #dfTrain.drop(dfTrain.columns[[0]], axis=1, inplace=True)
    threshold = 10  # the number of most relevant features --> all featuers with MI > 0.03.
    # Reason: evaluated that this yields the best results combined with KNN.
    high_score_features = []
    file = open("Set1_Set2_Set3_SMILE_featureSelection_noCorr.csv", "w")
    counter = 1

    #print(dfTrain)

    feature_scores = mutual_info_regression(dfTrain, trainLabels, random_state=0)
    for score, f_name in sorted(zip(feature_scores, dfTrain.columns), reverse=True)[:threshold]:
            print(counter, f_name, score)
            file.write(str(counter))
            file.write("    ")
            file.write(f_name)
            file.write("    ")
            file.write(str(score))
            file.write('\n')
            high_score_features.append(f_name)
            counter = counter + 1
    dfTrainSel = dfTrain[high_score_features]
    dfTrainSel.to_csv("Set1_Set2_Set_3_combined_sel_Features.csv")

def importanceFeaturesv02(df, labels):
    threshold = 10  # the number of most relevant features --> all featuers with MI > 0.03.
    # Reason: evaluated that this yields the best results combined with KNN.
    high_score_features = []
    feature_scores = mutual_info_regression(df, labels, random_state=0)
    for score, f_name in sorted(zip(feature_scores, df.columns), reverse=True)[:threshold]:
            high_score_features.append(f_name)
    dfSel = df[high_score_features]
    return dfSel


def RFfeatureImportance(dfTrain, trainLabels):
    # random forest for feature importance on a regression problem
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from matplotlib import pyplot as plt
    # define dataset
    #X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
    # define the model
    model = RandomForestRegressor(max_depth=17, random_state=0)
    # fit the model
    model.fit(dfTrain, trainLabels)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    #print(importance)
    indices = np.argsort(importance)[::-1]
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)

    print("Feature ranking:")

    for f in range(dfTrain.shape[1]):
        print("%d. %s (%f)" % (f + 1, dfTrain.columns[indices[f]], importance[indices[f]]))

def permutationImport(dfTrain, trainLabels):
    from sklearn.inspection import permutation_importance
    import eli5
    from eli5.sklearn import PermutationImportance

    model = RandomForestRegressor()
    # fit the model
    reg = RandomForestRegressor()
    # fit the model
    model = reg.fit(dfTrain, trainLabels)

    perm = PermutationImportance(reg, random_state=1).fit(dfTrain, trainLabels)
    print(eli5.show_weights(perm))



if __name__ == "__main__":
    
    # run RF regression with chosen features.
    #initiate()

    # run RF feature selection alhgorithm.
    dfTrain, trainLabels = initiateFeatures()
    RFfeatureImportance(dfTrain, trainLabels)


    # methods used for previous experiments.

    #stratifiedSample()
    #xgboost()
    #RandomForest(dfTrain, dfTest, trainLabels, testLabels)
    #correlation(dfTrain)
    #importanceFeatures(dfTrain, trainLabels)
    #permutationImport(dfTrain, trainLabels)

    #dfTrain, dfTest, trainLabels, testLabels = initiateFeatures()


    
