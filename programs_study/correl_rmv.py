# program finds correlated features in dataset, and when |correlation| >= 0.95, then the features are marked as correlated.
# From all correlated groups, only one feature is kept, which is the feature with the highest correlation with ln_Pe. 


import pandas as pd
import math

def correlation(dataset, threshold, target):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        colname_i = corr_matrix.columns[i]
        toDelete = []
        # columns where a correlation cannot be determined can be removed.
        if(math.isnan(dataset[colname_i].corr(dataset[target]))):
            col_corr.add(colname_i)
            del dataset[colname_i]
        
        else:
            highest = colname_i
            if(corr_matrix.columns[i] not in col_corr):
                colname_i = corr_matrix.columns[i]
                maxCorrel = abs(dataset[colname_i].corr(dataset[target]))
                toDelete = []
                highest = colname_i
                for j in range(i):
                    colname_j = corr_matrix.columns[j]       
                    if (abs(corr_matrix.iloc[i, j]) >= threshold) and (colname_j not in col_corr) and (colname_j not in toDelete):
                        colname_i = corr_matrix.columns[i] # getting the name of column
                        if(abs(dataset[colname_j].corr(dataset[target])) > maxCorrel):
                            maxCorrel = abs(dataset[colname_j].corr(dataset[target]))
                            toDelete.append(highest)
                            highest = colname_j
                        else:
                            toDelete.append(colname_j)
                for i in range(len(toDelete)):
                    col_corr.add(toDelete[i])
                    if toDelete[i] in dataset.columns:
                        del dataset[toDelete[i]] # deleting the column from the dataset
        #print(len(toDelete))

    return dataset

if __name__ == "__main__":
    dataset = pd.read_csv('../data/Set1_Set2_Set3_SMILE_3D.csv')
    dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
    print(dataset)

    # target and threshold can be adjusted by adjusting the values passed into the method below.  
    dataset_noCorrel = correlation(dataset, 0.95, 'ln_Pe')    
    print(dataset_noCorrel)
    #dataset_noCorrel.corr().to_csv("Set3_SMILE_3D_correlation.csv")
    dataset_noCorrel.to_csv('../data/Set1_Set2_Set3_SMILE_3D_noCorrel.csv')

