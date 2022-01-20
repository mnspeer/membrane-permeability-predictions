# Program to combine multiple datasets that have partially different headers, only keeping the common columns.

import pandas as pd


#df1 = pd.read_csv('Membrane_Permeability_Study/Set_3/Set3_combined/Set3_1_SMILE_3D.csv')
df1 = pd.read_csv('Membrane_Permeability_Study/no_LN_PERM/Set1_SMILE_3D_no_LN_PERM.csv')
print(df1.shape)
#df2 = pd.read_csv('Membrane_Permeability_Study/Set_3/Set3_combined/Set3_2_SMILE_3D.csv')
df2 = pd.read_csv('Membrane_Permeability_Study/no_LN_PERM/Set2_SMILE_3D_no_LN_PERM.csv')
print(df2.shape)
df3 = pd.read_csv('Membrane_Permeability_Study/no_LN_PERM/Set3_1_SMILE_3D_no_LN_PERM.csv')
#print(df3.shape)
df4 = pd.read_csv('Membrane_Permeability_Study/no_LN_PERM/Set3_2_SMILE_3D_no_LN_PERM.csv')
#print(df4.shape)


#df_new = pd.merge(df1,df2,sort=False)
df_new = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=True, join='inner')
print(df_new.shape)
df_new.to_csv('Set1_Set2_Set3_SMILE_3D_no_LN_PERM.csv')