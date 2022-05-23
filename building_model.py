import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statistics as st

df = pd.read_csv('processed_data.csv',header=None,delimiter='\t',error_bad_lines=False)
if df.isnull().values.any():
    df.dropna(axis=0, inplace=True)
print(df.shape)
df.drop(columns=[0,81,82],inplace=True)
df.reset_index(drop=True,inplace=True)

print('data shape is:', df.shape)
vth =[]
igp =[]
vds = []
igss = []
dc =[]
for NoRow in range (0,len(df.index)):
    for NoCol in range (0, len(df.columns)):
        if (NoCol % 5) ==0:
            vth.append(df.iloc[NoRow,NoCol])
            NoCol +=1
        elif (NoCol % 5) ==1:
            igp.append(df.iloc[NoRow,NoCol])
            NoCol +=1
        elif (NoCol % 5) ==2:
            vds.append(pd.to_numeric(df.iloc[NoRow,NoCol]))
            NoCol +=1
        elif (NoCol % 5) ==3:
            igss.append(df.iloc[NoRow,NoCol])
            NoCol +=1
        elif (NoCol % 5) ==4:
            dc.append(df.iloc[NoRow,NoCol])
            NoCol +=1
    NoRow+=1
vth_norm = vth/st.mean(vth)
igp_norm = igp/st.mean(igp)
vds_norm = vds/st.mean(vds)
igss_norm = igss/st.mean(igss)
dc_norm = dc/st.mean(dc)
matrix = np.column_stack((vth_norm, igp_norm,igss_norm,dc_norm))
np.savetxt("timeSeries.csv", matrix, delimiter=",")


