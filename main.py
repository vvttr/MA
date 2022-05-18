import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
# import data and remove malformed lines
df = pd.read_table('cMCU-Output.txt',header=None,delimiter=',',error_bad_lines=False)
if df.isnull().values.any():
    df.dropna(axis=0, inplace=True)

# remove unrelated data: Time and MCU ID
df.drop(columns=[0,1],inplace=True)
df.reset_index(drop=True,inplace=True)

# show infos
# print(df.head())
# print(df.info())
# print(df)

# show the raw data of a single variable
numRows = df.shape[0]
y = np.zeros(numRows)
x = np.zeros(numRows)
for curRow in range(0,numRows):
    x[curRow] = curRow
    y[curRow] = df.iloc[curRow,0]


plt.plot(x,y)
plt.show()

# ############################# remove the outliers####################################

y_ascending = np.sort(y)
upperFence = 2000
LowerFence = 1000
for iter in y:
    if iter>upperFence or iter < LowerFence:
        y.drop()


if y_ascending.size % 2 == 0:
    medIdx = [int(y_ascending.size/2)-1, int(y_ascending.size/2)]
    medArrFH = y_ascending[0:int(y_ascending.size/2)-1]
    medArrSH = y_ascending[int(y_ascending.size/2)+1:]
    medFH = np.median(medArrFH)
    medSH = np.median(medArrSH)
    IQR = medSH - medFH
    # upperFence = medSH + 1.5 * IQR
    # LowerFence = medFH - 1.5 * IQR
    # upperFence = 2000
    # LowerFence = 1000
else:
    medIdx = int(y_ascending.size/2)

print(y_ascending)
# medIdx = y_ascending.index(np.percentile(y_ascending,50,interpolation='nearest'))
median = np.median(y_ascending)
pos_med = np.where(y_ascending==median)
plt.plot(x,y_ascending)
plt.show()

