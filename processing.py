import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statistics as st
# import data and remove malformed lines
df = pd.read_table('cMCU-Output.txt',header=None,delimiter=',',error_bad_lines=False)
if df.isnull().values.any():
    df.dropna(axis=0, inplace=True)

# remove unrelated data: Time and MCU ID
df.drop(columns=[0,1],inplace=True)
df.reset_index(drop=True,inplace=True)
print(df.shape)
# show infos
# print(df.head())
# print(df.info())
# print(df)

# show the raw data of a single variable

numCols = len(df.columns)

for n in range (0, numCols-2):
    numRows = df.shape[0]
    y = np.zeros(numRows)
    x = np.zeros(numRows, dtype=np.int16)
    for curRow in range(0,numRows):
        x[curRow] = curRow
        y[curRow] = df.iloc[curRow,n]
    y_mean = st.mean(y)
    outlier_index = []
    for i in x:
        if y[i] > (2 * y_mean):
            outlier_index.append(i)

    df.drop(index=outlier_index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.shape)
df.to_csv('processed_data.csv', sep='\t')
# ############################# remove the outliers####################################
# calc the mean and drop the obvious outlier
# y_mean = st.mean(y)
# outlier_index = []
# for i in x:
#     if y[i] > (2*y_mean):
#         outlier_index.append(i)
#
# df.drop(index=outlier_index,inplace=True)
# df.reset_index(drop=True,inplace=True)
# print(df.shape)
# y_ascending = np.sort(y)
# upperFence = 2000
# LowerFence = 1000
# for iter in y:
#     if iter>upperFence or iter < LowerFence:
#         y.drop()
#
#
# if y_ascending.size % 2 == 0:
#     medIdx = [int(y_ascending.size/2)-1, int(y_ascending.size/2)]
#     medArrFH = y_ascending[0:int(y_ascending.size/2)-1]
#     medArrSH = y_ascending[int(y_ascending.size/2)+1:]
#     medFH = np.median(medArrFH)
#     medSH = np.median(medArrSH)
#     IQR = medSH - medFH
#     # upperFence = medSH + 1.5 * IQR
#     # LowerFence = medFH - 1.5 * IQR
#     # upperFence = 2000
#     # LowerFence = 1000
# else:
#     medIdx = int(y_ascending.size/2)
#
# print(y_ascending)
# # medIdx = y_ascending.index(np.percentile(y_ascending,50,interpolation='nearest'))
# median = np.median(y_ascending)
# pos_med = np.where(y_ascending==median)
# plt.plot(x,y_ascending)
# plt.show()

#hample
window_size = 6
n_sigmas = 3
n = len(y)
new_series = y.copy()
k = 1.4826

indices = []

for i in range ((window_size),(n-window_size)):
    x0 = np.median(y[(i - window_size):(i + window_size)])
    S0 = k * np.median(np.abs(y[(i - window_size):(i + window_size)] - x0))
    if (np.abs(y[i] - x0) > n_sigmas * S0):
        new_series[i] = x0
        indices.append(i)

fig, ax = plt.subplots(2)
ax[0].plot(x, y)
ax[0].set_title('Original data')
ax[0].grid()

ax[1].plot(x, new_series)
ax[1].set_title('processed data')
ax[1].grid()

fig.show()
# plt.plot(x,new_series)
# plt.show()
