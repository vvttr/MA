import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statistics as st
from sklearn import preprocessing, model_selection
import tensorflow as tf
from tensorflow import keras as ke
from keras import layers

# # -------------------------process the data(dimension)-----------------
# # Cast the all the data to float64 and
# # delete the unrelated data(column 0, 82 and 83)
# df = pd.read_csv('processed_data.csv', header=None, delimiter='\t', error_bad_lines=False)
# if df.isnull().values.any():
#     df.dropna(axis=0, inplace=True)
# print(df.shape)
# df.drop(columns=[0, 81, 82], inplace=True)
# df.reset_index(drop=True, inplace=True)
# df = df.astype('float64')
# print('data shape is:', df.shape)
#
# # transform the data in 1D-vectors in forms of different parameters
# vth = []
# igp = []
# vds = []
# igss = []
# dc = []
# for NoRow in range(0, len(df.index)):
#     for NoCol in range(0, len(df.columns)):
#         if (NoCol % 5) == 0:
#             vth.append(df.iloc[NoRow, NoCol])
#             NoCol += 1
#         elif (NoCol % 5) == 1:
#             igp.append(df.iloc[NoRow, NoCol])
#             NoCol += 1
#         elif (NoCol % 5) == 2:
#             vds.append((df.iloc[NoRow, NoCol]))
#             NoCol += 1
#         elif (NoCol % 5) == 3:
#             igss.append(df.iloc[NoRow, NoCol])
#             NoCol += 1
#         elif (NoCol % 5) == 4:
#             dc.append(df.iloc[NoRow, NoCol])
#             NoCol += 1
#     NoRow += 1
# # save them in csv file
# matrix = np.column_stack((vth, igp, igss, dc, vds))
# np.savetxt("timeSeries.csv", matrix, delimiter=",", header="vth,igp,igss,dc,vds", comments='')

# --------------------------------------process the input-------------------------
df = pd.read_csv('timeSeries.csv', header=None, delimiter=',', on_bad_lines='error')
if df.isnull().values.any():
    df.dropna(axis=0, inplace=True)
df.drop(index=[0], inplace=True)
print('The original data has the shape of', df.shape)
df_input = df.drop(columns=[4])
print('The input data has the shape of', df_input.shape)
df_output = df.drop(columns=[0, 1, 2, 3])
print('The output data has the shape of', df_output.shape)
print('-------------------------------------------------\n'
      'the first columns of input data are:\n',
      df_input.head(),
      '\n-------------------------------------------------')

# ---------------------build the model----------------------------
df_in_norm = preprocessing.normalize(df_input)
df_out_norm = preprocessing.normalize(df_output)
print(df_in_norm.shape)
train, test = model_selection.train_test_split(df_in_norm, test_size=0.5)
train_y, test_y = model_selection.train_test_split(df_out_norm, test_size=0.5)
timeStamps, features = df_in_norm.shape  # how many TIMESTAMPS and FEATURES in the input data
print('There are', timeStamps, 'timestamps in the input data.')
print('There are', features, 'features in the input data.')
model = ke.Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(1, features)))
model.add(layers.Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')
print(model.summary())
model.fit(df_in_norm,df_out_norm,epochs=200,verbose=0)
# vth_norm = vth / st.mean(vth)
# igp_norm = igp / st.mean(igp)
# vds_norm = vds / st.mean(vds)
# igss_norm = igss / st.mean(igss)
# dc_norm = dc / st.mean(dc)
