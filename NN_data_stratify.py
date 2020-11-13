#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:16:45 2020

@author: miagiandinoto
"""

import pandas as pd
import numpy as np
import wfdb
import ast
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f,channels=[0]) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f,channels=[0]) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = '/Users/miagiandinoto/Desktop/College/BMED 2250/phase 2 code/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate=500

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train


X_train = X[np.where(Y.strat_fold != test_fold)]
X_small=X[0]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
y_small=y_train[0:2]
X_norm=[]
Y_norm=[]

new_path='/Users/miagiandinoto/Desktop/College/BMED 2250/phase 2 code/'
for index, value in y_small.items():
    if value == ['NORM']or value==['MI']:
        x=X[index-1]
        x=x[0:5000]
        x_flat=x.flatten()
        #time=np.arange(0,len(x))
        #time=time/sampling_rate
        start=0
        stop=500
        for i in range(1,10):
            tempx=x_flat[start:stop]
            fig=plt.figure(figsize=(12,8))
            t=np.arange(0,len(tempx))
            plt.style.use('grayscale')
            ax = plt.gca()
            plt.plot(tempx)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.fill_between(t,tempx,0)
            X_norm.append(np.asarray(fig))
            Y_norm.append(value)
            plt.show()
            plt.close(fig)
            start=start+500
            stop=stop+500
            
        
        #Y_norm.append(value)
        
    #    #fig=plt.figure
    # elif value==['MI']:
    #     xmi=X[index-1]
    #     X_mi.append(xmi)
    #     Y_mi.append(value)
 
#Y_series_train_norm=pd.Series(Y_norm)

# # Test
X_test = X[np.where( Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
y_test_small=y_test[0:2]

X_test_myset=[]
Y_test_myset=[]
for index, value in y_test_small.items():
    if value == ['NORM'] or value==['MI']:
        x=X[index-1]
        x=x.flatten()
        time=np.arange(0,len(x))
        start=0
        stop=500
        for i in range(1,10):
            tempx=x_flat[start:stop]
            fig=plt.figure(figsize=(12,8))
            t=np.arange(0,len(tempx))
            plt.style.use('grayscale')
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.plot(tempx)
            plt.fill_between(t,tempx,0)
            X_test_myset.append(np.array(fig))
            Y_test_myset.append(value)
           # plt.show()
            plt.close(fig)
            start=start+500
            stop=stop+500

 
# os.chdir(new_path)
np.save('small_train_data.npy',X_small)
# np.save('test_data.npy',X_test_myset)

#Neural Netwrok mf
#tensorflow_version 2.x
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

#check image shape
print(X_norm[0].shape)




