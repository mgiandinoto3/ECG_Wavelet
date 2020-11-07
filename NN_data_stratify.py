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
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass

X_norm=[]
Y_norm=[]
X_mi=[]
Y_mi=[]

for index, value in y_train.items():
    if value == ['NORM']:
        x=X[index-1]
        X_norm.append(x)
        Y_norm.append(value)
    elif value==['MI']:
        xmi=X[index-1]
        X_mi.append(xmi)
        Y_mi.append(value)
 
Y_series_train_norm=pd.Series(Y_norm)

# Test
X_test = X[np.where( Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

X_test_myset=[]
Y_test_myset=[]
for index, value in y_test.items():
    if value == ['NORM'] or value==['MI']:
        x=X[index-1]
        X_test_myset.append(x)
        Y_test_myset.append(value)
    
        
Y_series_test=pd.Series(Y_test_myset)



