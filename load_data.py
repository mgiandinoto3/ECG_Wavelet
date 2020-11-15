#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:32:35 2020

@author: miagiandinoto
"""

#loop thru every file in the directory and convert the img into a (flattened) array, grayscale img
#generate a one-hot encoded label matrix
#build NN

import pandas as pd
import numpy as np
import wfdb
import ast
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D



def load_data(directory):
   x=[]
   y=[]
   for filename in os.listdir(directory):
    norm='NORM'
    mi='MI'
    name=directory + filename
    img=Image.open(name).convert('LA')
    #img=cv2.imread(filename)
    img=img.resize((432,288))
    x=np.append(x,img)
    if norm in filename:
        y=np.append(y,0)
    elif mi in filename:
        y=np.append(y,1) 
    
    return x,y

path= os.getcwd()
print(path)
x_train, y_train = load_data(os.getcwd()+'/CNNTrain/')
x_test,y_test = load_data(os.getcwd() + '/CNNTest/')

x_train.reshape(432,288,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
    
            
            
            
    

