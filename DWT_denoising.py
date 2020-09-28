    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:33:56 2020

@author: miagiandinoto
    """

  #this function takes in inputs noisySignal(the raw ECG), channels, beginning point of sample,
  #and endpoint of sample. 
  #outputs the denoised signal, as well as the array of coefficients used for DWT
  #also plots denoised signal

def DWT_denoising(noisySignal, channels,sampfrom,sampto):    
    from IPython.display import display
    from skimage.restoration import denoise_wavelet
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import shutil
    import posixpath
    import statistics
    import math
    
    import wfdb
    import pywt
    
    
    
 
    
    from scipy import signal
    
    startSamp=sampfrom
    endSamp=sampto

    
    noisySignal, fields = wfdb.rdsamp(noisySignal, channels=[channels], sampfrom=startSamp,sampto=endSamp)
    N = endSamp-startSamp
    
    fig1=plt.figure(figsize=(20,10))
    #clean_signal,field1 = wfdb.rdsamp('118', channels=[1], sampfrom=110000,sampto=120000)
    #plt.plot(clean_signal)
    
    plt.plot(noisySignal,color='blue')
    
    # sos=signal.butter(3,freq,btype='highpass',analog=False,output='sos')
    # filtered=signal.sosfilt(sos,noisySignal)
    # fig4=plt.figure(figsize=(20,10))
    # plt.plot(filtered,color='green')
    
    coeffs=pywt.wavedec(noisySignal,'db6',mode='symmetric',level=8)
    
    cD1=coeffs[8]
    #THRESHOLD SELECTION
    mediancD1=np.median(cD1)
    sigma=mediancD1/.6457
    t = sigma* math.sqrt(2*math.log(N,10))
    
    
    newCoeffs=[]
    for i in coeffs:
        newDat=pywt.threshold(i,t,mode='garrote')
        newCoeffs.append(newDat)
        
    cA8=coeffs[1]
    
    
    denoised=pywt.waverec(newCoeffs,'db6',mode='symmetric')
    fig2=plt.figure(figsize=(20,10))
    plt.plot(denoised)
    return denoised,newCoeffs

