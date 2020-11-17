#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:55:03 2020

@author: miagiandinoto
"""

from scipy.signal import butter, lfilter, filtfilt, freqz, welch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22}) 
from scipy import signal
import numpy as np
import os
import wfdb
import heartpy as hp
from DWT_ECG import r_isolate_wavelet

from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import welch, periodogram

#get signal- this is just a sample record from the MIT database
os.chdir("/Users/miagiandinoto/Desktop/College/BMED 2250/phase 2 code/Data118")

clean, fields = wfdb.rdsamp('118', channels=[0], sampfrom=0, sampto= 110000)
clean=clean.flatten()
fs=fields.get('fs')
xclean=r_isolate_wavelet(clean,fs,len(clean))
peaks, _ = signal.find_peaks(xclean, prominence=0.5*max(clean), distance=200)
ybeat=clean[peaks]

#function takes in the list of detected rpeaks, list of where each peak occurs in the signal(ybeat)
#ybeat should just be signal[peaks]
#fs is sampling frequency
def get_hrv(peaks,ybeat,fs,method):
    #create empty dicts 'wd' and 'measures'. 
    wd={}
    measures={}
    
    #populate 'wd' with inputs
    wd['peaklist']=peaks 
    wd['fs']=fs
    wd['ybeat']=ybeat
   
   #initialize RR_list variable, calculate RR intervals
    RR_list=[]
    RR_list = (np.diff(peaks) / fs) * 1000.0
    rr_indices = [(peaks[i], peaks[i+1]) for i in range(len(peaks) - 1)]
     
    #populate 'wd' with RR_list
    wd['RR_list']=RR_list
    
    #get rid of outlier RR intervals
    wd=hp.peakdetection.check_peaks(RR_list,peaks,ybeat,reject_segmentwise=True,working_data=wd)
    wd=hp.analysis.update_rr(wd)
    RR_list=wd['RR_list_cor']
    
    #resample the data at regular intervals in order to perform frequency analysis by using linear
    #interpolation
    rr_x = []
    pointer = 0
    for x in RR_list:
        pointer += x
        rr_x.append(pointer)
    start=int(rr_x[0])
    rr_x_new = np.linspace(start, int(rr_x[-1]), int(rr_x[-1]))
    interpolation_func = interp1d(rr_x, RR_list,fill_value='extrapolate')
    rr_interp=interpolation_func(rr_x_new)
    wd['rr_interp']=rr_interp
    
    #get new sampling frequency- should be about 1.1 Hz(about 1 RR interval per beat)
    dt = np.mean(RR_list)/1000
    fs_new=1/dt
    
       # compute PSD (one-sided, units of ms^2/Hz)
    if method=='fft':
        datalen=len(rr_x_new)
        frq = np.fft.fftfreq(datalen, d=(1/fs_new))
        frq = frq[range(int(datalen/2))]
        Y = np.fft.fft(rr_interp)/datalen
        Y = Y[range(int(datalen/2))]
        psd = np.power(Y, 2)

    elif method=='periodogram':
        frq, psd = periodogram(rr_interp, fs=fs_new)

    else:
        raise ValueError("specified method incorrect, use 'fft' or 'periodogram'")
    
    wd['psd'] = psd
    
    # compute absolute power band measures (units of ms^2)
    df = frq[1] - frq[0]
    measures['vlf'] = np.trapz(abs(psd[(frq >= 0.0033) & (frq < 0.04)]), dx=df)
    measures['lf'] = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)]), dx=df)
    measures['hf'] = np.trapz(abs(psd[(frq > 0.15) & (frq < 0.4)]), dx=df)
    measures['lf/hf'] = measures['lf'] / measures['hf']
    
    measures['p_total'] = measures['vlf'] + measures['lf'] + measures['hf']
    
    # compute relative and normalized power measures(we dont need these but eh)
    perc_factor = 100/measures['p_total']
    measures['vlf_perc'] = measures['vlf']*perc_factor
    measures['lf_perc'] = measures['lf']*perc_factor
    measures['hf_perc'] = measures['hf']*perc_factor
    
    nu_factor = 100/(measures['lf'] + measures['hf'])
    measures['lf_nu'] = measures['lf']*nu_factor
    measures['hf_nu'] = measures['hf']*nu_factor
    
    
    wd['interp_rr_function'] = interpolation_func
    wd['interp_rr_linspace'] = rr_x_new
 
    
    bpm = 60000 / np.mean(RR_list) #60000 ms (1 minute) / average R-R interval of signal
    measures['bpm']=bpm
    measures['ibi'] = np.mean(RR_list)
    
   
    #plt.plot(RR_list)
    return wd,measures

#call function, print lf/hf ratio and heart rate(bpm)
wdata,measures=get_hrv(peaks,ybeat,fs,'fft')

print('lf/hf ratio = %.3f' %measures['lf/hf'])
print ("Average Heart Beat is: %.01f" %measures['bpm']) #Round off to 1 decimal and print
#plt.plot(wdata['rr_interp'])
#plt.show()
