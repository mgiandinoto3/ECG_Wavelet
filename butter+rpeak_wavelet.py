#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:05:44 2020

@author: miagiandinoto
"""


from scipy.signal import butter, lfilter, filtfilt, freqz
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22}) 
from scipy import signal
import numpy as np
import os
import wfdb

from DWT_ECG import DWT_denoise, r_isolate_wavelet

os.chdir("/Users/miagiandinoto/Desktop/College/BMED 2250/phase 2 code/Data118")

sampTime = 120
sampling_freq = 360
sampTo = sampTime*sampling_freq

#read in noisy data for 2 seconds
noisy, fields = wfdb.rdsamp('118e12', channels=[0], sampfrom=108000, sampto=108000 + sampTo) #194400)
#read in clean data for 2 seconds
clean, fields0 = wfdb.rdsamp('118', channels=[0], sampfrom=108000, sampto=108000 + sampTo) #194400)
#assign variables and create time-frame
sampling_duration = sampTime
number_of_samples = fields.get('sig_len')
time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)
os.chdir('/Users/miagiandinoto/Desktop/College/BMED 2250/phase 2 code')

#butter_filter
N  = 2 # Filter order
Wn = 0.08 # Cutoff frequency bw 0 & 1 (30 Hz cutoff bcz of fourier which is 500(fs) * 0.08)
b, a = signal.butter(N, Wn, 'low') #point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”)
butter_filtered= signal.lfilter(b, a, noisy, axis=0)

butter_plot=plt.figure(figsize=(40,20))
plt.plot(time, noisy)
plt.plot(time, butter_filtered,color='green') #subtracting just for visuals
plt.legend(['Unfiltered ECG', 'Butter Filtered ECG'])
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Butter Filtered Data')
plt.savefig('Plots/2nd Order Butter @ 30.png')
plt.show()
#potential SNR calculation
ms1 = np.mean(butter_filtered**2)
ms2 = np.mean((butter_filtered-clean)**2)


xbutter = butter_filtered.T
xbutter = xbutter.flatten()

xclean = clean.T
xclean = xclean.flatten()

xwbutter = r_isolate_wavelet(xbutter,sampling_freq,sampTo)


wpeaks_butter, _ = signal.find_peaks(xwbutter, prominence=1, distance=200)
np.diff(wpeaks_butter)

peaks_clean, _ = signal.find_peaks(xclean, prominence=1, distance=200)
np.diff(peaks_clean)

#plotting r peak detection with partial wavelet deconstruction and find peaks
w_rpeaks=plt.figure(figsize=(200,100))
plt.grid(False)
plt.xlabel('Time (Sec)')
plt.ylabel('mV')
plt.title('Partial Wavelet Reconstruction R-Peaks + find_peaks',y=1.08)

ax_clean=w_rpeaks.add_subplot(3,1,1)
ax_clean.title.set_text("Clean Peaks")
plt.plot(xclean)
plt.plot(peaks_clean, xclean[peaks_clean], "ro")

ax_butter=w_rpeaks.add_subplot(3,1,2)
ax_butter.title.set_text("Butter Peaks")
plt.plot(xbutter,color='green')
plt.plot(wpeaks_butter, xbutter[wpeaks_butter], "ro")

ax_clean_butter=w_rpeaks.add_subplot(3,1,3)
plt.plot(xclean,color="blue")
plt.plot(wpeaks_butter,xclean[wpeaks_butter],"ro")

plt.tight_layout()
plt.savefig('Plots/PartialWaveletReconstructionFind_Peaks_butter.png')
plt.show()

