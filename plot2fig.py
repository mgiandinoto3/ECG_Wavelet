#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 23:15:41 2020

@author: miagiandinoto
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import matplotlib.image as mpimg
import cv2


    
X=np.load('small_train_data.npy')
X=X.flatten()

# start=0
# stop=500
# for i in range(1,10):
#     tempx=X[start:stop]
#     fig=plt.figure(figsize=(12,8))
#     t=np.arange(0,len(tempx))
#     plt.style.use('grayscale')
#     ax = plt.gca()
#     plt.plot(tempx)
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
#     plt.fill_between(t,tempx,0)
#     X_norm.append(np.asarray(fig))
#     Y_norm.append(value)
#     plt.show()
#     plt.close(fig)
#     start=start+500
#     stop=stop+500


# make an agg figure
fig, ax = plt.subplots()
tempx=X[0:500]
diff=(max(tempx)+min(tempx))/2
tempx=tempx+diff
t=np.arange(0,len(tempx))
ax.plot(t, tempx)
plt.style.use('grayscale')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.fill_between(t,tempx,0)
fig.canvas.draw()

# grab the pixel buffer and dump it into a numpy array
a= np.array(fig.canvas.renderer.buffer_rgba())
img = Image.fromarray(a, 'RGBA').convert('LA')

img.show()
# width, height =image.size


