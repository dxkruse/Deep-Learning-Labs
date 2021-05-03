# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 02:26:35 2021

@author: Dietrich
"""


import numpy as np
import cv2
from skimage import data
from skimage.transform import warp_polar, rescale
import matplotlib.pyplot as plt
from numpy.linalg import norm

img = cv2.imread('test.png')
H = len(img[:,0])
W = len(img[0,:])

x0 = W/2
y0 = H/2

warp = warp_polar(img, scaling = 'log', multichannel=True)






fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
ax[0].imshow(img)
ax[1].imshow(warp)
scale = [0.5,0.5]
rescaled = rescale(img, scale, multichannel=True)
rescaled_warped = warp_polar(rescaled, scaling='log', multichannel=True)
ax[2].imshow(rescaled)
ax[3].imshow(rescaled_warped)