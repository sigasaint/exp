# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:49:57 2024

@author: Crazy_Papi
"""

from skimage import io, filters, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

image = io.imread('imgdetection.jpg', as_gray=True)
gradient = filters.sobel(image)
distance = ndimage.distance_transform_edt(gradient)
local_max = peak_local_max(distance, footprint=morphology.disk(3))
markers = morphology.label(local_max)
labels = watershed(gradient, markers)
io.imshow(labels, cmap='nipy_spectral')
io.show()