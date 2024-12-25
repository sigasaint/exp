# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:32:24 2024

@author: Crazy_Papi
"""

from skimage import io, feature
import numpy as np

read = io.imread('imgdetection.jpg')
image = np.dot(read[...,:3],[0.2989, 0.5870, 0.1140])

edges = feature.canny(image, sigma = 1.0)

io.imshow(edges)
io.show()
