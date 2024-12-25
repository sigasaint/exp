# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:26:36 2024

@author: Crazy_Papi
"""

from skimage import io, feature
from skimage.transform import hough_circle_peaks, hough_circle
from skimage.draw import circle_perimeter
import numpy as np

image = io.imread('Example3.jpg', as_gray=True)
edges = feature.canny(image)

radius_range = [20, 50]
hough_radii = np.arange(radius_range[0], radius_range[1], 2)
hough_res = hough_circle(edges, hough_radii)

accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=20, min_ydistance=20)

for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius)
    image[circy, circx] = (220, 20, 20)

io.imshow(edges)
io.show()