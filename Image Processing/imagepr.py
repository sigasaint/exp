# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:50:10 2024

@author: Crazy_Papi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# Generate a random image
img = plt.imread('Example2.png', format='png')

# Convert to grayscale
gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# Apply Gaussian filter to blur the image
blurred = ndimage.gaussian_filter(gray, sigma=3)

# Apply threshold to segment the image
thresh = np.where(blurred > 0.5, 1, 0)

# Display the original and processed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(blurred, cmap='gray')
plt.title('Blurred Image')
plt.subplot(1, 3, 3)
plt.imshow(thresh, cmap='gray')
plt.title('Thresholded Image')
plt.show()