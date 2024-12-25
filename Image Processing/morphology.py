# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:48:36 2024

@author: Crazy_Papi
"""

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

# Generate a random image
read = plt.imread('imgdetection.jpg', format='jpg')
img = np.dot(read[...,:3],[0.2989, 0.5870, 0.1140])


# Define the kernel for morphological operations
kernel = np.ones((3, 3))

# Erosion
erosion = ndimage.binary_erosion(img, kernel)

# Dilation
dilation = ndimage.binary_dilation(img, kernel)

# Opening (Erosion followed by Dilation)
opening = ndimage.binary_opening(img, kernel)

# Closing (Dilation followed by Erosion)
closing = ndimage.binary_closing(img, kernel)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 5, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(1, 5, 2)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.subplot(1, 5, 3)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.subplot(1, 5, 4)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.subplot(1, 5, 5)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.show()