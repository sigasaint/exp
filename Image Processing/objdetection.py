# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:38:09 2024

@author: Crazy_Papi
"""

# importing the neeeded packages
import matplotlib.pyplot as plt
from skimage import io, feature, filters, morphology, exposure

# Load the image & Convert the image to grayscale
image = io.imread('blobs.jpg', as_gray=True)

# Apply image equalization
equalized_image = exposure.equalize_hist(image)

# Apply a median filter to smooth the image
# The size of the filter can be adjusted to suit your needs
smoothed_image = filters.median(equalized_image,morphology.disk(3))

# Detected blobs using the difference of Gaussian method
# Set parameters for the blob detection
blob_dog = feature.blob_dog(smoothed_image, max_sigma=30, threshold=0.1)

# Visualize the results
fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.imshow(image, cmap='gray')

# Plot the detected blobs
for blob in blob_dog:
    y, x, r = blob
    circle = plt.Circle((x,y), r, color='red', fill = False)
    ax.add_artist(circle)
ax.set_title('Blob Detection')
plt.axis('off')
plt.show()