# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:34:46 2024

@author: Crazy_Papi
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from sklearn.svm import LinearSVC
from skimage.segmentation import felzenszwalb
import os

# Load the training images
train_dir = 'training_images'
images = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]

# Extract HOG features and labels
X = []
y = []

for image in images:
    fd = hog(data.load_image(image), orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), visualize=True)[0]
    X.append(fd)
    if 'house' in image:
        y.append(1)
    elif 'car' in image:
        y.append(2)
    elif 'tree' in image:
        y.append(3)
    else:
        y.append(0)

# Train an SVM classifier
svm = LinearSVC()
svm.fit(X, y)

# Load the satellite image
read = data.load_image('satellite_image.jpg')
image = np.dot(read[...,:3],[0.2989, 0.5870, 0.1140])

# Preprocess the image
image = exposure.adjust_gamma(image, 2)
image = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

# Detect objects in the image
def detect_objects(image, svm):
    labeled_image, num_objects = (image)
    detections = []
    for i in range(1, num_objects + 1):
        object_mask = labeled_image == i
        object_image = image * object_mask
        fd = hog(object_image, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=True)[0]
        prediction = svm.predict([fd])
        if prediction != 0:  # Ignore background class
            detections.append((object_mask, prediction[0]))
    return detections

detections = detect_objects(image, svm)

# Display the results
fig, ax = plt.subplots()
ax.imshow(image)
for detection in detections:
    object_mask, class_name = detection
    ax.imshow(object_mask, alpha=0.5)
    ax.text(10, 10, ['house', 'car', 'tree'][class_name - 1], color='white')
plt.show()