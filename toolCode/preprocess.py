import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import pyramid_reduce, resize

import os, glob

img_list = sorted(glob.glob('C:/turtlebot/line_data/data_image/*.jpg'))
mask_list = sorted(glob.glob('C:/turtlebot/line_data/label_image/*.jpg'))

print(len(img_list), len(mask_list))


IMG_SIZE_X = 320
IMG_SIZE_Y = 240


x_data = np.empty((len(img_list), IMG_SIZE_X, IMG_SIZE_Y, 3), dtype=np.float16)
y_data = np.empty((len(img_list), IMG_SIZE_X, IMG_SIZE_Y, 1), dtype=np.float16)

for i, img_path in enumerate(img_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE_X, IMG_SIZE_Y, 3), preserve_range=True)
    x_data[i] = img
    
for i, img_path in enumerate(mask_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE_X, IMG_SIZE_Y, 1), preserve_range=True)
    y_data[i] = img
    
y_data /= 255.

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)

np.save('C:/turtlebot/line_data/dataset/x_train.npy', x_train)
np.save('C:/turtlebot/line_data/dataset/y_train.npy', y_train)
np.save('C:/turtlebot/line_data/dataset/x_val.npy', x_val)
np.save('C:/turtlebot/line_data/dataset/y_val.npy', y_val)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)