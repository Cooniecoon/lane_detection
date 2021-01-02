import os, math

from matplotlib import pyplot as plt

from image_labelling_tool import labelling_tool

import cv2
import pandas as pd




# Load in .JPG images from the 'images' directory.
labelled_images = labelling_tool.PersistentLabelledImage.for_directory('C:/turtlebot/line_data/original_image', image_filename_patterns=['*.jpg'])
print('Loaded {0} images'.format(len(labelled_images)))

for limg in labelled_images:
    img_name=os.path.basename(limg.image_path).lower()
    print(img_name)
    labelled_img = limg

    image_shape = labelled_img.image_size

    # Get the labels
    labels = labelled_img.labels


    labels_2d = labels.render_label_classes(
        label_classes={'tree' : 64, 'stem' : 255}, image_shape=image_shape, multichannel_mask=False)


    # plt.imshow(labels_2d,cmap='gray')
    # plt.show()

    plt.imsave('C:/turtlebot/line_data/label_image/'+img_name,labels_2d,cmap='gray')
    