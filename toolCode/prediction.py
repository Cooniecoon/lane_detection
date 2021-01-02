import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau


YELLOW = 255
WHITE = 64

def preprocessing(img):
    img = cv2.resize(img, dsize=(320, 240), interpolation=cv2.INTER_AREA)

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    height, width, channel = img.shape
    input_img = np.zeros((width, height, 3), dtype=np.float32)
    input_img[:, :, 0] = r.T
    input_img[:, :, 1] = g.T
    input_img[:, :, 2] = b.T

    input_img = input_img[None, :, :, :]
    return input_img


def inference(img):
    preds = model.predict(input_img)

    preds = preds[0]
    output_img = preds[:, :, 0].T
    return output_img


def seperate_line(img, conf):
    _, yellow = cv2.threshold(
        output_img, YELLOW*conf/255, 255, cv2.THRESH_BINARY)

    _, white = cv2.threshold(output_img, WHITE/255, 255, cv2.THRESH_BINARY)
    tmp = output_img-yellow
    _, white = cv2.threshold(tmp, WHITE*conf/255, 255, cv2.THRESH_BINARY)
    return yellow, white


with tf.device("gpu:0"):
    inputs = Input(shape=(320, 240, 3))

    net = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    net = MaxPooling2D(pool_size=2, padding='same')(net)

    net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)
    net = MaxPooling2D(pool_size=2, padding='same')(net)

    net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)
    net = MaxPooling2D(pool_size=2, padding='same')(net)

    net = Dense(128, activation='relu')(net)

    net = UpSampling2D(size=2)(net)
    net = Conv2D(128, kernel_size=3, activation='sigmoid', padding='same')(net)

    net = UpSampling2D(size=2)(net)
    net = Conv2D(64, kernel_size=3, activation='sigmoid', padding='same')(net)

    net = UpSampling2D(size=2)(net)
    outputs = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(net)

    model = Model(inputs=inputs, outputs=outputs)


# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])

weight_path = 'C:/turtlebot/weight/line_seg.weight'
model.load_weights(weight_path)

cap = cv2.VideoCapture(1)

while True:
    _, img = cap.read()

    input_img = preprocessing(img)

    output_img = inference(input_img)

    yellow, white = seperate_line(output_img, conf=0.9)

    cv2.imshow('output_img', output_img)
    cv2.imshow('yellow', yellow)
    cv2.imshow('white', white)
    cv2.imshow('img', img)

    if cv2.waitKey(10) == ord(' '):
        cap.release()
        cv2.destroyAllWindows()
        break


# img = cv2.imread('C:/turtlebot/line_data/data_image/img_1229_15.jpg', cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# input_img=preprocessing(img)
# output_img=inference(input_img)

# _, yellow = cv2.threshold(output_img, 254/255, 255, cv2.THRESH_BINARY)

# _, white = cv2.threshold(output_img, 100/255, 255, cv2.THRESH_BINARY)
# tmp = output_img-white
# _, white = cv2.threshold(tmp, 10/255, 255, cv2.THRESH_BINARY)

# # one = np.ones((output_img.shape[0], output_img.shape[1]))

# cv2.imshow('yellow',yellow)
# cv2.imshow('white',white)
# cv2.imshow('output_img',output_img)
# cv2.imshow('img',img)
# cv2.waitKey(5454514)
