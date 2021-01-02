
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU, UpSampling2D
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau

x_train = np.load('C:/turtlebot/line_data/dataset/x_train.npy')
y_train = np.load('C:/turtlebot/line_data/dataset/y_train.npy')
x_val = np.load('C:/turtlebot/line_data/dataset/x_val.npy')
y_val = np.load('C:/turtlebot/line_data/dataset/y_val.npy')


print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


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
    

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])


history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=16, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
])

save_path='C:/turtlebot/weight/line_seg.weight'
model.save_weights(save_path)

