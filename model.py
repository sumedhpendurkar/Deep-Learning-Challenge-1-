import keras
from keras.models import Sequential
from keras.layers import Deconv2D, Conv2D, Input, BatchNormalization, MaxPooling2D
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy as np
from keras.applications.vgg19 import VGG19


def model():
    model = Sequential()
    model.add(Conv2D(32, (3*3), kernel_initializer = 'glorot_uniform',
        activation = 'relu', use_bias=True, input_shape=(256,256,3)))
    model.add(Conv2D(32, (5*5), kernel_initializer = 'glorot_uniform',
        activation = 'relu', use_bias=True))
    model.add(MaxPooling2D((3,3), strides=(1,1)))
    model.add(Conv2D(64, (7*7), kernel_initializer = 'glorot_uniform',
        activation = 'relu', use_bias=True))
    model.add(Conv2D(64, (5*5), kernel_initializer = 'glorot_uniform',
        activation = 'relu', use_bias=True))
    model.add(MaxPooling2D((3,3), strides=(1,1)))
    model.add(Conv2D(128, (7*7), kernel_initializer = 'glorot_uniform',
        activation = 'relu', use_bias=True))
    model.add(Conv2D(128, (7*7), kernel_initializer = 'glorot_uniform',
        activation = 'relu', use_bias=True))
    model.add(MaxPooling2D((2,2)))
    model.compile(loss = 'categorical_crossentropy',  optimizer = Adam(lr = 0.0001, decay = 1e-6))
    return model

if __name__ == '__main__':
    base_model = VGG19(weights='imagenet', include_top= False, input_shape=(256,256,3))
    for layer in base_model.layers[:]:
        layer.trainable = False
    base_model.summary()
