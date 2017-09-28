import keras
from keras.models import Sequential
from keras.layers import Deconv2D, Conv2D, Input, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications import InceptionV3 
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint

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

def transfered_vgg19():
    base_model = VGG19(weights='imagenet', include_top= False, input_shape=(256,256,3))
    for layer in base_model.layers[:]:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(25, activation="softmax")(x)
    base_model = keras.models.Model(base_model.input, predictions)
    base_model.compile(loss = 'categorical_crossentropy',  optimizer = Adam(lr = 0.0001, decay = 1e-6))
    base_model.summary()
    return base_model

def transfered_inception():
    base_model = InceptionV3(weights='imagenet', include_top= False, pooling='max',input_shape=(256,256,3))
    #for layer in base_model.layers[:]:
    #    layer.trainable = False
    x = base_model.output
    #x = Flatten()(x)
    x = Dense(1024, activation="relu", kernel_regularizer = keras.regularizers.l2(0.0001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu", kernel_regularizer = keras.regularizers.l2(0.0001))(x)
    predictions = Dense(25, activation="softmax", kernel_regularizer = keras.regularizers.l2(0.0001))(x)
    base_model = keras.models.Model(base_model.input, predictions)
    base_model.compile(loss = 'categorical_crossentropy',  optimizer = Adam(lr = 0.0001, decay = 1e-6))
    return base_model



if __name__ == '__main__':
    model = transfered_inception()
    X_train = preprocess_input(np.load('training_X'))
    Y_train = np.load('training_Y')

    Y_train = keras.utils.to_categorical(Y_train, 25) 
    model.summary()
    model.fit(X_train, Y_train, verbose = 2, epochs=60, batch_size=8, validation_split=0.1, 
            callbacks=[ModelCheckpoint('inception-transferlearning_model3.h5', monitor='val_loss', save_best_only=True)])
    model.save('model3.h5',  overwrite = True)
