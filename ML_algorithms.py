from keras import models
from keras import layers
import numpy as np
import utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def keras_shallow(input_dim=39, n_layers=3, n_neurons=6, r_dropout=0.15, optimizer="rmsprop", loss="binary_crossentropy", init="uniform", metrics=["accuracy"], random_state=42):
    """
    shallow neural net, define the amount of layers either having 1 or 2 hidden layers., which optimizer you want to use and which loss function you want to apply.
    """

    model = models.Sequential()
    model.add(layers.Dense(n_neurons, activation="relu", input_dim=input_dim))
    model.add(layers.Dropout(r_dropout))
    for num in range(n_layers-2):
        model.add(layers.Dense(n_neurons, activation="relu"))
        model.add(layers.Dropout(r_dropout))
    model.add(layers.Dense(1, activation="sigmoid", init=init))
    model.compile(optimizer, loss, metrics=metrics)
    
    return model

def keras_deep(input_dim=39, n_layers=9, n_neurons=12, r_dropout=0.5, optimizer="rmsprop", loss="binary_crossentropy", init="uniform", metrics=["accuracy"], random_state=42):
    """
    Keras Neural Network, define the amount of layers you want, which optimizer you want to use and which loss function you want to apply.
    """ 
    model = models.Sequential()
    model.add(layers.Dense(n_neurons, activation="relu", input_dim=input_dim))
    model.add(layers.Dropout(r_dropout))
    for num in range(n_layers-2):
        model.add(layers.Dense(n_neurons, activation="relu"))
        model.add(layers.Dropout(r_dropout))
    model.add(layers.Dense(1, activation="sigmoid", init=init))
    model.compile(optimizer, loss, metrics=metrics)
    
    return model


def keras_cnn(n_neurons=32, n_layers=3, filter_size=(3, 3), activation="relu", 
               input_shape =(64,64,3), max_pooling=(2,2), dense_layer=128, 
               loss="binary_crossentropy",optimizer="adam",metrics="acc"):
    # NOTE: always alter the input_shape to the specific input shape off the problem.

    model = models.Sequential()
    model.add(layers.Conv2D(n_neurons, filter_size, activation=activation,
                           input_shape =input_shape))
    model.add(layers.MaxPooling2D(max_pooling))
    for num in range(n_layers-2):
        model.add(layers.Conv2D(n_neurons, filter_size, activation=activation))
        model.add(layers.MaxPooling2D(max_pooling))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_layer, activation=activation))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss=loss, 
                  optimizer=optimizer,
                  metrics=[metrics])
    
    return model

def keras_cnn_conv1D(filters=64, n_layers=2, kernel_size=2, activation="relu", 
               input_shape =(3,15), pool_size=2, dense_layer=50, 
               loss="binary_crossentropy",optimizer="adam",metrics="acc"):

    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=pool_size))
    for num in range(n_layers-2):
        model.add(Conv1D(filters=filters, kernel_size=kernen_size, activation=activation, input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=pool_size))    
    model.add(Flatten())
    model.add(Dense(dense_layer, activation=activation))
    model.add(Dense(1))
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metrics])
    
    return model

from keras.layers import SimpleRNN, Embedding

def simple_rnn(max_features, n_neurons=32, activation="sigmoid", optimizer="rmsprop",
               loss="binary_crossentropy", metrics=["acc"]):
    
    model = models.Sequential()
    model.add(Embedding(max_features, n_neurons))
    model.add(SimpleRNN(n_neurons))
    model.add(models.Dense(1, activation=activation))

    model.compile(optimizer=optimizer, loss=los, metrics=metrics)
    
    return model


def keras_lstm(max_features, n_neurons, optimizer="rmsprop",
               loss="binary_crossentropy", metrics = 'acc'):

    model = models.Sequential()
    model.add(layers.Embedding((max_features, n_neurons)))
    model.add(layers.LSTM(n_neurons))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer= optimizer,
                  loss = loss,
                  metrics=[metrics])

    return model