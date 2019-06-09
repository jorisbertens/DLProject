from keras import models
from keras import layers
import numpy as np
import utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def keras_shallow(input_dim=39, n_layers=3, n_neurons=6, r_dropout=0.15, optimizer="rmsprop", loss="binary_crossentropy", init="uniform", metrics=utils.f1):
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
    model.compile(optimizer, loss, metrics=[metrics, 'binary_accuracy', 'accuracy'])
    
    return model

def keras_deep(input_dim=39, n_layers=9, n_neurons=18, r_dropout=0.15, optimizer="rmsprop", loss="binary_crossentropy", init="uniform", metrics=utils.f1):
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
    model.compile(optimizer, loss, metrics=[metrics, 'binary_accuracy', 'accuracy'])
    
    return model


def keras_cnn(n_neurons=32, n_layers=3, filter_size=(3, 3), activation="relu", 
               input_shape =(64,64,3), max_pooling=(2,2), dense_layer=128, 
               loss="binary_crossentropy",optimizer="adam",metrics= utils.f1):
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
                  metrics=[metrics, 'binary_accuracy', 'accuracy'])
    
    return model

def keras_cnn_conv1D(filters=64, n_layers=2, kernel_size=2, activation="relu", 
               input_shape =(3,15), pool_size=2, dense_layer=50, 
               loss="binary_crossentropy",optimizer="adam",metrics= utils.f1):

    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=pool_size))
    for num in range(n_layers-2):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=pool_size))    
    model.add(Flatten())
    model.add(Dense(dense_layer, activation=activation))
    model.add(Dense(1))
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metrics, 'binary_accuracy', 'accuracy'])
    
    return model

from keras.layers import SimpleRNN, Embedding

def simple_rnn(train_X=None, input_shape=None, optimizer="rmsprop",
               loss="binary_crossentropy", metrics = utils.f1):
    
    model = models.Sequential()
    if train_X is None and input_shape is None:
        model.add(Embedding(19498,32,input_length=51))
        model.add(layers.SimpleRNN(50))
    elif input_shape is not None:
        model.add(layers.SimpleRNN(50, input_shape=input_shape))
    else:
        model.add(layers.SimpleRNN(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics, 'binary_accuracy', 'accuracy'])
    
    return model


def keras_lstm(train_X=None, input_shape=None, optimizer="rmsprop",
               loss="binary_crossentropy", metrics = utils.f1):
    # define model
    lstm_model = models.Sequential()
    if train_X is None and input_shape is None:
        lstm_model.add(Embedding(19498,32,input_length=51))
        lstm_model.add(layers.LSTM(50))
    elif input_shape is not None:
        lstm_model.add(layers.LSTM(50, input_shape=input_shape))
    else:
        lstm_model.add(layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))

    lstm_model.add(layers.Dense(1))

    lstm_model.compile(optimizer= optimizer,
                  loss = loss,
                  metrics=[metrics, 'binary_accuracy', 'accuracy'])

    return lstm_model