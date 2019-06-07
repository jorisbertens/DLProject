from keras import models
from keras import layers
import numpy as np
import utils

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



from keras.layers import SimpleRNN, Embedding

def simple_rnn(max_features, n_neurons=32, activation="sigmoid", optimizer="rmsprop",
               loss="binary_crossentropy", metrics=["acc"]):
    
    model = models.Sequential()
    model.add(Embedding(max_features, n_neurons))
    model.add(SimpleRNN(n_neurons))
    model.add(models.Dense(1, activation=activation))

    model.compile(optimizer=optimizer, loss=los, metrics=metrics)
    
    return model


def keras_lstm(train_X, optimizer="rmsprop",
               loss="binary_crossentropy", metrics = 'acc'):
    # define model
    lstm_model = models.Sequential()
    lstm_model.add(layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    lstm_model.add(layers.Dense(1))
    lstm_model.compile(loss='mae', optimizer='adam')

    lstm_model.compile(optimizer= optimizer,
                  loss = loss,
                  metrics=[metrics])

    return lstm_model