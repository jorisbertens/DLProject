import os
import datetime
import multiprocessing
import itertools

import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import utils
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import *
import feature_engineering
from ML_algorithms import *


file_name= "log_files/" + "results_"+ str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) +"_log.csv"

header_string = "Seed,Algorithm,dataset,time,train_f1,val_f1,train_loss,val_loss"
with open(file_name, "w") as myfile:
    myfile.write(header_string + "\n")

datasets_to_run = ['Easy', 'Big', 'Text', 'Image','TimeSeries']
models_to_run = ['Shallow', 'Deep', 'LSTM', 'RNN', 'CNN']

datasets_to_run = ['Text']
models_to_run = ['RNN']


models = [
          ######################### Easy ####################################
         ("Shallow", "Easy", 'X_train, X_test, y_train, y_test  = utils.get_titanic_dataset() \n'
                        'model = keras_shallow(input_dim=len(X_train.columns)) \n'
                        'model_result = model.fit( X_train, y_train, epochs=5, batch_size=30,validation_data=(X_test, y_test)) \n'),
         ("Deep", "Easy", 'X_train, X_test, y_train, y_test = utils.get_titanic_dataset() \n'
                              'model = keras_deep(input_dim=len(X_train.columns)) \n'
                              'model_result = model.fit( X_train, y_train, epochs=5, batch_size=30,validation_data=(X_test, y_test)) \n'),

         ######################### BIG ####################################
         ("Shallow", "Big", 'X_train, X_test, y_train, y_test  = utils.get_bank_dataset() \n'
                       'model = keras_shallow(input_dim=len(X_train.columns)) \n'
                       'model_result = model.fit( X_train, y_train, epochs=20, batch_size=512,validation_data=(X_test, y_test)) \n'),
         ("Deep", "Big", 'X_train, X_test, y_train, y_test = utils.get_bank_dataset() \n'
                    'model = keras_deep(input_dim=len(X_train.columns)) \n'
                    'model_result = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test)) \n'),

         ######################### Text ####################################
         ("Shallow", "Text", 'X_train, X_test, y_train, y_test  = utils.get_text_dataset() \n'
                         'model = keras_shallow(input_dim=len(X_train.columns)) \n'
                         'model_result = model.fit( X_train, y_train, epochs=20, batch_size=512,validation_data=(X_test, y_test)) \n'),
         ("Deep", "Text", 'X_train, X_test, y_train, y_test  = utils.get_text_dataset() \n'
                        'model = keras_deep(input_dim=len(X_train.columns)) \n'
                        'model_result = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test)) \n'),
         ("LSTM", "Text", 'X_train, X_test, y_train, y_test  = utils.get_text_dataset() \n'
                     'model = keras_lstm() \n'
                     'model_result = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test)) \n'),
         ("RNN", "Text", 'X_train, X_test, y_train, y_test  = utils.get_text_dataset() \n'
                     'model = simple_rnn() \n'
                     'model_result = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test)) \n'),

    ######################### Image ####################################
         ("Shallow", "Image", 'train_generator, test_generator  = utils.get_image_dataset(matrix_output=False) \n'
                        'model = keras_shallow(input_dim=64*64*3) \n'
                        'model_result = model.fit_generator( train_generator,steps_per_epoch=30,epochs=5,validation_data=test_generator,    validation_steps=50) \n'),
         ("Deep", "Image", 'train_generator, test_generator  = utils.get_image_dataset(matrix_output=False) \n'
                        'model = keras_deep(input_dim=64*64*3) \n'
                        'model_result = model.fit_generator( train_generator,steps_per_epoch=30,epochs=5,validation_data=test_generator,    validation_steps=50) \n'),
         ("CNN", "Image", 'train_generator, test_generator  = utils.get_image_dataset() \n'
                     'model = keras_cnn() \n'
                     'model_result = model.fit_generator( train_generator,steps_per_epoch=30,epochs=5,validation_data=test_generator,    validation_steps=50) \n'),

         ######################### TimeSeries ####################################
         ("Shallow", "TimeSeries", 'X_train, X_test, y_train, y_test  = utils.get_timeseries_dataset() \n'
                              'model = keras_shallow(input_dim=len(X_train.columns)) \n'
                              'model_result = model.fit( X_train, y_train, epochs=20, batch_size=512,validation_data=(X_test, y_test)) \n'),
         ("Deep", "TimeSeries", 'X_train, X_test, y_train, y_test  = utils.get_timeseries_dataset() \n'
                           'model = keras_deep(input_dim=len(X_train.columns)) \n'
                           'model_result = model.fit( X_train, y_train, epochs=20, batch_size=512,validation_data=(X_test, y_test)) \n'),
         ("CNN", "TimeSeries", 'X_train, X_test, y_train, y_test = utils.get_timeseries_dataset(cnn_or_lstm=True) \n'
                        'model = keras_cnn_conv1D() \n'
                        'model_result = model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test)) \n'),
         ("LSTM", "TimeSeries", 'X_train, X_test, y_train, y_test  = utils.get_timeseries_dataset() \n'
                        'model = keras_lstm(X_train) \n'
                        'model_result = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=1, shuffle=False) \n')
]


#seed = list(range(0,1))
seed = [0]

def algo_run(seed, model):

    if not (model[0] in models_to_run and model[1] in datasets_to_run):
        print("Skipped "+ model[0]+" "+model[1])
    else:
        reset_seed(seed)
        start_time = datetime.datetime.now()

        _locals = locals()
        exec(model[2], globals(),_locals)
        model_result = _locals['model_result']

        # visualizing losses and accuracy
        train_loss = model_result.history['loss']
        val_loss = model_result.history['val_loss']
        train_acc = model_result.history['f1']
        val_acc = model_result.history['val_f1']


        time_elapsed = datetime.datetime.now() - start_time
        # Create result string
        log_parameters = [seed,model[0],model[1], time_elapsed, train_acc[-1],val_acc[-1],train_loss[-1],val_loss[-1]]

        result_string = ",".join([str(value) for value in log_parameters])
        # Write result to a file
        with open(file_name, "a") as myfile:
            myfile.write(result_string + "\n")

        print(result_string)



def reset_seed(seed):
    # https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras

    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = seed

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


if __name__ ==  '__main__':
    possible_values = list(itertools.product(*[seed, models]))

    core_count = multiprocessing.cpu_count()
    #print("All possible combinations generated:")
    #print(possible_values)
    print(len(possible_values))
    print("Number of cpu cores: "+str(core_count))
    print()
    print(header_string)

    ####### Magic appens here ########
    # Neural networks are already parallel
    pool = multiprocessing.Pool(1)
    results = pool.starmap(algo_run, possible_values)
