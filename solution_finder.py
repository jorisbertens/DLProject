import os
import datetime
import multiprocessing
import itertools

from sklearn.model_selection import train_test_split
import utils
import warnings
warnings.filterwarnings("ignore")

from ML_algorithms import *


file_name= "log_files/" + "results_"+ str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) +"_log.csv"

header_string = "Seed,Algorithm,dataset,time,train_f1,val_f1,acc,val_acc,bin_acc,val_bin_acc,train_loss,val_loss"
with open(file_name, "w") as myfile:
    myfile.write(header_string + "\n")

datasets_to_run = ['Easy', 'Big', 'Text', 'Image','TimeSeries']
#models_to_run = ['Shallow', 'Deep', 'LSTM', 'RNN', 'CNN']
models_to_run = ['CNN']


models = [
          ######################### Easy ####################################
         ("Shallow", "Easy", 'X_train, X_test, y_train, y_test  = utils.get_titanic_dataset() \n'
                          'model = keras_shallow(input_dim=len(X_train.columns)) \n'
                          'model_result = model.fit( X_train, y_train, epochs=50, batch_size=32,validation_data=(X_test, y_test)) \n'),
         ("Deep", "Easy", 'X_train, X_test, y_train, y_test = utils.get_titanic_dataset() \n'
                          'model = keras_deep(input_dim=len(X_train.columns)) \n'
                          'model_result = model.fit( X_train, y_train, epochs=50, batch_size=32,validation_data=(X_test, y_test)) \n'),
         ("LSTM", "Easy", 'X_train, X_test, y_train, y_test = utils.get_titanic_dataset(cnn_or_lstm=True) \n'
                          'model = keras_lstm(X_train) \n'
                          'model_result = model.fit( X_train, y_train, epochs=50, batch_size=32,validation_data=(X_test, y_test)) \n'),
         ("RNN", "Easy", 'X_train, X_test, y_train, y_test = utils.get_titanic_dataset(cnn_or_lstm=True) \n'
                          'model = simple_rnn(X_train) \n'
                          'model_result = model.fit( X_train, y_train, epochs=50, batch_size=32,validation_data=(X_test, y_test)) \n'),
         ("CNN", "Easy", 'X_train, X_test, y_train, y_test = utils.get_titanic_dataset(cnn_conv2d=True) \n'
                          'model = keras_cnn(filter_size=(1,1), input_shape=(1, 36, 1), max_pooling=(1,1)) \n'
                          'model_result = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test)) \n'),

         ######################### BIG ####################################
         ("Shallow", "Big", 'X_train, X_test, y_train, y_test  = utils.get_bank_dataset() \n'
                         'model = keras_shallow(input_dim=len(X_train.columns)) \n'
                         'model_result = model.fit( X_train, y_train, epochs=50, batch_size=512,validation_data=(X_test, y_test)) \n'),
         ("Deep", "Big", 'X_train, X_test, y_train, y_test = utils.get_bank_dataset() \n'
                         'model = keras_deep(input_dim=len(X_train.columns)) \n'
                         'model_result = model.fit(X_train, y_train, epochs=50, batch_size=512, validation_data=(X_test, y_test)) \n'),
         ("CNN", "Big", 'X_train, X_test, y_train, y_test = utils.get_bank_dataset(cnn_conv2d=True) \n'
                         'model = keras_cnn(filter_size=(1,1), input_shape=(1, 29, 1), max_pooling=(1,1)) \n'
                         'model_result = model.fit(X_train, y_train, epochs=50, batch_size=512, validation_data=(X_test, y_test)) \n'),
         ("LSTM", "Big", 'X_train, X_test, y_train, y_test = utils.get_bank_dataset(cnn_or_lstm=True) \n'
                        'model = keras_lstm(X_train, optimizer="rmsprop",loss="binary_crossentropy") \n'
                        'model_result = model.fit(X_train, y_train, epochs=50, batch_size=512, validation_data=(X_test, y_test), verbose=0, shuffle=False) \n'),
         ("RNN", "Big", 'X_train, X_test, y_train, y_test = utils.get_bank_dataset(cnn_or_lstm=True) \n'
                        'model = simple_rnn(X_train, optimizer="rmsprop",loss="binary_crossentropy") \n'
                        'model_result = model.fit(X_train, y_train, epochs=50, batch_size=512, validation_data=(X_test, y_test), verbose=0, shuffle=False) \n'),

         ######################### Text ####################################
         ("Shallow", "Text", 'X_train, X_test, y_train, y_test  = utils.get_text_dataset() \n'
                         'model = keras_shallow(input_dim=len(X_train.columns)) \n'
                         'model_result = model.fit(X_train, y_train, epochs=15, batch_size=512,validation_data=(X_test, y_test)) \n'),
         ("Deep", "Text", 'X_train, X_test, y_train, y_test  = utils.get_text_dataset() \n'
                         'model = keras_deep(input_dim=len(X_train.columns)) \n'
                         'model_result = model.fit(X_train, y_train, epochs=15, batch_size=512, validation_data=(X_test, y_test)) \n'),
         ("LSTM", "Text", 'X_train, X_test, y_train, y_test  = utils.get_text_dataset() \n'
                         'model = keras_lstm() \n'
                         'model_result = model.fit(X_train, y_train, epochs=15, batch_size=512, validation_data=(X_test, y_test)) \n'),
         ("RNN", "Text", 'X_train, X_test, y_train, y_test  = utils.get_text_dataset() \n'
                         'model = simple_rnn() \n'
                         'model_result = model.fit(X_train, y_train, epochs=15, batch_size=512, validation_data=(X_test, y_test)) \n'),
         ("CNN", "Text", 'X_train, X_test, y_train, y_test = utils.get_text_dataset(cnn_conv2d=True) \n'
                        'model = keras_cnn(filter_size=(1,1), input_shape=(1, 51, 1), max_pooling=(1,1)) \n'
                        'model_result = model.fit(X_train, y_train, epochs=15, batch_size=512, verbose=1, validation_data=(X_test, y_test)) \n'),
    ######################### Image ####################################
         ("Shallow", "Image", 'train_generator, test_generator  = utils.get_image_dataset(matrix_output=False) \n'
                         'model = keras_shallow(input_dim=64*64*3) \n'
                         'model_result = model.fit_generator( train_generator,epochs=5,validation_data=test_generator,steps_per_epoch=30,validation_steps=50) \n'),
         ("Deep", "Image", 'train_generator, test_generator  = utils.get_image_dataset(matrix_output=False) \n'
                         'model = keras_deep(input_dim=64*64*3) \n'
                         'model_result = model.fit_generator( train_generator,epochs=5,validation_data=test_generator,steps_per_epoch=30,validation_steps=50) \n'),
         ("CNN", "Image", 'train_generator, test_generator  = utils.get_image_dataset() \n'
                         'model = keras_cnn() \n'
                         'model_result = model.fit_generator( train_generator,steps_per_epoch=30,epochs=5,validation_data=test_generator,    validation_steps=50) \n'),
         ("LSTM", "Image", 'train_generator, test_generator  = utils.get_image_dataset(matrix_output=False, rnn_output=True) \n'
                         'model = keras_lstm(input_shape=(1, 12288), optimizer="rmsprop",loss="binary_crossentropy") \n'
                         'model_result = model.fit_generator(train_generator, steps_per_epoch=30,epochs=5,validation_data=test_generator, validation_steps=50) \n'),
         ("RNN", "Image", 'train_generator, test_generator  = utils.get_image_dataset(matrix_output=False, rnn_output=True) \n'
                         'model = simple_rnn(input_shape=(1, 12288), optimizer="rmsprop",loss="binary_crossentropy") \n'
                         'model_result = model.fit_generator(train_generator, steps_per_epoch=30,epochs=5,validation_data=test_generator, validation_steps=50) \n'),

    ######################### TimeSeries ####################################
         ("Shallow", "TimeSeries", 'X_train, X_test, y_train, y_test  = utils.get_timeseries_dataset() \n'
                                'model = keras_shallow(input_dim=len(X_train.columns)) \n'
                                'model_result = model.fit( X_train, y_train, epochs=20, batch_size=512,validation_data=(X_test, y_test), shuffle=False) \n'),
         ("Deep", "TimeSeries", 'X_train, X_test, y_train, y_test  = utils.get_timeseries_dataset() \n'
                                'model = keras_deep(input_dim=len(X_train.columns)) \n'
                                'model_result = model.fit( X_train, y_train, epochs=20, batch_size=512,validation_data=(X_test, y_test), shuffle=False) \n'),
         ("CNN", "TimeSeries", 'X_train, X_test, y_train, y_test = utils.get_timeseries_dataset(cnn_or_lstm=True) \n'
                                'model = keras_cnn_conv1D() \n'
                                'model_result = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test), shuffle=False) \n'),
         ("LSTM", "TimeSeries", 'X_train, X_test, y_train, y_test  = utils.get_timeseries_dataset(cnn_or_lstm=True) \n'
                                'model = keras_lstm(X_train) \n'
                                'model_result = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test), shuffle=False) \n'),
         ("RNN", "TimeSeries", 'X_train, X_test, y_train, y_test  = utils.get_timeseries_dataset(cnn_or_lstm=True) \n'
                                'model = simple_rnn(X_train) \n'
                                'model_result = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test), shuffle=False) \n')

]


seed = list(range(120,1500))

def algo_run(seed,model):

    if not (model[0] in models_to_run and model[1] in datasets_to_run):
        print("Skipped "+ model[0]+" "+model[1])
    else:
        reset_seed(seed)
        start_time = datetime.datetime.now()

        _locals = locals()
        exec(model[2], globals(), _locals)
        model_result = _locals['model_result']

        # visualizing losses and accuracy
        train_loss = model_result.history['loss']
        val_loss = model_result.history['val_loss']
        train_f1 = model_result.history['f1']
        val_f1 = model_result.history['val_f1']
        train_acc = model_result.history['acc']
        val_acc = model_result.history['val_acc']
        train_bin_acc = model_result.history['binary_accuracy']
        val_bin_acc = model_result.history['binary_accuracy']


        time_elapsed = datetime.datetime.now() - start_time
        # Create result string
        log_parameters = [seed,model[0],model[1], time_elapsed, train_f1[-1],val_f1[-1],train_acc[-1],val_acc[-1],train_bin_acc[-1],val_bin_acc[-1],train_loss[-1],val_loss[-1] ]

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
    possible_values = list(itertools.product(*[seed,models]))



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
