import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk import word_tokenize
import codecs
import keras
import re
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import hstack


################################### New methods #############################################
def get_titanic_dataset(cnn_or_lstm=False, cnn_conv2d=False):
    '''
        Returns the dataset provided for the project as a dataframe
    '''
    df = pd.read_csv("data_files/Easy/titanic.csv")
    df = df.rename({'survived': 'y'}, axis=1)
    # Drop ticket, home.dest, boat, body
    df = df.drop(['ticket'], axis=1)
    df = df.drop(['home.dest'], axis=1)
    df = df.drop(['boat'], axis=1)
    df = df.drop(['body'], axis=1)
    
    # Deal with cabin missing values
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    df['cabin'] = df['cabin'].fillna("U0")
    df['deck'] = df['cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    df['deck'] = df['deck'].map(deck)
    df['deck'] = df['deck'].fillna(0)
    df['deck'] = df['deck'].astype(int)
    # we can now drop the cabin feature
    df = df.drop(['cabin'], axis=1)
    
    # Deal with Age missing values
    mean = df["age"].mean()
    std = df["age"].std()
    is_null = df["age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = df["age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    df["age"] = age_slice
    df["age"] = df["age"].astype(int)
    
    # Deal with Embarked missing values
    common_value = 'S'
    df['embarked'] = df['embarked'].fillna(common_value)
    # Converting Fares
    df['fare'] = df['fare'].fillna(0)
    df['fare'] = df['fare'].astype(int)
    
    # Converting Names
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    # extract titles
    df['title'] = df.name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    df['title'] = df['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                                'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['title'] = df['title'].replace('Mlle', 'Miss')
    df['title'] = df['title'].replace('Ms', 'Miss')
    df['title'] = df['title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    df['title'] = df['title'].map(titles)
    # filling NaN with 0, to get safe
    df['title'] = df['title'].fillna(0)
    df = df.drop(['name'], axis=1)
    
    # Converting Sex
    genders = {"male": 0, "female": 1}
    df['sex'] = df['sex'].map(genders)
    # Converting Embarked
    ports = {"S": 0, "C": 1, "Q": 2}
    df['embarked'] = df['embarked'].map(ports)
    # Rearrange columns
    df = df[['pclass', "sex","age","sibsp","parch","fare", "embarked", "deck", "title", "y"]]
    
    # Encoding
    onehotencoder = OneHotEncoder(categorical_features = [0, 3, 4, 7, 8])
    df = onehotencoder.fit_transform(df).toarray()
    df = pd.DataFrame(df)
    df = df.rename(columns = {36:"y"})
    
    # Create train, test set
    y = df.y
    X = df.drop("y", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if cnn_or_lstm == True:
        X_train = X_train.as_matrix().reshape((len(X_train), 36))
        y_train = y_train.as_matrix().reshape((len(y_train), 1))
        X_test = X_test.as_matrix().reshape((len(X_test), 36))
        y_test = y_test.as_matrix().reshape((len(y_test), 1))
        
        train_dataset=hstack((X_train,y_train))
        test_dataset=hstack((X_test,y_test)) 
        
        X_train, y_train = split_sequences(train_dataset, 1)
        X_test, y_test = split_sequences(test_dataset, 1)
     
    if cnn_conv2d == True:
        
        X_train = X_train.values.reshape(X_train.values.shape[0], 1, X_train.values.shape[1], 1)
        X_test = X_test.values.reshape(X_test.values.shape[0], 1, X_test.values.shape[1], 1)

    return X_train, X_test, y_train, y_test



def get_bank_dataset(cnn_or_lstm=False, cnn_conv2d=False):
    '''
        Returns the dataset provided for the project as a dataframe
    '''
    df = pd.read_csv("data_files/Big/creditcard.csv")
    df = df.rename({'Class': 'y'}, axis=1)
    #df = df.drop(["ID_code"], axis=1)
    
    # Get train and test set
    y = df.y
    X = df.drop(["y"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    if cnn_or_lstm == True:
        X_train = X_train.as_matrix().reshape((len(X_train), 29))
        y_train = y_train.as_matrix().reshape((len(y_train), 1))
        X_test = X_test.as_matrix().reshape((len(X_test), 29))
        y_test = y_test.as_matrix().reshape((len(y_test), 1))
        
        train_dataset=hstack((X_train,y_train))
        test_dataset=hstack((X_test,y_test)) 
        
        X_train, y_train = split_sequences(train_dataset, 1)
        X_test, y_test = split_sequences(test_dataset, 1)
     
    if cnn_conv2d == True:
        
        X_train = X_train.values.reshape(X_train.values.shape[0], 1, X_train.values.shape[1], 1)
        X_test = X_test.values.reshape(X_test.values.shape[0], 1, X_test.values.shape[1], 1)    
    
    
    return X_train, X_test, y_train, y_test


def get_timeseries_dataset(city = "Perth", cnn_or_lstm=False):
    '''
        Returns the dataset provided for the project as a dataframe
    '''
    df = pd.read_csv("data_files/TimeSeries/weatherAUS.csv")

    # only take the data of one city
    df = df.loc[df["Location"] == city, :]

    # drop columns with too many missing values, Location & Wind Directions as well (to not produce so many new columns when encoding)
    df = df.drop(["Evaporation","Sunshine","Cloud9am","Cloud3pm","WindGustDir","WindDir9am", "Location", 'WindDir3pm'], axis=1)

    # Get binary variables to 1 and 0
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    
    # impute missing values
    df["Temp3pm"] = df["Temp3pm"].fillna(df["Temp3pm"].mean())
    df["Pressure3pm"] = df["Pressure3pm"].fillna(df["Pressure3pm"].mean())
    df["Temp9am"] = df["Temp9am"].fillna(df["Temp9am"].mean())
    df["Pressure9am"] = df["Pressure3pm"].fillna(df["Pressure3pm"].mean())
    df["Humidity9am"] = df["Humidity9am"].fillna(df["Humidity9am"].mean())
    df["Humidity3pm"] = df["Humidity3pm"].fillna(df["Humidity3pm"].mean())
    df["MinTemp"] = df["MinTemp"].fillna(df["MinTemp"].mean())
    df["MaxTemp"] = df["MaxTemp"].fillna(df["MaxTemp"].mean())
    df["WindGustSpeed"] = df["WindGustSpeed"].fillna(df["WindGustSpeed"].mean())
    df["WindSpeed9am"] = df["WindSpeed9am"].fillna(df["WindSpeed9am"].mean())
    df["WindSpeed3pm"] = df["WindSpeed3pm"].fillna(df["WindSpeed3pm"].mean())
    
    # drop rest of the rows (e.g. rainfall should only be there if rain today is a 1. Would require kind of dependent imputation)
    df = df[pd.notnull(df['RainToday'])]
    df = df[pd.notnull(df['Rainfall'])]
    
    # convert date to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # convert date to days since start date (measures distance between minimum date)
    # This way at least some information of time can be kept, even though it might not be the perfect solution
    # Got this idea from here:
    # https://stackoverflow.com/questions/42044003/how-to-use-date-and-time-values-as-features-to-predict-a-value-using-a-neural-ne
    start = min(df["Date"])
    date_features = [(i - start) for i in df["Date"]]
    date_features = [i.days for i in date_features]
    
    # Replace date with days
    df["Date"] = date_features
    
    # Get train and test set
    y = df.RainTomorrow
    X = df.drop(["RainTomorrow"], axis=1)

    if cnn_or_lstm:
        
        X_train, y_train = X[:int(len(X)*0.7)], y[:int(len(y)*0.7)]
        X_test, y_test = X[int(len(X)*0.7):], y[int(len(y)*0.7):]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    X_train, X_test = Min_Max_Train(X_train, X_test)
   

    if cnn_or_lstm == True:
        X_train = X_train.as_matrix().reshape((len(X_train), 15))
        y_train = y_train.as_matrix().reshape((len(y_train), 1))
        X_test = X_test.as_matrix().reshape((len(X_test), 15))
        y_test = y_test.as_matrix().reshape((len(y_test), 1))
        
        train_dataset=hstack((X_train,y_train))
        test_dataset=hstack((X_test,y_test)) 
        
        X_train, y_train = split_sequences(train_dataset, 3)
        X_test, y_test = split_sequences(test_dataset, 3)

    return X_train, X_test, y_train, y_test

def Min_Max_Train(X_train, X_test):
    scaler = MinMaxScaler()
    # Only fit the training data
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train, X_test
    
def get_text_dataset(cnn_or_lstm=False, cnn_conv2d=False):
    '''
        Returns the dataset provided for the project as a dataframe
    '''
    import codecs
    pos_lines = codecs.open("data_files/Text/positive.txt", "r", encoding="latin2").read()
    neg_lines = codecs.open("data_files/Text/negative.txt", "r", encoding="latin2").read()

    all_words = []
    documents = []

    allowed_word_types = ["J"]

    for p in pos_lines.split("\n"):
        documents.append((p, "pos"))

    for p in neg_lines.split("\n"):
        documents.append((p, "neg"))

    # Get list of lists in dataframe
    headers = ["text", "y"]
    df = pd.DataFrame(documents, columns=headers)
    df["y"] = df["y"].map({"pos": 1, "neg": 0})

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    max_fatures = 9999
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(df['text'].values)
    X = tokenizer.texts_to_sequences(df['text'].values)
    X = pad_sequences(X)
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X), df['y'], test_size=0.3, shuffle=True)
    
    if cnn_or_lstm == True:
        X_train = X_train.as_matrix().reshape((len(X_train), 51))
        y_train = y_train.as_matrix().reshape((len(y_train), 1))
        X_test = X_test.as_matrix().reshape((len(X_test), 51))
        y_test = y_test.as_matrix().reshape((len(y_test), 1))
        
        train_dataset=hstack((X_train,y_train))
        test_dataset=hstack((X_test,y_test)) 
        
        X_train, y_train = split_sequences(train_dataset, 1)
        X_test, y_test = split_sequences(test_dataset, 1)
     
    if cnn_conv2d == True:
        
        X_train = X_train.values.reshape(X_train.values.shape[0], 1, X_train.values.shape[1], 1)
        X_test = X_test.values.reshape(X_test.values.shape[0], 1, X_test.values.shape[1], 1)

    return X_train, X_test, y_train, y_test

def get_image_dataset_old():
    train_dir = 'data_files/Cactus_Image/training_set'
    test_dir = "data_files/Cactus_Image/testing_set"
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=20,
        class_mode="binary")

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        batch_size=20,
        class_mode="binary")
    return train_generator, test_generator


def get_image_dataset(batch_size=20, matrix_output=True, rnn_output=False):
    # import the necessary packages
    from sklearn.preprocessing import LabelBinarizer
    import numpy as np

    def csv_image_generator(inputPath, bs, lb):
        # open the CSV file for reading
        f = open(inputPath, "r")

        # loop indefinitely
        while True:
            # initialize our batches of images and labels
            images = []
            labels = []

            # keep looping until we reach our batch size
            while len(images) < bs:
                # attempt to read the next line of the CSV file
                line = f.readline()

                # check to see if the line is empty, indicating we have
                # reached the end of the file
                if line == "":
                    # reset the file pointer to the beginning of the file
                    # and re-read the line
                    f.seek(0)
                    line = f.readline()


                # extract the label and construct the image
                line = line.strip().split(",")
                label = line[0]
                image = np.array([int(x)/255 for x in line[1:]])
                if rnn_output:
                    image = image.reshape((1, len(image)))
                if matrix_output:
                    image = image.reshape((64, 64, 3))

                # update our corresponding batches lists
                images.append(image)
                labels.append(label)

            # one-hot encode the labels
            labels = lb.transform(np.array(labels))

            # yield the batch to the calling function
            yield (np.array(images), labels)

    # initialize the paths to our training and testing CSV files
    TRAIN_CSV = "data_files/Cactus_Image/cactus_training.csv"
    TEST_CSV = "data_files/Cactus_Image/cactus_testing.csv"

    # initialize the total number of training and testing image
    NUM_TRAIN_IMAGES = 0
    NUM_TEST_IMAGES = 0

    # open the training CSV file, then initialize the unique set of class
    # labels in the dataset along with the testing labels
    labels = set()
    testLabels = []
    with open(TRAIN_CSV, "r") as f:
        # loop over all rows of the CSV file
        for line in f:
            # extract the class label, update the labels list, and increment
            # the total number of training images
            label = line.strip().split(",")[0]
            labels.add(label)
            NUM_TRAIN_IMAGES += 1

    with open(TEST_CSV, "r") as f:

        # loop over the lines in the testing file
        for line in f:
            # extract the class label, update the test labels list, and
            # increment the total number of testing images
            label = line.strip().split(",")[0]
            testLabels.append(label)
            NUM_TEST_IMAGES += 1


    # create the label binarizer for one-hot encoding labels, then encode
    # the testing labels
    lb = LabelBinarizer()
    lb.fit(list(labels))

    # initialize both the training and testing image generators
    train_generator = csv_image_generator(TRAIN_CSV, batch_size, lb)
    test_generator = csv_image_generator(TEST_CSV, batch_size, lb)

    return train_generator, test_generator


    

def split_data_scale(df, test_size, val_size, target_column, exception_columns, random_state):
    """This function takes a dataframe and performs the train, test, validation split
    including scaling the data using the MinMaxScaler

    Inputs:
    df = pandas dataframe
    test_size = size of the test set
    val_size = size of the validation set
    target_column = column name of the y column
    exception_columns = columns that shouldn't be scaled in a list
    random_state = random_state

    Returns:
    X_train, X_test, X_val, y_train, y_test, y_val

    """

    idx = df.shape[0] - 1  # indices start with 0
    test_size = int(df.shape[0] * test_size)
    val_size = int(df.shape[0] * val_size)
    train_size = df.shape[0] - test_size - val_size

    train_idx = random_state.choice(idx, train_size, replace=False)

    test_idx = random_state.choice([index for index in range(idx)
                                    if index not in train_idx],
                                   test_size,
                                   replace=False)

    # leftovers go to the validation set
    val_idx = [index for index in range(idx) if index not in train_idx and index not in test_idx]

    # splitting the datasets
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    val = df.iloc[val_idx]

    # scaling
    scaler = MinMaxScaler()

    scaling_columns = [column for column in df.columns if column not in exception_columns]

    train_scale = train.loc[:, scaling_columns]
    test_scale = test.loc[:, scaling_columns]
    val_scale = val.loc[:, scaling_columns]

    scaler.fit(train_scale)  # fitting the scaler only to the train data

    # transform the data
    train_scale = pd.DataFrame(scaler.transform(train_scale), index=train_scale.index, columns=train_scale.columns)
    test_scale = pd.DataFrame(scaler.transform(test_scale), index=test_scale.index, columns=test_scale.columns)
    val_scale = pd.DataFrame(scaler.transform(val_scale), index=val_scale.index, columns=val_scale.columns)

    # overwrite with scaled data
    train.loc[:, scaling_columns] = train_scale
    test.loc[:, scaling_columns] = test_scale
    val.loc[:, scaling_columns] = val_scale

    # splitting X and y
    X_train, y_train = train.loc[:, train.columns != target_column], train[target_column]
    X_test, y_test = test.loc[:, test.columns != target_column], test[target_column]
    X_val, y_val = val.loc[:, val.columns != target_column], val[target_column]

    return X_train, X_test, X_val, y_train, y_test, y_val

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)



from keras import backend as K
#https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

