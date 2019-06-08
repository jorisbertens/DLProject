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



def get_bank_dataset():
    '''
        Returns the dataset provided for the project as a dataframe
    '''
    df = pd.read_csv("data_files/Big/titanic.csv")
    df = df.rename({'target': 'y'}, axis=1)
    df = df.drop(["ID_code"], axis=1)
    
    # Get train and test set
    y = df.y
    X = df.drop(["y"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
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
        
        X_train, y_train = split_sequences(train_dataset, 1)
        X_test, y_test = split_sequences(test_dataset, 1)

    return X_train, X_test, y_train, y_test, df

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


def get_image_dataset(batch_size=20, matrix_output=True):
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

################################### Old methods #############################################
def get_dataset():
    '''
        Returns the dataset provided for the project as a dataframe
    '''
    return pd.read_excel("data_files/ml_project1_data.xlsx")

def missing_values_reporter(df):
    '''
        Returns the number of missing values in each columns
        Credit to Professor Ilya
    '''
    na_count = df.isna().sum() 
    ser = na_count[na_count > 0]
    return pd.DataFrame({"N missings": ser, "% missings": ser.divide(df.shape[0])})

def data_split(df, test_size=0.33,random_state=42):
    '''
        Selects the outcome variable and calls sklearn trai_test_split
    '''
    y = df["Response"]
    X = df.loc[:, df.columns != "Response"] 
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def simple_train_split(df, test_size=0.33, random_state=42):
    test = df.sample(frac=test_size, random_state=random_state)
    train = df.drop(test.index)
    return train, test

def X_y_split(df):
    '''
        Selects the outcome variable and calls sklearn trai_test_split
    '''
    y = df["Response"]
    X = df.loc[:, df.columns != "Response"] 
    return (X, y)

def calculate_accuracy(y_true, y_pred):
    '''
        Accuracy classification score.

        In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    '''
    return accuracy_score(y_true, y_pred)


def calculate_auc(y_true, y_pred):
    '''
        Compute Area Under the Curve (AUC) using the trapezoidal rule

        This is a general function, given points on a curve. For computing the area under the ROC-curve, see roc_auc_score
    '''
    return roc_auc_score(y_true, y_pred)

def calculate_average_precision_score(y_true, y_pred):
    '''
        Compute average precision (AP) from prediction scores

        AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight
    '''
    return average_precision_score(y_true, y_pred)

def calculate_precision_score(y_true, y_pred):
    '''
       Compute the precision

    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

    The best value is 1 and the worst value is 0.
    '''
    return precision_score(y_true, y_pred)

def calculate_recall_score(y_true, y_pred):
    '''
        Compute the recall

        The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

        The best value is 1 and the worst value is 0.
    '''
    return recall_score(y_true, y_pred)

def calculate_confusion_matrix(y_true, y_pred):
    '''
        Compute confusion matrix to evaluate the accuracy of a classification

        By definition a confusion matrix
        is such that is equal to the number of observations known to be in group but predicted to be in group
    '''
    return confusion_matrix(y_true, y_pred)

def cross_validation_average_results(model, X, y, n_splits=5, sampling_technique=None, scaler=None, **model_kwargs):
    '''
        Does cross validation with n_splits and returns an array with y size as predictions.
        !!!!Currently not working with transformations calculated on train data and applied in test data!!!
        
        example with 5 splits:
        
        split 1 -   |||------------
        split 2 -   ---|||---------
        split 3 -   ------|||------
        split 4 -   ---------|||---
        split 5 -   ------------|||
        
        returns     |||||||||||||||  <- which represents the predictions for the whole array
        
    '''
    kf = KFold(n_splits=n_splits, shuffle=False)
    predictions = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, _ = y.iloc[train_index], y.iloc[test_index]
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        if sampling_technique is not None:
            X_train, y_train = sampling_technique.fit_resample(X_train, y_train)            
        if type(model) == keras.engine.sequential.Sequential:
            model.fit(X_train, y_train, epochs=100, verbose=0)
        else:
            model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        predictions.extend(prediction)
    return np.array(predictions)

def leave_one_out_cross_validation_average_results(model, X, y, n_splits=5, scaler=None, sampling_technique=None, **model_kwargs):
    '''
        Does cross validation with n_splits and returns an array with y size as predictions.
        !!!!Currently not working with transformations calculated on train data and applied in test data!!!
        
        example with 5 splits:
        
        split 1 -   |--------------
        split 2 -   -|-------------
        split 3 -   --|------------
                  ...
        split n-1 - -------------|-
        split n -   --------------|
        
        returns     |||||||||||||||  <- which represents the predictions for the whole array
        
    '''
    kf = LeaveOneOut()
    predictions = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, _ = y.iloc[train_index], y.iloc[test_index]
        
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        if sampling_technique is not None:
            X_train, y_train = sampling_technique.fit_resample(X_train, y_train)            
        
        if type(model) == keras.engine.sequential.Sequential:
            model.fit(X_train, y_train, epochs=100, verbose=0)
        else:
            model.fit(X_train, y_train)
            
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        predictions.extend(prediction)
    return np.array(predictions)

def profit_share(y_true, y_pred):
    """
    Computes the profit. For each True/True +8, for each True/False -3 and compares it with the possible profit.
    E.g. 0.26 means, that one got 26% of the max possible profit.
    """
    score = 0
    for i in (y_true - (y_pred * 2)):
        if i == -1:
            score += 8
        elif i == -2:
            score -= 3
    
    if sum(y_true) == 0:
        return 0.00

    return round(score / (sum(y_true) * 8), 3)

def max_threshold(y_pred, y_test, threshold_range = (0.4, 0.6), iterations = 100, visualization=False):
    '''
        For a given continuos predictions array with value [0,1] returns the best threshold to use when categorizint the data
    '''
    profits, thresholds = threshold_optimization(y_pred, y_test, threshold_range, iterations, visualization)
    profits = np.array(profits)
    thresholds = np.array(thresholds)
    if visualization:
        data_visualization.arg_max_plot(thresholds, profits)
    return thresholds[np.argmax(profits)]

def predict_with_threshold(y_pred_cont, threshold):
    '''
        Generates a boolean array with a given continuos array [0,1] and a defining threshold 
    '''
    return [1 if value > threshold else 0 for value in y_pred_cont ]

def threshold_optimization(y_pred_cont, y_test, threshold_range = (0.4, 0.6), iterations = 100, visualization=False):
    '''
        Given a set of treshold boundaries and a iteration number it calculates the profit for each treshold
    '''
    step = (threshold_range[1] - threshold_range[0]) / iterations
    thresholds = np.arange(threshold_range[0], threshold_range[1], step)
    profits = []
    for threshold in thresholds:
        y_pred = predict_with_threshold(y_pred_cont, threshold)
        
        # Evaluation metric should be dynamic
        profit = profit_share(y_pred, y_test)
        profits.append(profit)
    
    if visualization:
        data_visualization.xy_plot(x=thresholds, y=profits)
    
    return profits, thresholds

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)
    
def NN_evaluation(model, X_test, y_test):
    y_predicted = model.predict(X_test)
    threshold = max_threshold(y_predicted, y_test, threshold_range = (0.1, 0.99),iterations=10000, visualization=False)
    y_pred = predict_with_threshold(y_predicted,threshold)

    print("Accuracy {:1.2f}".format(calculate_accuracy(y_pred, y_test)))
    print("Area under the curve {:1.2f}".format(calculate_auc(y_pred, y_test)))
    print("Precision {:1.2f}".format(calculate_precision_score(y_pred, y_test)))
    print("Recall {:1.2f}".format(calculate_recall_score(y_pred, y_test)))
    print("Profit Share {:1.2f}".format(profit_share(y_pred, y_test)))
    return calculate_accuracy(y_pred, y_test), calculate_auc(y_pred, y_test), calculate_precision_score(y_pred, y_test), calculate_recall_score(y_pred, y_test), profit_share(y_pred, y_test)

def Cross_Val_Models(models, X, y, scaler=None, n_splits=5, sampling_technique=None):
    """
    Pass the dictionary of all the model you want to do the cross validation for. 
    For Example:  {"GaussianNB" : GaussianNB(), "MultinomialNB" : MultinomialNB()}
    """
    results = {}
    for model in models.keys():
        y_predicted = cross_validation_average_results(models[model], X, y, n_splits,scaler=scaler,sampling_technique=sampling_technique)
        threshold = max_threshold(y_predicted, y, threshold_range = (0.1, 0.99),iterations=1000, visualization=True)
        y_pred = predict_with_threshold(y_predicted,threshold)
        results[model] = profit_share(y_pred, y)
    return results