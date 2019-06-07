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
from preprocessing import Min_Max_Train

################################### New methods #############################################
def get_titanic_dataset():
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test


def get_timeseries_dataset():
    '''
        Returns the dataset provided for the project as a dataframe
    '''
    df = pd.read_csv("data_files/TimeSeries/weatherAUS.csv")
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train, X_test = Min_Max_Train(X_train, X_test)
    
    return X_train, X_test, y_train, y_test

    
def get_text_dataset():
    '''
        Returns the dataset provided for the project as a dataframe
    '''
    pos_lines_imdb = codecs.open("data_files/Text/positive.txt","r",encoding="latin2").read()
    neg_lines_imdb = codecs.open("data_files/Text/negative.txt","r", encoding="latin2").read()
    
    all_words_imdb = []
    documents_imdb = []
    
    allowed_word_types = ["J"]
    
    for p in pos_lines_imdb.split("\n"):
        documents_imdb.append((p,"pos"))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words_imdb.append(w[0].lower())
    
    for p in neg_lines_imdb.split("\n"):
        documents_imdb.append((p,"neg"))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words_imdb.append(w[0].lower())
                
    # Get list of lists in dataframe
    headers = ["text", "y"]
    df = pd.DataFrame(documents_imdb, columns=headers)
    df["y"] = df["y"].map({"pos": 1, "neg": 0})
    
    return df
    
def get_image_dataset():
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

def get_image_for_normal_nn(limit_train, limit_test):
    
    train_generator, test_generator = get_image_dataset()
    # limit m is choosen for testing reasons, computation over all images takes 1000 years and is problematic due
    # to the fact that 13999 can not be divided by the batch size of 20. did not know how to access that somehow else
    limit_train=limit_train
    limit_test=limit_test
    # get dataframe with flattend train images corresponding to size 64x64x3
    images_train = pd.DataFrame()
    m = 0
    for i in train_generator:
        n=0
        m = m+1
        if m == limit_train:
            break
        else:
            while n <= 19:
                a = i[0][n].flatten()
                print(a)
                n = n+1
                b= pd.DataFrame(a)
                b=b.transpose()
                images_train=images_train.append(b)
                
    # get dataframe with flattend train images corresponding to size 64x64x3
    # also smaller sample
    m = 0
    for i in test_generator:
        n=0
        m = m+1
        if m == limit_test:
            break
        else:
            while n <= 19:
                a = i[0][n].flatten()
                print(a)
                n = n+1
                b= pd.DataFrame(a)
                b=b.transpose()
                images_test=images_test.append(b)
                
    # get train target (same m as in image data)
    target_train = []
    m=0
    for i in train_generator:
        m = m+1
        if m==limit_train:
            break
        else:
            c = i[1].tolist()
            target_train = target_train + c
    
    # get train target (same m as in image data)
    target_test = []
    m=0
    for i in test_generator:
        m = m+1
        if m==limit_test:
            break
        else:
            c = i[1].tolist()
            target_test = target_test + c
            
    X_train, X_test, y_train, y_test = images_train, images_test, target_train, target_test
    
    return X_train, X_test, y_train, y_test
    

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