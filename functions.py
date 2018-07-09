import pandas as ps
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

ps.options.mode.chained_assignment = None  # default='warn' turns off warning messages


def twitter_analyzer(path, column1, column2, has_neutral, neutral_name, embed_units, embed_dim, lstm_output, batch_size, random_state):
    # load data and transform it so that only positive and negative sentiment values are left
    data = read_data(path, column1, column2)
    data = transform_data(data, has_neutral, neutral_name, column1, column2)

    #print("Positive elements: %i" % data[data[column2] == 'positive'].size)
    #print("Negative elements: %i" % data[data[column2] == 'negative'].size)

    # tokenize the data so that the neural network can use the data
    x = tokenize_data(data, 2000, column1)

    # split the data into 33% test data and rest for training
    x_train, x_test, y_train, y_test = split_training_test_data(data, x, column2, 0.33, random_state)

    #print("Shape of x training, y training: %s,%s" % (x_train.shape, y_train.shape))
    #print("Shape of x test, y test: %s,%s" % (x_test.shape, y_test.shape))

    # build the model with the given parameters
    model = build_model(embed_units, embed_dim, x.shape[1], lstm_output)
    #print model.summary()

    # train the model
    train_model(model, x_train, y_train, 5, batch_size, (x_test, y_test))

    # evaluate the previous trained model
    score, acc = evaluate_model(model, x_test, y_test, batch_size)

    print("score: %.2f" % score)
    print("acc: %.2f" % acc)
    return score, acc


def tune_hyperparameters(path, column1, column2, has_neutral, neutral_name, embed_units, random_state):
    score = 0.00
    acc = 0.00
    embed_dim = 128
    tmp_embed_dim = embed_dim
    lstm_output = 196
    tmp_lstm_output = lstm_output
    batch_size = 32
    tmp_batch_size = batch_size

    for i in range(0,5):
        tmpscore, tmpacc = twitter_analyzer(path, column1, column2, has_neutral, neutral_name, embed_units, embed_dim, lstm_output,
                         batch_size, random_state)

        if(tmpscore < score and tmpacc >= acc):
            acc = tmpacc
            score = tmpscore
            embed_dim = tmp_embed_dim
            lstm_output = tmp_lstm_output
            batch_size = tmp_batch_size

        tmp_embed_dim = tmp_embed_dim / 2
        tmp_lstm_output = tmp_lstm_output / 2
        tmp_batch_size = tmp_batch_size / 2

    print (score, acc)
    return embed_dim, lstm_output, batch_size

# Reads a csv file from a string path to the file and returns
def import_csv(path):
    return ps.read_csv(path)


# Reads data from csv. Expects the path of csv,
# column1 as name of text column and column2 as name of sentiment column in csv
def read_data(path, column1, column2):
    data = import_csv(path)
    return data[[column1, column2]]


# Gets data in form of numpy array Transforms the data so that every tweet text is in lowercase
# and contains only letters and digits.
# If the dataset contains neutral sentiments, has_neutral is to be set to true,
# and neutral_name must be set to the string of the corresponding instance in data set.
# Then those data entries are no longer relevant
# Returns the transformed data
def transform_data(data, has_neutral, neutral_name, text, sentiment):
    if has_neutral:
        data = data[data[sentiment].values != neutral_name]
    tmpdata = data
    data[text] = tmpdata[text].apply(lambda x: x.lower())
    tmpdata2 = data
    data[text] = tmpdata2[text].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    for idx, row in data.iterrows():
        row[0] = row[0].replace('rt', ' ')

    return data


# Gets the data set in form of numpy array and tokenizes the data, so that Neural Network can work with the data.
# Transforms the text data to vectors
# Tokenizer used is from keras module
# Returns the tokenized data set
def tokenize_data(data, max_features, text):
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(data[text].values)
    X = tokenizer.texts_to_sequences(data[text].values)
    X = pad_sequences(X)
    return X


# Builds the Neural Network model.
# Max_features is number of units in Embedding layer.
# embed_dim is the dimension of Embedding layer.
# in_length is the expected length of input in first layer.
# lstm_out is the number of units of the output of LSTM layer in Network
def build_model(max_features, embed_dim, in_length, lstm_out):
    model = Sequential()
    model = add_embedding_layer(model, max_features, embed_dim, input_length=in_length)
    model = add_sptialdropout_layer(model, 0.4)
    model = add_lstm_layer(model, lstm_out, 0.2, 0.2)
    model = add_dense_layer(model, 2, 'softmax')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Adds an Embedding layer to the model with the specified parameters
# units is the number of units of the Embedding layer
# dim is the dimension of the Embedding layer
# input_length is the expected length of the input
# Returns the modified model
def add_embedding_layer(model, units, dim, input_length):
    model.add(Embedding(units, dim, input_length=input_length))
    return model


# Adds an SpatialDropout1D layer to the model.
# rate is the rate that changes the data for avoid overfitting
# Returns the modified model
def add_sptialdropout_layer(model, rate):
    model.add(SpatialDropout1D(rate))
    return model


# Adds an LSTM layer to the model.
# output is the dimension of the output value
# dropout is the Float between 0 and 1. Fraction to drop of the values for linear transformation of input
# recurrent_dropout is Float between 0 and 1. Fraction of the units
# to drop for the linear transformation of the recurrent state.
# Returns the modified model
def add_lstm_layer(model, output, dropout, recurrent_dropout):
    model.add(LSTM(output, dropout=dropout, recurrent_dropout=recurrent_dropout))
    return model


# Adds an Dense layer to the model.
# output is the ouput dimension of the layer
# activation is the activation function to use
# Returns the modified model
def add_dense_layer(model, output, activation):
    model.add(Dense(output, activation=activation))
    return model


# First gets the sentiment data and puts it in y
# then uses sklearns algorithm to shuffle the data into training and test data
# test_size is a float between 0 and 1, which defines how much data is going to be test data
# if random_state is se, the data will always get split the same way
def split_training_test_data(data, x, sentiment, test_size, random_state):
    y = ps.get_dummies(data[sentiment]).values
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_model(model, x, y, epochs, batch_size, validation_data):
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=validation_data)


def evaluate_model(model, x, y, batch_size):
    return model.evaluate(x, y, verbose=2, batch_size=batch_size)
