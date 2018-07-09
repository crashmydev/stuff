import pandas as ps
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D


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

    data[text] = data[text].apply(lambda x: x.lower())
    data[text] = data[text].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

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
# Returns the modified model
def add_lstm_layer(model, output, dropout, recurrent_dropout):
    model.add(LSTM(output, dropout=dropout, recurrent_dropout=recurrent_dropout))
    return model


# Adds an Dense layer to the model.
# Returns the modified model
def add_dense_layer(model, output, activation):
    model.add(Dense(output, activation=activation))
    return model
