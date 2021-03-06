import pandas as ps
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

import keras
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

import matplotlib.pyplot as plot

ps.options.mode.chained_assignment = None  # default='warn' turns off warning messages

# Main function of the API
# Retuns score of loss function and accuracy of the model
def analyze(data, has_neutral, positive_name, neutral_name, negative_name, show_visualizations, embed_units, embed_dim, lstm_output, batch_size, random_state,
            epochs):
    # load data and transform it so that only positive and negative sentiment values are left
    column1, column2 = get_column_names(data)

    # draw wordcloud for given dataset
    #show_wordcloud(data, show_visualizations, sentiment_column_name=column2, text_column_name=column1, negative_sentiment=negative_name, positive_sentiment=positive_name)

    data = transform_data(data, has_neutral, neutral_name, column1, column2)

    # tokenize the data so that the neural network can use the data
    x = tokenize_data(data, 2000, column1)

    # split the data into 33% test data and rest for training
    x_train, x_test, y_train, y_test = split_training_test_data(data, x, column2, 0.33, random_state)

    # build the model with the given parameters
    model = build_model(embed_units, embed_dim, x.shape[1], lstm_output, y_train.shape[1])

    # train the model
    history = train_model(model, x_train, y_train, epochs, batch_size, (x_test, y_test))

    # visualize training history
    show_training_history(history, show_visualizations)

    # evaluate the previous trained model
    score, acc = evaluate_model(model, x_test, y_test, batch_size)

    print("score: %.2f" % score)
    print("acc: %.2f" % acc)
    return score, acc


# Method to tune the hyperparameters for the model
# Hyperparameters tuned are dimension of embedding layer (embed_dim), output of LSTM layer (lstm_output)
# Passed arguments are the dataset as DataFrame, the used epochs and batch_size
# embed_dim is an array with possible values for embedding dimension and lstm_output is an array with possible values
# returns values of best embed_dim, lstm_output and accuracy and score of loss function
def tune_hyperparameters(data, has_neutral, neutral_name, positive, negative, epochs, batch_size, embed_dim, lstm_output):
    score = 10.00
    acc = 0.00
    i = 0
    j = 0
    best_embed = 0
    best_lstm = 0
    # Take only a little amount of data for test purposes
    data = load_small_data_set(data)
    total_iterations = len(embed_dim) * len(lstm_output)
    count = 1

    while i < len(embed_dim):
        j= 0
        while j < len(lstm_output):
            print("%i  iteration of %i" % (count, total_iterations))
            tmpscore, tmpacc = analyze(data, has_neutral, positive, neutral_name, negative, False, 2000, embed_dim[i], lstm_output[j], batch_size, 42, epochs)

            if(tmpscore < score and tmpacc >= acc):
                score = tmpscore
                acc = tmpacc
                best_embed = embed_dim[i]
                best_lstm = lstm_output[j]

            j = j + 1
            count = count + 1

        i = i + 1

    print (score,acc)
    return best_embed, best_lstm, score, acc

# Tunes the epoch and batch size for the model
# Returns best epoch and batch size combination of given values
def tune_parameters_epoch_batch(data,has_neutral, neutral_name, positive, negative, epochs, batch_sizes):
    data = load_small_data_set(data)
    i = 0
    j = 0
    score = 10.00
    acc = 0.00
    best_epoch = 0
    best_batch = 0
    total_iterations = len(epochs) * len(batch_sizes)
    count = 1

    while i < len(batch_sizes):
        j = 0
        while j < len(epochs):
            print("%i  iteration of %i" % (count, total_iterations))
            tmpscore, tmpacc = analyze(data, has_neutral, positive, neutral_name, negative, False, 2000, 128, 196, batch_sizes[i], 42, epochs[j])

            if(tmpscore < score and tmpacc >= acc):
                score = tmpscore
                acc = tmpacc
                best_epoch = epochs[j]
                best_batch = batch_sizes[i]

            j = j + 1
            count = count + 1

        i = i + 1

    return best_epoch, best_batch

# Takes the DataFrame and returns the first n rows of it
def load_small_data_set(data):
    n = data.shape[0]/ 10
    return data.head(n)


# Gets the name of the text column and the sentiment column
def get_column_names(data):
    return data.dtypes.index[0], data.dtypes.index[1]


# Reads a csv file from a string path to the file and returns pandas array
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
    data = data.dropna()

    if has_neutral:
        data = data.replace(neutral_name, ps.NaT).dropna()

    data[text].str.replace("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", True)
    tmpdata = data
    data[text] = tmpdata[text].str.lower()

    for idx, row in data.iterrows():
        row[0] = row[0].replace('rt', ' ')

    return data


# Gets the data set in form of numpy array and tokenizes the data, so that Neural Network can work with the data.
# Transforms the text data to vectors
# Tokenizer used is from keras module
# max_features: size of the vocabulary
# Returns the tokenized data set
def tokenize_data(data, max_features, text):
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(data[text].values)
    X = tokenizer.texts_to_sequences(data[text].values)
    X = pad_sequences(X)
    return X


# Builds the Neural Network model.
# Max_features is the size of the vocabulary
# embed_dim is the dimension of Embedding layer.
# in_length is the expected length of input in first layer.
# lstm_out is the number of units of the output of LSTM layer in Network
def build_model(max_features, embed_dim, in_length, lstm_out, dense_in):
    model = Sequential()
    model = add_embedding_layer(model, max_features, embed_dim, input_length=in_length)
    model = add_sptialdropout_layer(model, 0.4)
    model = add_lstm_layer(model, lstm_out, 0.2, 0.2)
    model = add_dense_layer(model, dense_in, 'softmax')
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


# Trains a model with given data, validation data and batch size and number of epochs
def train_model(model, x, y, epochs, batch_size, validation_data):
    return model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=validation_data)


# Evaluates a model with given data and batch size
# Returns score of loss function and accuracy of predictions
def evaluate_model(model, x, y, batch_size):
    return model.evaluate(x, y, verbose=2, batch_size=batch_size)

def show_wordcloud(data, show_visualizations, sentiment_column_name, text_column_name, negative_sentiment, positive_sentiment):
    # TODO: put into different function

    if show_visualizations:
        print data.columns.get_values()

        data_positive_sent = data[data[sentiment_column_name] == positive_sentiment]
        data_positive_sent = data_positive_sent[text_column_name]

        positive_words = ' '.join(data_positive_sent)
        cleaned_word = ' '.join([word for word in positive_words.split()
                                 if 'http' not in word
                                 and not word.startswith('@')
                                 and not word.startswith('#')
                                 and word != 'RT'
                                 ])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=2500, height=2000).generate(
            cleaned_word)
        plot.title('Positive sentiment')
        plot.figure(1, figsize=(13, 13))
        plot.imshow(wordcloud)
        plot.axis('off')
        plot.show()

        data_negative_sent = data[data[sentiment_column_name] == negative_sentiment]
        data_negative_sent = data_negative_sent[text_column_name]

        negative_words = ' '.join(data_negative_sent)
        cleaned_word = ' '.join([word for word in negative_words.split()
                                 if 'http' not in word
                                 and not word.startswith('@')
                                 and not word.startswith('#')
                                 and word != 'RT'
                                 ])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=2500, height=2000).generate(
            cleaned_word)
        plot.title('Negative sentiment')
        plot.figure(1, figsize=(13, 13))
        plot.imshow(wordcloud)
        plot.axis('off')
        plot.show()



def show_training_history(history, show_visualizations):

    if show_visualizations:
        # plot accuracy
        plot.plot(history.history['acc'])
        plot.plot(history.history['val_acc'])
        plot.title('model accuracy')
        plot.ylabel('accuracy')
        plot.xlabel('epoch')
        plot.legend(['train', 'test'], loc='upper left')
        plot.show()

        # plot loss
        plot.plot(history.history['loss'])
        plot.plot(history.history['val_loss'])
        plot.title('model loss')
        plot.ylabel('loss')
        plot.xlabel('epoch')
        plot.legend(['train', 'test'], loc='upper left')
        plot.show()

