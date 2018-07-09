import functions

embed_dim, lstm_output, batch_size = functions.tune_hyperparameters('./Airline.csv', 'text','airline_sentiment', True, "neutral", 2000, 42)
print (embed_dim, lstm_output, batch_size)
#functions.twitter_analyzer('./Airline.csv', 'text','airline_sentiment', True, "neutral", 2000, 128, 196, 32)