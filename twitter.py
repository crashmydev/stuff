import functions

data = functions.read_data('./Airline.csv', 'text', 'airline_sentiment')
batch_sizes = [128]
epochs = [7]
embed_dims = [64, 128, 256]
lstm_outputs = [98, 196, 294]

epoch, batch = functions.tune_parameters_epoch_batch(data, epochs, batch_sizes)
print "best epoch is %i and best batch size is %i" % (epoch, batch)

embed_dim, lstm_output, score, acc = functions.tune_hyperparameters(data, epoch, batch, embed_dims, lstm_outputs)
print "best embed_dim is %i and best lstm_ouput is %i" % (embed_dim, lstm_output)

print "score: %.2f" % score
print "acc: %.2f" % acc

functions.twitter_analyzer('./Airline.csev', 'text','airline_sentiment', True, "neutral", 2000, embed_dim, lstm_output,
                           batch, 0, epoch)