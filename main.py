import twitter_analyzer

data = twitter_analyzer.read_data('./tweet_product_company.csv', 'tweet_text', 'is_there_an_emotion_directed_at_a_brand_or_product')
#data = twitter_analyzer.read_data('./Airline.csv', 'text', 'airline_sentiment')

batch_sizes = [32, 64, 128]
epochs = [7, 10, 15]
embed_dims = [64, 128, 256]
lstm_outputs = [98, 196, 294]

epoch, batch = twitter_analyzer.tune_parameters_epoch_batch(data, epochs, batch_sizes)
print "best epoch is %i and best batch size is %i" % (epoch, batch)

embed_dim, lstm_output, score, acc = twitter_analyzer.tune_hyperparameters(data, epoch, batch, embed_dims, lstm_outputs)
print "best embed_dim is %i and best lstm_ouput is %i" % (embed_dim, lstm_output)

print "score: %.2f" % score
print "acc: %.2f" % acc

#twitter_analyzer.analyze(data, True, "neutral", 2000, embed_dim, lstm_output, batch, 0, epoch)

twitter_analyzer.analyze(data, True, "No emotion toward brand or product", 2000, embed_dim, lstm_output, batch, 0, epoch)