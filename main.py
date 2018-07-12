import sys, getopt
import twitter_analyzer


def main(argv):
    dataset = None
    tune_params = ""
    data = None
    try:
        opts, args = getopt.getopt(argv, "hd:t:", ["dataset=", "tune-paras="])
    except getopt.GetoptError:
        print 'main.py -d <dataset> -t <tune_parameters>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -d <dataset> -t <tune_parameters>'
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-t", "--tune-paras"):
            tune_params = arg

    if dataset == "airlines":
        # Airline data
        data = twitter_analyzer.read_data('./Airline.csv', 'text', 'airline_sentiment')
    elif dataset == 'apple':
        # Apple data
        data = twitter_analyzer.read_data('./train.csv', 'text', 'sentiment')
    elif dataset == 'product':
        # product/company data
        data = twitter_analyzer.read_data('./tweet_product_company.csv', 'tweet_text', 'is_there_an_emotion_directed_at_a_brand_or_product')

    if tune_params == "true":
        batch_sizes = [32, 64, 128]
        epochs = [7, 10, 15]
        embed_dims = [64, 128, 256]
        lstm_outputs = [98, 196, 294]

        epoch, batch = twitter_analyzer.tune_parameters_epoch_batch(data, epochs, batch_sizes)
        print "best epoch is %i and best batch size is %i" % (epoch, batch)

        embed_dim, lstm_output, score, acc = twitter_analyzer.tune_hyperparameters(data, epoch, batch, embed_dims,
                                                                                   lstm_outputs)
        print "best embed_dim is %i and best lstm_ouput is %i" % (embed_dim, lstm_output)

        print "score: %.2f" % score
        print "acc: %.2f" % acc

        if dataset == "airlines":
            # Airline analyzer call
            twitter_analyzer.analyze(data, True, "positive", "neutral", "negative", True, 2000,
                                     embed_dim, lstm_output, batch, 0, epoch)
        elif dataset == 'apple':
            # Apple data
            twitter_analyzer.analyze(data, True, 5, "3", 1, False, 2000,
                                     embed_dim, lstm_output, batch, 0, epoch)
        elif dataset == 'product':
            # product/company data
            twitter_analyzer.analyze(data, True,"Positive emotion", "No emotion toward brand or product", "Negative emotion", True, 2000,
                                     embed_dim, lstm_output, batch, 0, epoch)

    else:
        if dataset == "airlines":
            # Airline analyzer call
            twitter_analyzer.analyze(data, True, "positive", "neutral", "negative", True, 2000, 128, 196, 32, 0, 7)
        elif dataset == 'apple':
            # Apple data
            twitter_analyzer.analyze(data, True, 5, "3", 1, 2000, 128, 196, 32, 0, 7)
        elif dataset == 'product':
            # product/company data
            twitter_analyzer.analyze(data, True, "Positive emotion", "No emotion toward brand or product", "Negative emotion", True, 2000, 128, 196, 32, 0, 7)


if __name__ == "__main__":
   main(sys.argv[1:])