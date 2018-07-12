import twitter_analyzer


data = twitter_analyzer.read_data('./train.csv', 'text', 'sentiment')
twitter_analyzer.analyze(data, True, 5, "3", 1, False, 2000, 128, 196, 32, 0, 7)

#data = twitter_analyzer.read_data('./tweet_product_company.csv', 'tweet_text', 'is_there_an_emotion_directed_at_a_brand_or_product')
#twitter_analyzer.analyze(data, True, "Positive emotion", "No emotion toward brand or product", "Negative emotion", True, 2000, 128, 196, 32, 0, 7)

#data = twitter_analyzer.read_data('./Airline.csv', 'text', 'airline_sentiment')
#twitter_analyzer.analyze(data, True, "positive", "neutral", "negative", True, 2000, 128, 196, 32, 0, 7)

#functions.twitter_analyzer('./Airline.csv', 'text','airline_sentiment', True, "neutral", 2000, 128, 196, 32)