import pandas as ps
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, GRU
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


dataset = ps.read_csv("tweet_product_company.csv", sep = ',')
count_vec = CountVectorizer()
data = count_vec.fit_transform(dataset)
print count_vec.get_feature_names()