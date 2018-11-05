from keras.models import Model, model_from_json
import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

# Loading the train and test files
test = pd.read_csv('../input/test.csv')
testLabel = pd.read_csv('../input/test_labels.csv')

# The dependent variables are in the training set itself so we need to split them up, into X and Y sets.
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
Y_test = testLabel[list_classes].values
list_sentences_test = test["comment_text"]

Y_test *= -1

# Padding is needed as each comment/post may have different length of words
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_test))
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
maxlen = 200
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

# evaluate the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
