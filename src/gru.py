import sys, os, re, csv, codecs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Model, model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Input, GRU
from keras.layers import LSTM, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers


max_features = 20000
maxlen = 100
embed_size = 350  # 128
lstm_output = 50
relu_output = 50
dropout1 = 0.10
dropout2 = 0.10
weight_file_path = "gru_weights.ckpt"

model_file_path = "gru_model.json"\
    .format(max_features, maxlen, embed_size, lstm_output, relu_output, int(dropout1*100), int(dropout2*100))
# Loading the train and test files
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
testLabel = pd.read_csv('../input/test_labels.csv')

train = train.sample(frac=1)
# A common pre-processing step is to check for nulls,
# and fill the null values with something before proceeding to the next steps.
# If you leave the null values intact, it will trip you up at the modelling stage later
train.isnull().any(), test.isnull().any()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

# The approach that we are taking is to feed the comments into the LSTM as part of the neural network
# but we can't just feed the words as it is.
#
# So this is what we are going to do:
# Tokenization - We need to break down the sentence into unique words.
#               For eg, "I love cats and love dogs" will become ["I","love","cats","and","dogs"]
# Indexing - We put the words in a dictionary-like structure and give them an index each
#               For eg, {1:"I",2:"love",3:"cats",4:"and",5:"dogs"}
# Index Representation - We could represent the sequence of words in the comments in the form of index,
# and feed this chain of index into our LSTM.
#               For eg, [1,2,3,4,2,5]
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

# # Padding is needed as each comment/post may have different length of words
# # If maxlen is too short, might lose some useful feature that could cost some accuracy points down the path
# # If maxlen is too long, LSTM cell will have to be larger to store the possible values or states
# # Solution ==> check the distribution of the number of words in sentences
# totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
# plt.hist(totalNumWords, bins=np.arange(0, 410, 10))
# plt.show()
# from the result ==> 200 seems to be a fair number
X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)


def createModel():
    # Input layer
    # Setup input dimension (an empty space after comma => tell Keras to infer the number automatically)
    inputData = Input(shape=(maxlen,))

    # Embedding layer
    # basically it is word2vec
    # Embedding allows us to reduce model size and most importantly the huge dimensions we have to deal with,
    # in the case of using one-hot encoding to represent the words in our sentence.
    # Also, the distance of these coordinates can be used to detect relevance and context.
    #
    # The output of the Embedding layer is just a list of the coordinates of the words in this vector space.
    # We need to define the size of the "vector space" we have mentioned above,
    # and the number of unique words(max_features) we are using.
    x = Embedding(max_features, embed_size)(inputData)          # output: 3-D tensor (None, 200, 128)

    # LSTM layler
    # LSTM takes in a tensor of [Batch Size, Time Steps, Number of Inputs].
    #       Batch Size is the number of samples in a batch,
    #       Time Steps is the number of recursion it runs for each input
    #       Number of Inputs is the number of variables (number of words in each sentence) you pass into LSTM
    # we want the unrolled version, where 50 is the output dimension we have defined.
    x = Bidirectional(GRU(lstm_output, return_sequences=True, name='lstm_layer'))(x)
    # output: 3-D tensor (None, 200, 60)

    # Max pooling layer
    # Reshape the 3D tensor into a 2D one by max pooling.
    # We reshape carefully to avoid throwing away data that is important to us,
    # and ideally we want the resulting data to be a good representative of the original data.
    x = GlobalMaxPool1D()(x)

    # Dropout Layer
    # Dropout some nodes to achieve better generalization
    # Dropout 15% of nodes
    x = Dropout(dropout1)(x)

    # Activation Layer
    # A relu layer with output dimension of 60
    x = Dense(relu_output, activation="relu")(x)

    # Dropout Layer
    # Dropout 5% of nodes
    x = Dropout(dropout2)(x)

    # Activation Layer
    # A sigmoid layer since we are trying to achieve a binary classification(1,0) for each of the 6 labels,
    # and the sigmoid function will squash the output between the bounds of 0 and 1.
    x = Dense(6, activation="sigmoid")(x)

    # Wrap up
    # learning rate, the default is set at 0.001.
    model = Model(inputs=inputData, outputs=x)

    return model


def loadModel():
    try:
        # load json and create model
        json_file = open(model_file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weights into new model
        # model.load_weights("model.h5")
        loadWeights(model)
        print("Loaded model from disk")
        return model

    except Exception:
        model = createModel()
        print("Creating new model")
        return model


def saveModel(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    # model.save_weights("model.h5")
    print("Saved model to disk")


def testModel(model, X_test):
    y_test = model.predict(X_test)
    submission = pd.read_csv("../input/submission.csv")
    submission[list_classes] = y_test
    submission.to_csv("../input/submission_gru.csv", index=False)
    # submission.to_csv("../input/submission_lstm_{}_{}_{}_{}_{}_{}_{}.csv".format(max_features, maxlen, embed_size,
    #                                                                              lstm_output, relu_output,
    #                                                                              int(dropout1*100), int(dropout2*100))
    # ,
    #                   index=False)


def setupEarlyStop():
    checkpoint = ModelCheckpoint(weight_file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=30)
    callbacks_list = [checkpoint, early]

    return callbacks_list


def loadWeights(model):
    try:
        print('Loading weights...')
        model.load_weights(weight_file_path)
    except IOError:
        print('No weight file, creating one...')


model = loadModel()

''''''''''''''''''''
'''   Training   '''
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])   # SGD, Momentum, RMSprop, Adam

callbacks_list = setupEarlyStop()

model.fit(X_train, y, batch_size=32, epochs=2, validation_split=0.05, callbacks=callbacks_list)
model.summary()
saveModel(model)
loadWeights(model)

# test model (predicting)
testModel(model, X_test)

# save the prediction for training
trainResult = model.predict(X_train)
# result = pd.read_csv("../input/result.csv")
result = pd.concat([pd.DataFrame(trainResult, columns=list_classes)], axis=1)
result.to_csv('../input/trainResult_gru.csv', index=False)
