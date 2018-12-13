import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model, model_from_json
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, Reshape, Flatten, Concatenate, Dropout, \
    SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import warnings
import os


np.random.seed(42)
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'

# train and test are data-sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# submitFormat is the format for submission
submission = pd.read_csv('../input/submission.csv')

EMBEDDING_FILE = '../input/crawl-300d-2M.vec'
max_features = 100000
maxlen = 250
embed_size = 300

# we need to fill all the N/A in data with "unknown" (or something meaningless) in case
X_train = train["comment_text"].fillna("fillna").values
X_test = test["comment_text"].fillna("fillna").values

# train label
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train[list_classes].values

# The approach that we are taking is to feed the comments into the model as part of the neural network
# but we can't just feed the words as it is.
#
# So this is what we are going to do:
# Tokenization - We need to break down the sentence into unique words.
#               For eg, "I love cats and love dogs" will become ["I","love","cats","and","dogs"]
# Indexing - We put the words in a dictionary-like structure and give them an index each
#               For eg, {1:"I",2:"love",3:"cats",4:"and",5:"dogs"}
# Index Representation - We could represent the sequence of words in the comments in the form of index,
# and feed this chain of index into the model.
#               For eg, [1,2,3,4,2,5]
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# # Padding is needed as each comment/post may have different length of words
# # If maxlen is too short, might lose some useful feature that could cost some accuracy points down the path
# # If maxlen is too long, LSTM cell will have to be larger to store the possible values or states
# # Solution ==> check the distribution of the number of words in sentences
# totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
# plt.hist(totalNumWords, bins=np.arange(0, 410, 10))
# plt.show()
# from the result ==> 200 seems to be a fair number
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefficients(word, *arr):
    return word, np.asarray(arr, dtype='float32')


# import embedding index by looking up from the word2vec file
embeddings_index = dict(get_coefficients(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

# index each word
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))

# matrix: wordNumber x 300 (embed_size)
embedding_matrix = np.zeros((nb_words, embed_size))
# building matrix
for word, i in word_index.items():
    # only use word which is within max_features
    if i >= max_features:
        continue
    # if it is within max_features, put it into the embedding_matrix
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Build ROC AUC callback function
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    # derive ROC AUC statistics
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


# filterSize = [1, 2, 3, 5]
filterSize = [1, 2, 3, 5, 7, 8]
filterNumber = 32


# build model
def getModel():
    # the input size will be limited to maxlen
    inputLayer = Input(shape=(maxlen,))

    # Embedding allows us to reduce model size and most importantly the huge dimensions we have to deal with,
    # in the case of using one-hot encoding to represent the words in our sentence.
    # Also, the distance of these coordinates can be used to detect relevance and context.
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inputLayer)  # output:(None, maxlen, embed_size)

    # performs the same function as Dropout, however it drops entire 1D feature maps instead of individual elements
    x = SpatialDropout1D(x)    # output: (None, maxlen, embed_size)
    # reshape the layer to 2D in order to perform cnn
    x = Reshape((maxlen, embed_size, 1))(x)  # output: (None, maxlen, embed_size, 1)

    # computes a 2-D convolution
    conv_0 = Conv2D(filterNumber, kernel_size=(filterSize[0], embed_size), kernel_initializer='normal',
                    activation='elu')(x)    # output: (None, 100, 1, 32)
    conv_1 = Conv2D(filterNumber, kernel_size=(filterSize[1], embed_size), kernel_initializer='normal',
                    activation='elu')(x)    # output: (None, 99, 1, 32)
    conv_2 = Conv2D(filterNumber, kernel_size=(filterSize[2], embed_size), kernel_initializer='normal',
                    activation='elu')(x)    # output: (None, 98, 1, 32)
    conv_3 = Conv2D(filterNumber, kernel_size=(filterSize[3], embed_size), kernel_initializer='normal',
                    activation='elu')(x)    # output: (None, 96, 1, 32)
    conv_4 = Conv2D(filterNumber, kernel_size=(filterSize[4], embed_size), kernel_initializer='normal',
                    activation='elu')(x)    # output: (None, 94, 1, 32)
    conv_5 = Conv2D(filterNumber, kernel_size=(filterSize[5], embed_size), kernel_initializer='normal',
                    activation='elu')(x)    # output: (None, 93, 1, 32)

    # Max pooling operation for spatial data on each cnn
    maxpool0 = MaxPool2D(pool_size=(maxlen - filterSize[0] + 1, 1))(conv_0)    # output: (None, 1, 1, 32)
    maxpool1 = MaxPool2D(pool_size=(maxlen - filterSize[1] + 1, 1))(conv_1)    # output: (None, 1, 1, 32)
    maxpool2 = MaxPool2D(pool_size=(maxlen - filterSize[2] + 1, 1))(conv_2)    # output: (None, 1, 1, 32)
    maxpool3 = MaxPool2D(pool_size=(maxlen - filterSize[3] + 1, 1))(conv_3)    # output: (None, 1, 1, 32)
    maxpool4 = MaxPool2D(pool_size=(maxlen - filterSize[4] + 1, 1))(conv_4)    # output: (None, 1, 1, 32)
    maxpool5 = MaxPool2D(pool_size=(maxlen - filterSize[5] + 1, 1))(conv_5)    # output: (None, 1, 1, 32)

    # connect all cnn layers' outputs together (in parallel)
    z = Concatenate(axis=1)([maxpool0, maxpool1, maxpool2, maxpool3, maxpool4, maxpool5])  # output: (None, 6, 1, 32)
    # restore layer back to 1D in order to have 6 results
    z = Flatten()(z)  # output: (None, 192)
    # Dropout some nodes to achieve better generalization
    z = Dropout(0.1)(z)  # output: (None, 192)

    # A sigmoid layer since we are trying to achieve a binary classification(1,0) for each of the 6 labels,
    # and the sigmoid function will squash the output between the bounds of 0 and 1.
    outputLayer = Dense(6, activation="sigmoid")(z)

    # wrap up
    model = Model(inputs=inputLayer, outputs=outputLayer)

    return model


def saveModel(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    # model.save_weights("model.h5")
    print("Saved model to disk")


def loadModel():
    try:
        # load json and create model
        json_file = open("cnn_model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weights into new model
        # model.load_weights("model.h5")
        loadWeights(model)
        print("Loaded model from disk")
        return model

    except Exception:
        model = getModel()
        print("Creating new model")
        return model


def loadWeights(model):
    try:
        print('Loading weights...')
        model.load_weights('cnn_weights.ckpt')
    except IOError:
        print('No weight file, creating one...')


def setupEarlyStop():
    checkpoint = ModelCheckpoint('cnn_weights.ckpt', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=30)
    callbacks_list = [checkpoint, early]

    return callbacks_list


model = loadModel()

batch_size = 256/2
epochs = 2

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)#233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
#                  callbacks=[RocAuc], verbose=1)
model.summary()
# keep model and weights
saveModel(model)
model.save_weights('cnn_weights.ckpt')

# prediction of testing data
y_pred = model.predict(x_test, batch_size=1024)
submission[list_classes] = y_pred
submission.to_csv('../input/submission_cnn.csv', index=False)

# save the prediction for training
# ''' temp '''
# model = loadModel()
# ''''''
trainResult = model.predict(x_train)
result = pd.concat([pd.DataFrame(trainResult, columns=list_classes)], axis=1)
result.to_csv("../input/trainResult_cnn.csv", index=False)
