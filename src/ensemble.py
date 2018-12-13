import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout

submission = pd.read_csv("../input/submission.csv")
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

''' 0.9828 '''
nbsvm = pd.read_csv("../input/submission_nbsvm.csv")
lstm = pd.read_csv("../input/submission_lstm.csv")
cnn = pd.read_csv("../input/submission_cnn.csv")

p = 0.5
q = 0.4
submission[label_cols] = p*cnn[label_cols] + q*nbsvm[label_cols] + (1-p-q)*lstm[label_cols]
submission.to_csv('../input/submission_ensemble.csv', index=False)

'''
# training data
trainLabel = pd.read_csv("../input/train.csv")[label_cols]
lstmTrain = np.array(pd.read_csv("../input/trainResult_lstm.csv")[label_cols])
nbsvmTrain = np.array(pd.read_csv("../input/trainResult_nbsvm.csv")[label_cols])
cnnTrain = np.array(pd.read_csv("../input/trainResult_cnn.csv")[label_cols])
# reshape input from 159571*3*6 to 159571*18
train = np.array(zip(lstmTrain, nbsvmTrain, cnnTrain)).reshape(len(zip(lstmTrain, nbsvmTrain, cnnTrain)), 3 * 6)

# testing data
lstmTest = np.array(pd.read_csv("../input/submission_lstm.csv")[label_cols])
nbsvmTest = np.array(pd.read_csv("../input/submission_lstm.csv")[label_cols])
cnnTest = np.array(pd.read_csv("../input/submission_lstm.csv")[label_cols])
# reshape input from 159571*3*6 to 159571*18
test = np.array(zip(lstmTest, nbsvmTest, cnnTest)).reshape(len(zip(lstmTest, nbsvmTest, cnnTest)), 3 * 6)

# build model
model = Sequential()
model.add(Dense(6, input_dim=18, activation='sigmoid'))
# model.add(Dropout(0.1))
# model.add(Dense(3, activation="relu"))
# model.add(Dropout(0.1))
# model.add(Dense(6, activation="sigmoid"))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(train, trainLabel, epochs=2, batch_size=4, validation_split=0.1)
scores = model.evaluate(train, trainLabel)

# prediction
submission = pd.read_csv("../input/submission.csv")
prediction = model.predict(test)
print np.array(prediction).shape
submission[label_cols] = prediction
submission.to_csv("../input/submission_ensemble.csv", index=False)
'''


# # training data
# trainLabel = np.array(pd.read_csv("../input/train.csv")[label_cols])
# lstmTrain = np.array(pd.read_csv("../input/trainResult_lstm.csv")[label_cols])
# nbsvmTrain = np.array(pd.read_csv("../input/trainResult_nbsvm.csv")[label_cols])
# cnnTrain = np.array(pd.read_csv("../input/trainResult_cnn.csv")[label_cols])
# # reshape input from 159571*3*6 to 159571*18
# train = []
# for i in range(6):
#     train.append(np.array(zip(lstmTrain[:, i], nbsvmTrain[:, i], cnnTrain[:, i])))
# train = np.array(train)
#
# # testing data
# lstmTest = np.array(pd.read_csv("../input/submission_lstm.csv")[label_cols])
# nbsvmTest = np.array(pd.read_csv("../input/submission_lstm.csv")[label_cols])
# cnnTest = np.array(pd.read_csv("../input/submission_lstm.csv")[label_cols])
# # reshape input from 159571*3*6 to 159571*18
# test = []
# for i in range(6):
#     test.append(np.array(zip(lstmTest[:, i], nbsvmTest[:, i], cnnTest[:, i])))
# test = np.array(test)
#
# # build model
# model = []
# for i in range(6):
#     model.append(Sequential())
#     model[i].add(Dense(1, input_dim=3, activation='sigmoid'))
#     # model[i].add(Dense(1, activation="sigmoid"))
#     model[i].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model[i].fit(train[i], trainLabel[:, i], epochs=10, batch_size=32, validation_split=0.25)
#     # scores = model.evaluate(train, trainLabel)
#
# # prediction
# submission = pd.read_csv("../input/submission.csv")
# prediction = []
# for i in range(6):
#     prediction.append(model[i].predict(test[i]))
#     submission[label_cols[i]] = prediction[i]
# submission.to_csv("../input/submission_ensemble.csv", index=False)


