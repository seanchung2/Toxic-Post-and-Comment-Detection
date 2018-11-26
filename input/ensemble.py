import pandas as pd


submission = pd.read_csv("../input/submission.csv")
nbsvm = pd.read_csv("../input/submission_nbsvm.csv")
lstm = pd.read_csv("../input/submission_lstm_20000_100_350_50_50_10_10.csv")

p = 0.7
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submission[label_cols] = p*nbsvm[label_cols] + (1-p)*lstm[label_cols]
submission.to_csv('../input/submission_ensemble.csv', index=False)
