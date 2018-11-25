# encoding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import string


# train and test are data-sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# submitFormat is the format for submission
submitFormat = pd.read_csv("../input/submission.csv")

# labels for each kind of toxic type
toxicLabels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# we need to fill all the N/A in data with "unknown" (or something meaningless) in case
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)


''' Build model '''
# re_token is a way to pre compile what needs to be split (this can speed up for recurrence)
re_token = re.compile('{string.punctuation}\",\.\'')


# split by the delimiters from each comment/post
def tokenize(s):
    return re_token.sub(r' \1 ', s).split()


# TF-IDF: term frequency–inverse document frequency
# Is a numerical statistic that is intended to reflect how important a word is to a document
# The merits of it is to filter some frequent but meaningless words (e.g., I, this, etc.)
# TF: the frequency of THE word
# IDF: log(a/b), a is number of document, b is number of document has THE word
''' Specification:
min_df(max_df): float in range [0.0, 1.0] or int, default=1
    When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
    This value is also called cut-off in the literature.
    If float, the parameter represents a proportion of documents, integer absolute counts.
    This parameter is ignored if vocabulary is not None.
smooth_idf: Smooth idf weights by adding one to document frequencies,
    as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.
sublinear_tf: Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
'''
# was (1, 2), 3
vector = TfidfVectorizer(ngram_range=(1, 3), tokenizer=tokenize, min_df=1, max_df=0.97, strip_accents='unicode',
                         use_idf=1, smooth_idf=1, sublinear_tf=1)
# fit_transform => fit then transform
# fit: tokenize and build vocab
# transform: encode document
trainTermDocument = vector.fit_transform(train[COMMENT])
test_term_doc = vector.transform(test[COMMENT])


# this is the function to calculate the conditional probability for naive bayes
def conditionalProbability(y_i, y):
    # the reason to add one at numerator and denominator is to prevent from zero occurrence
    return (trainTermDocument[y == y_i].sum(0) + 1) / ((y == y_i).sum() + 1)


def get_mdl(y):
    y = y.values
    # r is the conditional probability (parameter) for naive bayes
    r = np.log(conditionalProbability(1, y) / conditionalProbability(0, y))
    # Specification:
    # C: Inverse of regularization strength; must be a positive float.
    #    Like in support vector machines, smaller values specify stronger regularization.
    # dual: Dual or primal formulation.
    #    Dual formulation is only implemented for l2 penalty with liblinear solver.
    #    Prefer dual=False when n_samples > n_features.
    m = LogisticRegression(C=4, dual=True)
    x_nb = trainTermDocument.multiply(r)
    return m.fit(x_nb, y), r


# result of prediction
prediction = np.zeros((len(test), len(toxicLabels)))

# iterate through each label
for i, j in enumerate(toxicLabels):
    print('fit', j)
    m, r = get_mdl(train[j])
    prediction[:, i] = m.predict_proba(test_term_doc.multiply(r))[:, 1]

# final submission
submission = pd.concat([pd.DataFrame({'id': submitFormat["id"]}), pd.DataFrame(prediction, columns=toxicLabels)], axis=1)
submission.to_csv('../input/submission_nbsvm.csv', index=False)
