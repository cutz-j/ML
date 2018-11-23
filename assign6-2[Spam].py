import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.io
import re
from stemming.porter2 import stem
import nltk, nltk.stem.porter
from sklearn import svm

### 2 Spam Classification ###
fileOpen = open("d:/data/ex6/emailSample1.txt", "r")
ham1 = fileOpen.readlines()
ham1 = "".join(ham1)

def preProcess( email ):
    # Make the entire e-mail lower case
    email = email.lower()
    
    # Strip html tags (strings that look like <blah> where 'blah' does not
    # contain '<' or '>')... replace with a space
    email = re.sub('<[^<>]+>', ' ', email);
    
    #Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)
    
    #Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    
    #Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email);
    
    #The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email);
    
    return email

def email2TokenList( raw_email ):
    stemmer = nltk.stem.porter.PorterStemmer()
    email = preProcess( raw_email )
    #Split the e-mail into individual words (tokens) (split by the delimiter ' ')
    #but also split by delimiters '@', '$', '/', etc etc
    #Splitting by many delimiters is easiest with re.split()
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)
    
    #Loop over each word (token) and use a stemmer to shorten it,
    #then check if the word is in the vocab_list... if it is,
    #store what index in the vocab_list the word is
    tokenlist = []
    for token in tokens:
        #Remove any non alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token);

        #Use the Porter stemmer to stem the word
        stemmed = stemmer.stem( token )
        #Throw out empty tokens
        if not len(token): continue
        #Store a list of all unique stemmed words
        tokenlist.append(stemmed)
    return tokenlist

## 2.1.1 Vocabulary List ##
voca = email2TokenList(preProcess(ham1))

def getVocabDict(reverse=False):
    vocab_dict = {}
    with open("d:/data/ex6/vocab.txt") as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key
                
    return vocab_dict


def email2VocabIndices( raw_email, vocab_dict ):
    tokenlist = email2TokenList( raw_email )
    index_list = [ vocab_dict[token] for token in tokenlist if token in vocab_dict ]
    return index_list

all_dict = getVocabDict()
index_list = email2VocabIndices(ham1, all_dict)

## 2.2 Extracting Features from Emails ##
def email2FeatureVector(raw_email, vocab_dict):
    n = len(vocab_dict)
    result = np.zeros((n,1))
    vocab_indices = email2VocabIndices(raw_email, vocab_dict )
    for idx in vocab_indices:
        result[idx] = 1
    return result

res = email2FeatureVector(ham1, all_dict)

train = scipy.io.loadmat("d:/data/ex6/spamTrain.mat")
test = scipy.io.loadmat("d:/data/ex6/spamTest.mat")

x_train, y_train = train['X'], train['y']
x_test, y_test = test['Xtest'], test['ytest']

linear_svm = svm.SVC(C=1.0, kernel='linear')
linear_svm.fit(x_train, y_train)

y_hat = linear_svm.predict(x_test)
def accuracy(y_hat, y):
    y_hat = y_hat.reshape(len(y_hat), 1)
    acc = float(len(y[np.equal(y_hat, y)]) / len(y))
    return acc 

print(accuracy(y_hat, y_test)) # 0.978





















