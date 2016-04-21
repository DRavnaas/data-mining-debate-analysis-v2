import re
import csv

import random

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFpr
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

#gloable for feature extraction
FEATURELIST = []

#######################################original simple demo functions
#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    # #Convert www.* or https?://* to URL
    # tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    # #Convert @username to AT_USER
    # tweet = re.sub('@[^\s]+','AT_USER',tweet)
    # #Remove additional white spaces
    # tweet = re.sub('[\s]+', ' ', tweet)
    # #Replace #word with word
    # tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # #trim
    # tweet = tweet.strip('\'"')
    return tweet
#end
######################################original simple demo functions ends



#cross validation function, it takes a function "algo" as parameter,
#then call "algo" to run NB, anything can be defined in "algo"
def cross_validation(dataSet, algo):
    total=0
    correct=0

    for i in range(0,1):
        trainSet, testSet = devideFullSet(dataSet,i)

        print "\n Fold %d" % (i+1)
        total_fold,correct_fold = algo(trainSet, testSet)

        total = total + total_fold
        correct = correct + correct_fold

    accuracy =  (correct / float(total)) * 100

    print "-------------------------------------------"
    print "Average Accuracy : %.2f" % accuracy
    print  "(" + str(correct) + "/" + str(total) + ")"
    return accuracy

#divide a full dataset into train and test for 5-fold cross validation
#its input must be list, and a train data list and a test data list will be returned
#data point can be anything
def devideFullSet(list, seed):
    seed = seed % 5
    test = []
    train = []
    for i in range(0,len(list)):
        if i % 5 == seed:
            test.append(list[i])
        else:
            train.append(list[i])
    return train, test

#tokenize tweets
def tokenize(tweet):
    featureVector = []
    tweet = processTweet(tweet)
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        match_re = re.search('(\w|\')+',w)
        if match_re == None:
            continue
        w = match_re.group()
        if w !='_':
            featureVector.append(w.lower())
    return featureVector


#start extract_features
def extract_features(tweet):
      global FEATURELIST
      tweet_words = []
      tweet_words.extend(set(tweet))

      features = []
      for word in FEATURELIST:
         features.append(word in tweet_words)
      return features
 #end



def classifyAlgo(trainSet, testSet):
    global FEATURELIST
    FEATURELIST = []
    tweets= []
    for row in trainSet:
        sentiment = row[1]
        tweetTokens = row[0]
        FEATURELIST.extend(tweetTokens)

    FEATURELIST = list(set(FEATURELIST))

    train_data = []
    train_target = []
    test_data = []
    test_target = []

    for train_point in trainSet:
        train_data.append(extract_features(train_point[0]))
        train_target.append(train_point[1])

    for test_point in testSet:
        test_data.append(extract_features(test_point[0]))
        test_target.append(test_point[1])

    global DROP_NEUTRAL

    clf1 = MultinomialNB()
    clf2 = LogisticRegression()
    clf3 = BernoulliNB()


    clf1 = clf1.fit(train_data,train_target)
    clf2 = clf2.fit(train_data,train_target)
    clf3 = clf3.fit(train_data,train_target)

    eclf = VotingClassifier(estimators=[('mnb', clf1), ('lr', clf2), ('bnb', clf3)], voting='soft', weights=[2,2,1])

    eclf = eclf.fit(train_data,train_target)


    prediction_result = eclf.predict(test_data)

    total = 0
    correct = 0

    for i in range(0, len(prediction_result)):
       predicted_sentiment = str(prediction_result[i]).lower()
       actual_sentiment =   str(test_target[i]).lower()

       if predicted_sentiment != '|neutral|':
            total +=1

            if  predicted_sentiment == actual_sentiment:
                correct +=1

    accuracy =  (correct / float(total)) * 100

    print "Accuracy : %.2f" % accuracy
    print  "(" + str(correct) + "/" + str(total) + ")"

    compute_confusion_matrix(test_target,prediction_result)

    return total,correct

def compute_confusion_matrix(actual,predicted):
    cm = confusion_matrix(actual, predicted)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()

    label_names = list(set(actual))

    #plot_confusion_matrix(cm,label_names)



def plot_confusion_matrix(cm,label_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def preprocess(data_file):
    rawTweets = csv.reader(open(data_file, 'rU'))
    tweets = []
    for each in rawTweets:
        tweets.append(each)

    cleanTweets = []
    for row in tweets:
        sentiment = row[1]
        tweet = row[2]
        tokenizedTweet = tokenize(tweet)
        cleanTweets.append((tokenizedTweet, sentiment))
    return cleanTweets

if __name__=='__main__':
    CROSSVALIDFLAG = False

    if CROSSVALIDFLAG:
        data_file = '../data/gop/august/combined_sample_unique_quote_form'
        cleanTweets = preprocess(data_file)
        random.shuffle(cleanTweets)
        cross_validation(cleanTweets, classifyAlgo)
    else:
        train_file  = '../data/gop/august/august_full_form.csv'
        test_file   = '../data/gop/march/combined_sample_unique_quote_form.csv'

        trainTweets = preprocess(train_file)
        testTweets  = preprocess(test_file)

        classifyAlgo(trainTweets, testTweets)