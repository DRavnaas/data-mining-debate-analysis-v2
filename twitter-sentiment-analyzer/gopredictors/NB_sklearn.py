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

from sklearn.linear_model import LogisticRegression

#gloable for feature extraction
FEATURELIST = []
REMOVESTPWORDSFLAG = None
CROSSVALIDFLAG = None
NGRAMSFLAG = None


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
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    #return list(set(stopwords.words('english')))
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

######################################original simple demo functions ends

def removeStopWords(words):
    featureVector = []
    stopWords = getStopWordList('data/feature_list/stopwords.txt')
    for w in words:
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector

#cross validation function, it takes a function "algo" as parameter,
#then call "algo" to run NB, anything can be defined in "algo"
def cross_validation(dataSet, algo):
    total=0
    correct=0

    for i in range(0,5):
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


def preprocess(data):
    global REMOVESTPWORDSFLAG
    cleanTweets = []
    for row in data:
        sentiment = row[1]
        tweet = row[2]
        processedTweet = processTweet(tweet)
        tokenizedTweet = tokenize(processedTweet)
        if REMOVESTPWORDSFLAG:
            tokensWithoutStopWords = removeStopWords(tokenizedTweet)
            cleanTweets.append((tokensWithoutStopWords, sentiment))
        else:
            cleanTweets.append((tokenizedTweet, sentiment))


    return cleanTweets


#start extract_features
def extract_features(tweet):
      global FEATURELIST
      global NGRAMSFLAG
      tweet_words = []
      tweet_words.extend(set(tweet))
      if NGRAMSFLAG == True:
          bigram_finder = BigramCollocationFinder.from_words(tweet)
          bigrams = bigram_finder.nbest(BigramAssocMeasures.pmi, n=20)
          tweet_words.extend(set(bigrams))

      features = []
      for word in FEATURELIST:
         features.append(word in tweet_words)
      return features
 #end



def classifyAlgo(trainSet, testSet):
    global FEATURELIST
    global NGRAMSFLAG
    FEATURELIST = []
    tweets= []
    for row in trainSet:
        sentiment = row[1]
        tweetTokens = row[0]
        FEATURELIST.extend(tweetTokens)
        if NGRAMSFLAG == True:
            bigram_finder = BigramCollocationFinder.from_words(tweetTokens)
            bigrams = bigram_finder.nbest(BigramAssocMeasures.pmi, n=20)
            FEATURELIST.extend(bigrams)

    FEATURELIST = list(set(FEATURELIST))

    train_data = []
    train_target = []

    for train_point in trainSet:
        train_data.append(extract_features(train_point[0]))
        train_target.append(train_point[1])

    #feature_selector = SelectKBest(chi2, k = selection_para)
    #feature_selector = SelectFpr(chi2, alpha = 0.1)
    #train_data = feature_selector.fit_transform(train_data, train_target)
    clf1 = LogisticRegression()
    #model = GaussianNB()
    clf2  = MultinomialNB()
    clf3 = SVC(kernel='linear', probability=True )
    model= VotingClassifier(estimators=[('lr', clf1), ('mnb', clf2), ('svm', clf3)], voting='hard')
    model.fit(train_data, train_target)

    test_data = []
    test_target = []
    for test_point in testSet:
        test_data.append(extract_features(test_point[0]))
        test_target.append(test_point[1])

    #test_data = feature_selector.transform(test_data)
    predict = model.predict(test_data)


    total = 0
    correct = 0
    for i in range(0, len(predict)):
        total +=1
        if str(predict[i]).lower() == str(test_target[i]).lower():
            correct +=1

    accuracy =  (correct / float(total)) * 100

    print "Accuracy : %.2f" % accuracy
    print  "(" + str(correct) + "/" + str(total) + ")"
    return total,correct


def getCleanTweets(data_file):
    rawTweets = csv.reader(open(data_file, 'rU'))
    cleanTweets = preprocess(rawTweets)
    return cleanTweets

if __name__=='__main__':
    REMOVESTPWORDSFLAG = False
    CROSSVALIDFLAG = True
    NGRAMSFLAG = False

    if CROSSVALIDFLAG:
        data_file = 'data/gop/august/august_candidates_form.csv'
        cleanTweets = getCleanTweets(data_file)
        random.shuffle(cleanTweets)
        cross_validation(cleanTweets, classifyAlgo)