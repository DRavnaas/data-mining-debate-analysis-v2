import re
import csv
import pprint
import nltk.classify
import random

import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

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

      features = {}
      for word in FEATURELIST:
         features[word] = (word in tweet_words)
      return features
 #end

def getTotalCount(trainSet):
    pos_count = 0
    neg_count = 0
    neu_count = 0
    for each in trainSet:
        if each[1] == '|Positive|':
            pos_count += 1
        elif each[1] == '|Negative|':
            neg_count += 1
        elif each[1] == '|Neutral|':
            neu_count += 1
    print pos_count,neg_count,neu_count
    return pos_count,neg_count,neu_count

def selectionFunc1(count):
    if count['|Positive|'] == 0 or count['|Negative|'] == 0:
        c1 = (count['|Positive|'] + 1) * 1.0 / (count['|Negative|'] + 1)
    else:
        c1 = count['|Positive|'] * 1.0 / count['|Negative|']
    if c1 < 1:
        c1 = 1 / c1

    if count['|Neutral|'] == 0 or count['|Negative|'] == 0:
        c2 = (count['|Neutral|'] + 1) * 1.0 / (count['|Negative|'] + 1)
    else:
        c2 = count['|Neutral|'] * 1.0 / count['|Negative|']
    if c2 < 1:
        c2 = 1 / c2

    if count['|Neutral|'] == 0 or count['|Positive|'] == 0:
        c3 = (count['|Neutral|'] + 1) * 1.0 / (count['|Positive|'] + 1)
    else:
        c3 = count['|Neutral|'] * 1.0 / count['|Positive|']
    if c3 < 1:
        c3 = 1 / c3

    m = max(c1, c2, c3)

def selectionFunc2(feature_count,total_pos_count,total_neg_count,total_neu_count):
    #              pos,neg,neu
    # not contain
    # contain
    totalData = total_pos_count+total_neg_count+total_neu_count
    pt = (feature_count['|Positive|'] + feature_count['|Negative|'] + feature_count['|Neutral|'])*1.0/totalData
    p_not_t = 1-pt
    p_pos = total_pos_count * 1.0 / totalData
    p_neg = total_neg_count * 1.0 / totalData
    p_neu = total_neu_count * 1.0 / totalData
    N = [[0,0],[0,0],[0,0]]
    E = [[0,0],[0,0],[0,0]]
    N[0][0] =  total_pos_count - feature_count['|Positive|']
    N[0][1] = feature_count['|Positive|']
    N[1][0] = total_neg_count - feature_count['|Negative|']
    N[1][1] = feature_count['|Negative|']
    N[2][0] = total_neu_count - feature_count['|Neutral|']
    N[2][1] = feature_count['|Neutral|']
    E[0][0] = totalData*p_pos*p_not_t
    E[0][1] = totalData * p_pos * pt
    E[1][0] = totalData * p_neg * p_not_t
    E[1][1] = totalData * p_neg * pt
    E[2][0] = totalData * p_neu * p_not_t
    E[2][1] = totalData * p_neu * pt
    chi = 0
    for i in range(0,2):
        for j in range(0,3):
            chi += (pow(N[j][i] - E[j][i],2))*1.0/E[j][i]
    return chi

def selectFeatures(trainSet, featureList):
    global NGRAMSFLAG
    sortedList = []
    total_pos_count,total_neg_count,total_neu_count  = getTotalCount(trainSet)
    int_count = 0
    for feature in featureList:
        int_count += 1
        count = dict([('|Positive|',0),('|Negative|',0),('|Neutral|',0)])
        for each in trainSet:
            tweetTokens = each[0]
            if NGRAMSFLAG == True:
                bigram_finder = BigramCollocationFinder.from_words(tweetTokens)
                bigrams = bigram_finder.nbest(BigramAssocMeasures.pmi, n=20)
            if (feature in tweetTokens) or (NGRAMSFLAG and feature in bigrams):
                count[each[1]] += 1
        if count['|Positive|'] + count['|Negative|'] +count['|Neutral|'] == 1:
            continue
        #m = selectionFunc1(count)
        m = selectionFunc2(count,total_pos_count,total_neg_count,total_neu_count)
        sortedList.append((feature, m))
        print int_count, feature, m
    #topN  = 12000
    sortedList = sorted(sortedList, key=lambda tup: tup[1])
    sortedList = sortedList[::-1]
    topN = 0
    # for i in range(0,len(featureList)):
    #     if sortedList[i][1] > 2.71:
    #         topN+=1
    #     else:
    #         break
    #topN = len(featureList)
    topN = 12000
    return [sortedList[i][0] for i in range(0,topN)]


def NaiveBayes(trainSet, testSet):
    global FEATURELIST
    global NGRAMSFLAG
    FEATURELIST = []
    tweets= []
    for row in trainSet:
        sentiment = row[1]
        featureDict = row[0]
        FEATURELIST.extend(featureDict)
        if NGRAMSFLAG == True:
            bigram_finder = BigramCollocationFinder.from_words(featureDict)
            bigrams = bigram_finder.nbest(BigramAssocMeasures.pmi, n=20)
            FEATURELIST.extend(bigrams)


    FEATURELIST = list(set(FEATURELIST))

    FEATURELIST = selectFeatures(trainSet, FEATURELIST)

    training_set = nltk.classify.util.apply_features(extract_features, trainSet)
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

    correct = 0
    total =0
    for row in testSet:
        total +=1
        testTweet = row[0]
        predicted_sentiment = NBClassifier.classify(extract_features(testTweet))
        actual_sentiment = row[1]
        if str(actual_sentiment).lower() == str(predicted_sentiment).lower():
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
    NGRAMSFLAG = True

    if CROSSVALIDFLAG:
        data_file = 'data/gop/august/august_full_form.csv'
        cleanTweets = getCleanTweets(data_file)
        random.shuffle(cleanTweets)
        cross_validation(cleanTweets, NaiveBayes)

    else:
        train_file = 'data/gop/august/august_full_form.csv'
        test_file = 'data/gop/march/combined_sample_unique_form.csv'

        trainTweets = getCleanTweets(train_file)
        testTweets = getCleanTweets(test_file)

        NaiveBayes(trainTweets, testTweets)