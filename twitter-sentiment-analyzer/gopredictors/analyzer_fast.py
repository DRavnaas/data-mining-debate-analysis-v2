import re
import csv
import pprint
import nltk.classify

import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

#gloable for feature extraction
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
    #tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('@[^\s]+','\1',tweet)

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
    stopWords = getStopWordList('data/gop/stopwords.txt')
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
    tknizer = RegexpTokenizer(r'\w+')
    #words = tweet.split()
    words = tknizer.tokenize(tweet)
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        #w = w.strip('\'"?,.')
        if len(w) >1 :
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
            cleanTweets.append((extract_features(tokensWithoutStopWords), sentiment))
        else:
            cleanTweets.append((extract_features(tokenizedTweet), sentiment))


    return cleanTweets



def extract_features(words):
    global NGRAMSFLAG
    if NGRAMSFLAG:
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(BigramAssocMeasures.pmi, n=10)
        return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
    else:
        return dict([(word, True) for word in words])


def NaiveBayes(trainSet, testSet):
    global DROP_NEUTRAL
    global PRINT_INCORRECT_PREDICTIONS

    NBClassifier = nltk.NaiveBayesClassifier.train(trainSet)

    correct = 0
    total =0
    for row in testSet:
        testTweet = row[0]
        predicted_sentiment = str(NBClassifier.classify(testTweet)).lower()
        actual_sentiment = str(row[1]).lower()

        if DROP_NEUTRAL:
            if predicted_sentiment != '|neutral|':
                total +=1
                if actual_sentiment == predicted_sentiment:
                    correct +=1
        else:
            total +=1
            if actual_sentiment == predicted_sentiment:
                correct +=1
            else:
                if PRINT_INCORRECT_PREDICTIONS:
                    tweet = ""
                    for word in testTweet:
                        tweet = tweet + word + " "

                    print ("Tweet: %s  \n actual_sentiment: %s   predicted_sentiment: %s" \
                            %(tweet,actual_sentiment,predicted_sentiment))

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
    NGRAMSFLAG = True
    DROP_NEUTRAL = True
    PRINT_INCORRECT_PREDICTIONS = False

    CROSSVALIDFLAG = False


    if  CROSSVALIDFLAG:
        data_file   = '../data/gop/august/august_full_form.csv'
        data = getCleanTweets(data_file)
        cross_validation(data, NaiveBayes)

    else:
        train_file  = '../data/gop/august/august_full_active_form_manip.csv'
        test_file   = '../data/gop/march/combined_sample_unique_quote_form.csv'

        trainTweets = getCleanTweets(train_file)
        testTweets  = getCleanTweets(test_file)

        NaiveBayes(trainTweets, testTweets)
