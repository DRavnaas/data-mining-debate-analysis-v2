import re
import csv
import pprint
import nltk.classify

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
        #strip punctuation
        w = w.strip('\'"?,.')
        featureVector.append(w.lower())
    return featureVector


def preprocess(data,removeStopWordsFlag):
    cleanTweets = []
    for row in data:
        sentiment = row[1]
        tweet = row[2]
        processedTweet = processTweet(tweet)
        tokenizedTweet = tokenize(processedTweet)
        if removeStopWordsFlag:
            tokensWithoutStopWords = removeStopWords(tokenizedTweet)
            cleanTweets.append((tokensWithoutStopWords, sentiment))
        else:
            cleanTweets.append((tokenizedTweet, sentiment))


    return cleanTweets


#start extract_features
# def extract_features(tweet):
#      global FEATURELIST
#      tweet_words = set(tweet)
#      features = {}
#      for word in FEATURELIST:
#         features['contains(%s)' % word] = (word in tweet_words)
#      return features
# # #end

def extract_features(words):
    return dict([(word, True) for word in words])

def NaiveBayes(trainSet, testSet):
    global FEATURELIST
    tweets= []
    for row in trainSet:
        sentiment = row[1]
        featureDict = row[0]
        FEATURELIST.extend(featureDict)
        tweets.append((extract_features(featureDict), sentiment));

    FEATURELIST = list(set(FEATURELIST))

    # training_set = nltk.classify.util.apply_features(extract_features, trainSet)
    #NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

    NBClassifier = nltk.NaiveBayesClassifier.train(tweets)

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

def getCleanTweets(data_file,removeStpWordsFlag):
    rawTweets = csv.reader(open(data_file, 'rU'))
    cleanTweets = preprocess(rawTweets,removeStpWordsFlag)
    return  cleanTweets

if __name__=='__main__':
    featureList = []
    removeStpWordsFlag = False
    crossValdFlag = False

    if  crossValdFlag:
        data_file   = 'data/gop/august/august_full_form.csv'
        cleanTweets = getCleanTweets(data_file,removeStpWordsFlag)
        cross_validation(cleanTweets, NaiveBayes)

    else:
        train_file  = 'data/gop/august/august_full_form.csv'
        test_file   = 'data/gop/march/combined_sample_unique_form.csv'

        trainTweets = getCleanTweets(train_file,removeStpWordsFlag)
        testTweets  = getCleanTweets(test_file,removeStpWordsFlag)

        NaiveBayes(trainTweets, testTweets)