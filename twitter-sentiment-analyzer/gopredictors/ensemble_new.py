import re
import csv

import random

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFpr
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import TweetTokenizer
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
from nltk import pos_tag


STOPWORDS = [u'i',u'me',u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself',u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her',u'hers'
,u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those'
,u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if'
,u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above'
,u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how'
,u'all', u'any',u'both', u'each', u'few', u'more',u'most',u'other', u'some', u'such', u'only', u'own', u'same', u'so', u'than', u'too', u'very',u'can', u'will', u'just',u'should'
u'now',u'rt']
#gloable for feature extraction
FEATURELIST = []




#######################################original simple demo functions
#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end
######################################original simple demo functions ends



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
# def tokenize(tweet):
#     featureVector = []
#     tweet = processTweet(tweet)
#     words = tweet.split()
#     for w in words:
#         #replace two or more with two occurrences
#         w = replaceTwoOrMore(w)
#         match_re = re.search('(\w|\')+',w)
#         if match_re == None:
#             continue
#         w = match_re.group()
#         if w !='_':
#             featureVector.append(w.lower())
#     return featureVector
def tokenize(tweet):
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(tweet)
    cleanedTks = []
    for tk in tokens:
        tk = tk.lower()
        if tk[0] == '@':
            continue
        elif tk =='':
            continue
        elif tk[0] == '#' and len(tk)!=1:
            cleanedTks.append(tk[1:])
        elif re.match("^((http://)|(https://)){0,1}(\w+\.)+\w+[\/\w\:]*$", tk):
            continue
        elif re.match("^(\w|')+$", tk) != None and len(tk) > 1:
            if re.match("^\d+$", tk) != None:
                continue
            else:
                cleanedTks.append(tk)
    return cleanedTks


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
    global STOPWORDS

    FEATURELIST = []
    featureDict = {}
    tweets = []
    for row in trainSet:
        sentiment = row[1]
        tweetTokens = row[0]
        for each in tweetTokens:
            if featureDict.has_key(each):
                featureDict[each] += 1
            else:
                featureDict[each] = 1

    for key in featureDict:
        if featureDict[key] > 1:
            FEATURELIST.append(key)


    train_data = []
    train_target = []

    for train_point in trainSet:
        train_data.append(extract_features(train_point[0]))
        train_target.append(train_point[1])


    #feature_selector = SelectKBest(chi2, k = 8000)
    feature_selector = SelectFpr(chi2, alpha = 0.2)
    train_data = feature_selector.fit_transform(train_data, train_target)


    clf1  = BernoulliNB()
    clf2 = LogisticRegression(C=0.9)
    clf3 = KNeighborsClassifier()
    #model3 = SVC(kernel='linear', probability=True )

    model = VotingClassifier(estimators=[('bnb', clf1), ('lr', clf2), ('knn', clf3)], voting='soft', weights=[1, 1, 1])
    model.fit(train_data, train_target)

    stdWdDict ={}
    for each in STOPWORDS:
        stdWdDict[each] = 1

    test_data = []
    test_target = []
    test_text = []
    for test_point in testSet:
        test_data.append(extract_features(test_point[0]))
        test_target.append(test_point[1])
        testWithoutStpWd = []
        for word in test_point[0]:
            if pos_tag([word])[0][1] == 'PRP' or stdWdDict.has_key(word):
                continue
            else:
                testWithoutStpWd.append(word)
        test_text.append(' '.join(testWithoutStpWd))
        #test_text.append(' '.join(test_point[0]))

    test_data = feature_selector.transform(test_data)
    predict = model.predict_proba(test_data)
    #predict = model.predict(test_data)

    total = 0
    correct = 0
    for i in range(0, len(predict)):
        vad_res = vaderSentiment(test_text[i])
        total += 1
        pos_prob =  predict[i][1] +vad_res['pos']
        neg_prob =  predict[i][0] +vad_res['neg']
        if pos_prob - neg_prob > 0:
            pred_tag = '|Positive|'
        #elif neg_prob -pos_prob > 0.5:
        elif neg_prob - pos_prob > 0.2:
            pred_tag = '|Negative|'
        else:
            total -=1
            continue

        if str(pred_tag).lower() == str(test_target[i]).lower():
            correct += 1
        else:
            print '=================================================='
            print  vad_res['neg'], vad_res['neu'], vad_res['pos']
            print predict[i], test_target[i]

    accuracy = (correct / float(total)) * 100

    print "Accuracy : %.2f" % accuracy
    print  "(" + str(correct) + "/" + str(total) + ")"
    return total, correct




def preprocess(data_file, neutral):
    rawTweets = csv.reader(open(data_file, 'rU'))
    tweets = []
    for each in rawTweets:
        if neutral == False:
            if each[1] !='|Positive|' and each[1] !='|Negative|':
                continue
        tweets.append(each)
    cleanTweets = []
    for row in tweets:
        sentiment = row[1]
        tweet = row[0]
        tokenizedTweet = tokenize(tweet)
        cleanTweets.append((tokenizedTweet, sentiment))
    return cleanTweets


if __name__=='__main__':
    # data_file = 'data/gop/august/AllLabeledMini.csv'
    # cleanTweets = preprocess(data_file)
    # random.shuffle(cleanTweets)
    # cross_validation(cleanTweets, classifyAlgo)

    train_file = '../data/gop/august/august_full_active_form.csv'
    test_file = '../data/gop/march/combined_sample_unique_quote_form.csv'

    trainTweets = preprocess(train_file, False)
    testTweets = preprocess(test_file,True)

    # cleanTweets = trainTweets + testTweets
    # random.shuffle(cleanTweets)
    # cross_validation(cleanTweets, classifyAlgo)
    classifyAlgo(trainTweets, testTweets)


