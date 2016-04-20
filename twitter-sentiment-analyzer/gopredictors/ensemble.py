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

from sklearn.linear_model import LogisticRegression

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

    for train_point in trainSet:
        train_data.append(extract_features(train_point[0]))
        train_target.append(train_point[1])



    model1  = MultinomialNB()
    model2 = LogisticRegression()
    #model2 = SVC(kernel='linear', probability=True )

    model1.fit(train_data, train_target)
    model2.fit(train_data, train_target)
    sig = SentimentIntensityAnalyzer()

    # test_data = []
    # test_target = []
    # for test_point in testSet:
    #     test_data.append(extract_features(test_point[0]))
    #     test_target.append(test_point[1])
    #
    # #test_data = feature_selector.transform(test_data)
    # predict = model.predict(test_data)

    total = 0
    correct = 0
    for test_point in testSet:
         total += 1
         testTweet = extract_features(test_point[0])
         m1_prob = model1.predict_proba([testTweet])[0]
         m2_prob = model2.predict_proba([testTweet])[0]
         vad_res = sig.polarity_scores("".join(test_point[0]))
         if vad_res['neg'] == vad_res['pos'] == 0:
             vad_prob = [0.5,0.5]
         else:
             vad_prob = [vad_res['neg']/(vad_res['neg']+vad_res['pos']), vad_res['pos']/(vad_res['neg']+vad_res['pos'])]
         m1_weight = 3
         m2_weight = 2
         vad_weight = 1
         combined_prob = [m1_prob[0]*m1_weight+m2_prob[0]*m2_weight+vad_prob[0]*vad_weight,m1_prob[1]*m1_weight+m2_prob[1]*m2_weight+vad_prob[1]*vad_weight ]
         if combined_prob[0] > combined_prob[1]:
             pred_class = '|Negative|'
         else:
             pred_class = '|Positive|'
         if str(pred_class).lower() == str(test_point[1]).lower():
             correct += 1

    # for i in range(0, len(predict)):
    #     total +=1
    #     if str(predict[i]).lower() == str(test_target[i]).lower():
    #         correct +=1

    accuracy =  (correct / float(total)) * 100

    print "Accuracy : %.2f" % accuracy
    print  "(" + str(correct) + "/" + str(total) + ")"
    return total,correct


def preprocess(data_file):
    rawTweets = csv.reader(open(data_file, 'rU'))
    tweets = []
    for each in rawTweets:
        if each[1] =='|Neutral|':
            continue
        tweets.append(each)

    cleanTweets = []
    for row in tweets:
        sentiment = row[1]
        tweet = row[2]
        tokenizedTweet = tokenize(tweet)
        cleanTweets.append((tokenizedTweet, sentiment))
    return cleanTweets

if __name__=='__main__':
    data_file = 'data/gop/august/august_candidates_form.csv'
    cleanTweets = preprocess(data_file)
    random.shuffle(cleanTweets)
    cross_validation(cleanTweets, classifyAlgo)