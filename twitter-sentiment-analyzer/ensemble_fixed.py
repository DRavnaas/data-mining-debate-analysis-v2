import re
import csv


from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFpr
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import TweetTokenizer
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
from nltk import pos_tag
from sklearn.metrics import confusion_matrix
import random

STOPWORDS = [u'i',u'me',u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself',u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her',u'hers'
,u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those'
,u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if'
,u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above'
,u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how'
,u'all', u'any',u'both', u'each', u'few', u'more',u'most',u'other', u'some', u'such', u'only', u'own', u'same', u'so', u'than', u'too', u'very',u'can', u'will', u'just',u'should'
u'now',u'rt']

STOPWORDSDICT = None

#gloable for feature extraction
FEATURELIST = []

REPLACEDICT = {'kkktrump': 'hate trump', 'dumptrump': 'hate trump', 'stoptrump': 'hate trump',
               'nevertrump': 'hate trump', 'hesstilltedcruz': 'hate cruz', 'yeahthattedcruz': 'hate Cruz',
               'kkk': 'racist', 'nazi': 'evil', 'fascist': 'evil', 'furher': 'evil', 'idiot': 'stupid',
               'bigot': 'racist', 'bigotry': 'racist', 'vile': 'evil', 'lame': 'stupid', 'trump2016': 'love trump',
               'alwaystrump': 'love trump', 'cruzcrew': 'love cruz',
               'trumpwins': 'love trump', 'kasich4us': 'love kasich', 'kasichcan': 'love Kasich',
               'kasich2016': 'love kasich'}


#helper functions

#remove extra continous occurance of one letter, like make huuuuuungry to huungry
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


#cross validation function, it takes a function "algo" as parameter,
#then call "algo" to run NB, anything can be defined in "algo"
def cross_validation(dataSet, algo):
    total=0
    correct=0
    total_target = []
    total_predict = []
    for i in range(0,5):
        trainSet, testSet = devideFullSet(dataSet,i)

        print "\n Fold %d" % (i+1)
        total_fold,correct_fold,target_fold,predict_fold= algo(trainSet,testSet)
        total = total + total_fold
        correct = correct + correct_fold
        total_target.extend(target_fold)
        total_predict.extend(predict_fold)

    accuracy =  (correct / float(total)) * 100

    print "-------------------------------------------"
    print confusion_matrix(total_target,total_predict,labels= ['Positive','Negative'])
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

# tweets tokenize function, nltk TweetTokenizer is used to tokenize.
# Then do further selection, @user, url,numbers and punctuation are removed
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
                tk = replaceTwoOrMore(tk)
                cleanedTks.append(tk)
    return cleanedTks

# using feature list to create feature vector for tweets
# feature list is like [a,b,c,d]
# feature vector if [True, False,True,False] for tweet "a c"
def extract_features(tweet):
    global FEATURELIST
    tweet_words = []
    tweet_words.extend(set(tweet))
    features = []
    for word in FEATURELIST:
        features.append(word in tweet_words)
    return features


def getFeatureList(trainSet):
    global FEATURELIST
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
def removeStopWords(words):

    featureVector = []
    for w in words:
        if STOPWORDSDICT.has_key(w):
            continue
        else:
            featureVector.append(w)
    return featureVector

def runLexicon(dataSet):
    global STOPWORDS
    global STOPWORDSDICT
    if STOPWORDSDICT == None:
        stdWdDict = {}
        for each in STOPWORDS:
            stdWdDict[each] = 1
        STOPWORDSDICT = stdWdDict
    ret= []
    for point in dataSet:
        testWithoutStpWd = []
        for word in point[0]:
            if pos_tag([word])[0][1] == 'PRP' or STOPWORDSDICT.has_key(word):
                continue
            else:
                if REPLACEDICT.has_key(word):
                    testWithoutStpWd.append(REPLACEDICT[word])
                else:
                    testWithoutStpWd.append(word)
        ret.append(vaderSentiment(' '.join(testWithoutStpWd)))
    return ret



def classifyAlgo(trainSetNB, testSet):
    getFeatureList(trainSetNB)
    trainNB_data = []
    trainNB_target = []

    for train_point in trainSetNB:
        trainNB_data.append(extract_features(train_point[0]))
        trainNB_target.append(train_point[1])

    feature_selector = SelectFpr(chi2, alpha = 0.2)
    trainNB_data = feature_selector.fit_transform(trainNB_data, trainNB_target)

    clf1 = BernoulliNB()
    clf2 = LogisticRegression(C=0.9)
    clf3 = KNeighborsClassifier()
    model1 = VotingClassifier(estimators=[('lr', clf1), ('mnb', clf2), ('knn', clf3)], voting='soft', weights=[1, 1, 1])
    model1.fit(trainNB_data, trainNB_target)


    test_data = []
    test_target = []
    test_lexi = []
    for test_point in testSet:
        test_data.append(extract_features(test_point[0]))
        test_target.append(test_point[1])
    test_lexi = runLexicon(testSet)

    test_data = feature_selector.transform(test_data)
    #predict = model1.predict_proba(test_data)
    predict = model1.predict(test_data)

    total = 0
    correct = 0
    for i in range(0, len(predict)):
        pred_tag = predict[i]
        total += 1
        # vad_res = test_lexi[i]
        # pos_prob = predict[i][1] + vad_res['pos']
        # neg_prob = predict[i][0] + vad_res['neg']
        # if pos_prob - neg_prob > 0:
        #     pred_tag = 'Positive'
        # # elif neg_prob -pos_prob > 0.5:
        # elif neg_prob - pos_prob > 0:
        #     pred_tag = 'Negative'
        # else:
        #     if pos_prob - neg_prob > 0:
        #         pred_tag = 'Positive'
        #     else:
        #         pred_tag = 'Negative'
        #predict_tag = svmModel.predict([list(predict[i]) + [test_lexi[i]['pos'], test_lexi[i]['neu'], test_lexi[i]['neg']]])[0] #,test_lexi[i]['compound']
        if str(pred_tag).lower() == str(test_target[i]).lower():
            correct += 1
        else:
            print pred_tag

    accuracy = (correct / float(total)) * 100

    print confusion_matrix(test_target,predict,labels= ['Positive','Negative'])
    print "Accuracy : %.2f" % accuracy
    print  "(" + str(correct) + "/" + str(total) + ")"
    return total, correct,test_target,predict

def fileForROC(trainSetNB, testSet):
    getFeatureList(trainSetNB)
    trainNB_data = []
    trainNB_target = []

    for train_point in trainSetNB:
        trainNB_data.append(extract_features(train_point[0]))
        trainNB_target.append(train_point[1])

    feature_selector = SelectFpr(chi2, alpha=0.2)
    trainNB_data = feature_selector.fit_transform(trainNB_data, trainNB_target)

    # model1 = BernoulliNB()
    # model1.fit(trainNB_data, trainNB_target)
    clf1 = BernoulliNB()
    clf2 = LogisticRegression(C=0.9)
    clf3 = KNeighborsClassifier()
    model1 = VotingClassifier(estimators=[('lr', clf1), ('mnb', clf2), ('knn', clf3)], voting='soft', weights=[1, 1, 1])
    model1.fit(trainNB_data, trainNB_target)

    test_data = []
    test_target = []
    for test_point in testSet:
        test_data.append(extract_features(test_point[0]))
        test_target.append(test_point[1])

    test_data = feature_selector.transform(test_data)
    predict = model1.predict_proba(test_data)

    ofile = open('rocfile.csv', "w")
    writer = csv.writer(ofile, lineterminator='\n')
    writer.writerow(
        ['', 'id', 'tweet_id', 'candidate', 'tweet_created', 'tweet_location', 'user_timezone',  'sentiment',
         'predictedLabel'])

    for i in range(0, len(predict)):
        # predict_tag = svmModel.predict([list(predict[i]) + [test_lexi[i]['pos'], test_lexi[i]['neu'], test_lexi[i]['neg']]])[0] #,test_lexi[i]['compound']
        if predict[i][0] >= predict[i][1]:
            pred_tag = 'Negative'
        else:
            pred_tag = 'Positive'
        # if pos_prob - neg_prob > 0.2:
        #     pred_tag = '3'
        # # elif neg_prob -pos_prob > 0.5:
        # elif neg_prob - pos_prob > 0.6:
        #     pred_tag = '1'
        # else:
        #     pred_tag = '2'
        # if pred_tag == 'Positive':
        #     num_tag = 3
        # elif pred_tag == 'Negative':
        #     num_tag = 1
        row = []
        row.extend(testSet[i][2][0:8])
        row.append(pred_tag)
        row.append(predict[i][0])
        row.append(predict[i][1])

        # row.append(predict[i][0])
        # row.append(test_lexi[i]['pos'])
        # row.append(test_lexi[i]['neu'])
        # row.append(test_lexi[i]['neg'])
        # row.append(test_lexi[i]['compound'])
        writer.writerow(row)

    ofile.close()


def labelAndOutput(trainSetNB, testSet):
    getFeatureList(trainSetNB)
    trainNB_data = []
    trainNB_target = []

    for train_point in trainSetNB:
        trainNB_data.append(extract_features(train_point[0]))
        trainNB_target.append(train_point[1])

    feature_selector = SelectFpr(chi2, alpha = 0.2)
    trainNB_data = feature_selector.fit_transform(trainNB_data, trainNB_target)

    #model1 = BernoulliNB()
    #model1.fit(trainNB_data, trainNB_target)
    clf1 = BernoulliNB()
    clf2 = LogisticRegression(C=0.9)
    clf3 = KNeighborsClassifier()
    model1 = VotingClassifier(estimators=[('lr', clf1), ('mnb', clf2), ('knn', clf3)], voting='soft', weights=[1, 1, 1])
    model1.fit(trainNB_data, trainNB_target)


    test_data = []
    test_target = []
    test_lexi = []
    for test_point in testSet:
        test_data.append(extract_features(test_point[0]))
        test_target.append(test_point[1])
    test_lexi = runLexicon(testSet)

    test_data = feature_selector.transform(test_data)
    predict = model1.predict(test_data)

    ofile = open('simpleProb.csv', "w")
    writer = csv.writer(ofile, lineterminator='\n')
    writer.writerow(
        ['', 'id', 'tweet_id', 'candidate', 'tweet_created','tweet_location', 'user_timezone','text','sentiment', 'predictedLabel'])

    for i in range(0, len(predict)):
        #predict_tag = svmModel.predict([list(predict[i]) + [test_lexi[i]['pos'], test_lexi[i]['neu'], test_lexi[i]['neg']]])[0] #,test_lexi[i]['compound']
        pred_tag = predict[i]
        # if pos_prob - neg_prob > 0.2:
        #     pred_tag = '3'
        # # elif neg_prob -pos_prob > 0.5:
        # elif neg_prob - pos_prob > 0.6:
        #     pred_tag = '1'
        # else:
        #     pred_tag = '2'
        row = []
        row.extend(testSet[i][1])
        row.append(pred_tag)

        # row.append(predict[i][0])
        # row.append(test_lexi[i]['pos'])
        # row.append(test_lexi[i]['neu'])
        # row.append(test_lexi[i]['neg'])
        # row.append(test_lexi[i]['compound'])
        writer.writerow(row)

    ofile.close()


def preprocess(data_file):
    global STOPWORDS
    global STOPWORDSDICT
    rawTweets = csv.reader(open(data_file, 'rU'))
    tweets = []
    for each in rawTweets:
        if each[1] != '|Positive|' and each[1] != '|Negative|':
            continue
        tweets.append(each)
    cleanTweets = []
    for row in tweets:
        sentiment = row[1]
        tweet = row[2]
        tokenizedTweet = tokenize(tweet)
        cleanTweets.append((tokenizedTweet, sentiment))
    return cleanTweets

def preprocessTrain(data_file, neutral):
    rawTweets = csv.reader(open(data_file, 'rU'))
    tweets = []
    for each in rawTweets:
        if neutral == False:
            if each[1] != '|Positive|' and each[1] != '|Negative|':
                continue
        tweets.append(each)
    cleanTweets = []
    for row in tweets:
        sentiment = row[1]
        tweet = row[0]
        tokenizedTweet = tokenize(tweet)
        cleanTweets.append((tokenizedTweet, sentiment))
    return cleanTweets

def preprocessTest(data_file):
    rawTweets = csv.reader(open(data_file, 'rU'))
    cleanTweets = []
    for row in rawTweets:
        if row[0]=='':
            continue
        tweet = row[7]
        tokenizedTweet = tokenize(tweet)
        cleanTweets.append((tokenizedTweet, row[0:7]))
    return cleanTweets

def preprocessSimple(data_file):
    rawTweets = csv.reader(open(data_file, 'rU'))
    cleanTweets = []
    for row in rawTweets:
        if row[0]=='':
            continue
        tweet = row[7]
        tokenizedTweet = tokenize(tweet)
        sentiment = row[9]
        cleanTweets.append((tokenizedTweet, sentiment))
    return cleanTweets

def preprocessAugAndMarTrain(data_file):
    rawTweets = csv.reader(open(data_file, 'rU'))
    tweets = []
    for each in rawTweets:
        if each[0]=='':
            continue
        if each[5] != 'Positive' and each[5] != 'Negative':
            continue
        tweets.append(each)
    cleanTweets = []
    for row in tweets:
        sentiment = row[5]
        tweet = row[8]
        tokenizedTweet = tokenize(tweet)
        cleanTweets.append((tokenizedTweet, sentiment))
    return cleanTweets

def preprocessROCInput(data_file):
    rawTweets = csv.reader(open(data_file, 'rU'))
    tweets = []
    for each in rawTweets:
        if each[0]=='':
            continue
        if each[5] != 'Positive' and each[5] != 'Negative':
            continue
        tweets.append(each)
    cleanTweets = []
    for row in tweets:
        sentiment = row[5]
        tweet = row[8]
        tokenizedTweet = tokenize(tweet)
        cleanTweets.append((tokenizedTweet, sentiment,row))
    return cleanTweets

def preprocessAugAndMarTest(data_file):
    rawTweets = csv.reader(open(data_file, 'rU'))
    tweets = []
    for each in rawTweets:
        if each[0]=='':
            continue
        tweets.append(each)
    cleanTweets = []
    for row in tweets:
        tweet = row[8]
        tokenizedTweet = tokenize(tweet)
        cleanTweets.append((tokenizedTweet, row))
    return cleanTweets

if __name__=='__main__':
    data_file = 'AugAndMarchLabeledQuote.csv'
    #data_file = 'data/gop/august/august_candidates_form.csv'
    #cleanTweets = preprocess(data_file)
    cleanTweets = preprocessAugAndMarTrain(data_file)
    random.shuffle(cleanTweets)
    cross_validation(cleanTweets, classifyAlgo)

    #data_file = 'AugAndMarchLabeledQuote.csv'
    #cleanTweets = preprocessROCInput(data_file)
    #random.shuffle(cleanTweets)
    #testTweets = cleanTweets[0:2213]
    #trainTweets = cleanTweets[2213:]
    #fileForROC(trainTweets, testTweets)


    #train_file = 'data/gop/august/ttest_neutral.csv'
    #train_file = 'data/gop/august/AllLabeledQuoteMini_form.csv'
    #test_file = 'data/gop/august/combined_sample_form.csv'
    #train_file = 'data/gop/august/unlabeled.csv'
    #rainTweets = preprocess(train_file, False)
    #testTweets = preprocess(test_file,True)

    #train_file1 = 'AugAndMarchLabeledQuote.csv'
    #train_file2 = 'data/gop/august/combined_sample_form.csv'
    #test_file = 'UnlabeledMarchForR.csv'
    #test_file = 'UnlabeledMarchForR.csv'
    #test_file = 'AugAndMarchLabeledQuote.csv'
    #trainTweets = preprocessAugAndMarTrain(train_file1)
    #trainTweets2 = preprocessTrain(train_file2, False)
    #testTweets= preprocessTest(test_file)
    #testTweets = preprocessAugAndMarTest(test_file)
    #labelAndOutput(trainTweets, testTweets)


    # train_file1 = 'AugAndMarchLabeledQuote.csv'
    # test_file = 'file_confusion.csv'
    # trainTweets = preprocessAugAndMarTrain(train_file1)
    # testTweets = preprocessSimple(test_file)
    # total_fold, correct_fold, target_fold, predict_fold = classifyAlgo(trainTweets, testTweets)
    #
    # print "-------------------------------------------"
    # print confusion_matrix(target_fold, predict_fold, labels=['Positive', 'Negative'])




