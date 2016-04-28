import re
import csv

from nltk.tokenize import TweetTokenizer
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
from nltk import pos_tag
from nltk.corpus import stopwords

#simplified stopwords list
STOPWORDS = [u'i',u'me',u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself',u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her',u'hers'
,u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those'
,u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if'
,u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above'
,u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how'
,u'all', u'any',u'both', u'each', u'few', u'more',u'most',u'other', u'some', u'such', u'only', u'own', u'same', u'so', u'than', u'too', u'very',u'can', u'will', u'just',u'should'
u'now',u'rt']
#gloable variable for feature extraction
FEATURELIST = []
#replacement dictionary for hashtags
REPLACEDICT = {'kkktrump': 'hate trump', 'dumptrump': 'hate trump', 'stoptrump': 'hate trump',
               'nevertrump': 'hate trump', 'hesstilltedcruz': 'hate cruz', 'yeahyhatyedcruz': 'hate Cruz',
               'kkk': 'racist', 'nazi': 'evil', 'fascist': 'evil', 'furher': 'evil', 'idiot': 'stupid',
               'bigot': 'racist', 'bigotry': 'racist', 'vile': 'evil', 'lame': 'stupid', 'trump2016': 'love trump',
               'alwaystrump': 'love trump', 'cruzcrew': 'love cruz',
               'trumpwins': 'love trump', 'kasich4us': 'love kasich', 'kasichcan': 'love Kasich',
               'kasich2016': 'love kasich'}

#remove continue occurance of a letter in a word, if more than 2
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

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

#tokenize function based on nltk package
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


# For comparison, this function predicts process tweets and unprocessed tweets.
# It first does process for each tweet and generate 2 lists of test data, one for
# processed tweets and the other for unprocessed.
# Stopwords, puncuations, preposition words are removed.
# hashtags with sentiment are replaced with sentiment words
#input format listof (tokenized tweet, sentiment, raw tweet)
def classifyAlgo(testSet):
    global STOPWORDS
    STOPWORDS = stopwords.words('english')
    stdWdDict ={}
    for each in STOPWORDS:
        stdWdDict[each] = 1
    test_data = []
    test_target = []
    test_text = []
    for test_point in testSet:
        test_target.append(test_point[1])
        testWithoutStpWd = []
        for word in test_point[0]:
             if pos_tag([word])[0][1] == 'PRP' or stdWdDict.has_key(word) :
                  continue
             else:
                 if REPLACEDICT.has_key(word):
                     testWithoutStpWd.append(REPLACEDICT[word])
                 else:
                     testWithoutStpWd.append(word)
        test_text.append(' '.join(testWithoutStpWd))
        test_data.append(test_point[2])

    total =0
    correct =0
    for i in range(0, len(test_text)):
        total += 1
        vad_res = vaderSentiment(test_text[i])
        tw_res = vaderSentiment(test_data[i])
        if  vad_res['neg']<vad_res['pos']:
            pred_tag = '|Positive|'
        else:
            pred_tag = '|Negative|'
        if pred_tag ==  test_target[i]:
            correct += 1
        print '=================================================='
        print test_text[i]
        print test_data[i]
        print  vad_res['neg'], vad_res['neu'], vad_res['pos']
        print  tw_res['neg'], tw_res['neu'], tw_res['pos']
        print  test_target[i]
    print correct * 1.0/ total


# process input data, droppping neutral and tokenize.
def preprocess(data_file):
    rawTweets = csv.reader(open(data_file, 'rU'))
    tweets = []
    for each in rawTweets:
        tweets.append(each)
    cleanTweets = []
    for row in tweets:
        sentiment = row[1]
        tweet = row[2]
        if sentiment=='|Neutral|':
            continue
        tokenizedTweet = tokenize(tweet)
        cleanTweets.append((tokenizedTweet, sentiment, tweet))
    return cleanTweets


if __name__=='__main__':
    file = 'data/august_full_form.csv'
    tweets = preprocess(file)
    classifyAlgo(tweets)


