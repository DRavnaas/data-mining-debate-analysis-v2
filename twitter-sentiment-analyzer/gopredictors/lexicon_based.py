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
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import TweetTokenizer
from nltk.corpus import sentiwordnet as swn

#gloable for feature extraction
NGRAMSFLAG = None
FEATURELIST = []
STOPWORDS = [u'i',u'me',u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself',u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her',u'hers'
,u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those'
,u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if'
,u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above'
,u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how'
,u'all', u'any',u'both', u'each', u'few', u'more',u'most',u'other', u'some', u'such', u'only', u'own', u'same', u'so', u'than', u'too', u'very',u'can', u'will', u'just',u'should'
u'now']
SENTIMENTDICT =None

#######################################original simple demo functions
#start replaceThreeOrMore
def replaceThressOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end


#tokenize tweets
def tokenize(tweet):
    tknzr = TweetTokenizer()
    tweet = tknzr.tokenize(tweet)
    return tweet


def processTokens(tokens):
    cleanedTks = []
    questionMkCount = 0
    exclamationMkCount = 0
    negationCount = 0
    url = []
    for tk in tokens:
        tk = tk.lower()
        if tk[0] == '@':
            continue
        elif tk[0] == '#':
            cleanedTks.append(tk[1:])
        elif tk.endswith("n't") or tk == 'no' or tk == 'not':
            negationCount += 1
        elif re.match("^((http://)|(https://)){0,1}(\w+\.)+\w+[\/\w\:]*$", tk):
            url.append(tk)
        elif  re.match("^(\w|')+$", tk) !=None and len(tk)>1:
            if re.match("^\d+$", tk) !=None:
                continue
            else:
                cleanedTks.append(tk)
        elif tk == '!':
            exclamationMkCount += 1
        elif tk == '?':
            questionMkCount += 1
    return cleanedTks, questionMkCount, exclamationMkCount, negationCount

def preprocess(data_file):
    rawTweets = csv.reader(open(data_file, 'rU'))
    tweets = []
    for each in rawTweets:
        #if each[1] =='|Neutral|':
            #continue
        tweets.append(each)

    cleanTweets = []
    count = 0;
    for row in tweets:
        sentiment = row[1]
        tweet = row[2]
        tokenizedTweet = tokenize(tweet)
        cleanedTks, questionMkCount, exclamationMkCount, negationCount = processTokens(tokenizedTweet)
        senti_scr = calSentimentScore(cleanedTks)
        if senti_scr ==0 and negationCount!=0:
            senti_scr += -1
        elif senti_scr ==0 and exclamationMkCount!=0:
            senti_scr += 1
        elif senti_scr == 0 and questionMkCount != 0:
            senti_scr += -1

        if senti_scr > 0:
            if sentiment == '|Positive|':
                count += 1
            else:
                print sentiment, senti_scr
        elif senti_scr < 0:
            if sentiment == '|Negative|':
                count += 1
            else:
                print sentiment, senti_scr
        elif senti_scr==0 and sentiment == '|Neutral|':
                count += 1
        else:
            print sentiment , senti_scr

        cleanTweets.append((cleanedTks, sentiment))
    print count,len(tweets)
    return cleanTweets

def readSenWdFile(sentiment_word_file):
    sentimentDict = {}
    f = open(sentiment_word_file)
    for each in f.readlines():
        each = each.rstrip('\n')
        word,score = each.split('\t')
        sentimentDict[word] = score
    return sentimentDict

def searchSynonyms(word):
    syn = []
    for each in list(swn.senti_synsets(word)):
        syn_word = each.synset.name().split('.')[0]
        if syn_word!=word:
            syn.append(syn_word)
    return syn

def calSentimentScore(tokens):
    score = 0
    for each in tokens:
        if SENTIMENTDICT.has_key(each):
            score += float(SENTIMENTDICT[each])
        else:
            synonyms = searchSynonyms(each)
            synonyms_score = 0
            syn_found =0
            for syn in synonyms:
                if SENTIMENTDICT.has_key(syn):
                    synonyms_score += float(SENTIMENTDICT[syn])
                    syn_found += 1
            if syn_found!=0:
                score += (synonyms_score *1.0)/ syn_found
    return score




if __name__=='__main__':
    NGRAMSFLAG = True
    data_file = 'data/gop/august/august_candidates_form.csv'
    sentiment_word_file = 'data/AFINN-111.txt'
    SENTIMENTDICT = readSenWdFile(sentiment_word_file)

    cleanTweets = preprocess(data_file)
