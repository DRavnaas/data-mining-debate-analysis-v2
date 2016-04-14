#import regex
import re
import csv
import pprint
import nltk.classify

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

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector    
#end

def extract_features(words):
    return dict([(word, True) for word in words])

#extract feature with bigram
#def word_feats(words, score_fn=BigramAssocMeasures.pmi, n=5):
    #bigram_finder = BigramCollocationFinder.from_words(words)
    #bigrams = bigram_finder.nbest(score_fn, n)
    #return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


#start extract_features
# def extract_features(tweet):
#     tweet_words = set(tweet)
#     features = {}
#     for word in featureList:
#         features['contains(%s)' % word] = (word in tweet_words)
#     return features
# #end


#Read the tweets one by one and process it
#inpTweets = csv.reader(open('data/augustW.csv', 'rb'), delimiter=',', quotechar='|')

trainTweets = csv.reader(open('data/gop/august/august_full_train_form.csv', 'rU'))
#trainTweets = csv.reader(open('data/gop/march/before_sample_form.csv', 'rU'))

#testTweets = csv.reader(open('data/gop/august/august_full_test_form.csv', 'rU'))
testTweets = csv.reader(open('data/gop/march/combined_sample_unique_form.csv', 'rU'))


stopWords = getStopWordList('data/gop/stopwords.txt')
#stopWords = []

count = 0;
featureList = []
tweets = []
for row in trainTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((extract_features(featureVector), sentiment));
#end loop

# Remove featureList duplicates
featureList = list(set(featureList))

# with open('data/gop/feature_list.txt','w') as f:
#
#     for feature in featureList:
#         f.write("%s\n" % feature)



# Generate the training set
#training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(tweets)
#NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
print NBClassifier.show_most_informative_features(100)


correct = 0
total =0
for row in testTweets:
# Test the classifier
    total +=1
    testTweet = row[1]
    processedTestTweet = processTweet(testTweet)
    predicted_sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
    actual_sentiment = row[0]

    if str(actual_sentiment).lower() == str(predicted_sentiment).lower():
        correct +=1
    else:
        if str(predicted_sentiment).lower() == "||":
            print "Error predicting: " + str(testTweet) + str(predicted_sentiment).lower()
        # else:
        #     print "Incorrect prediction: " + str(testTweet) + "\n actual :  " + str(actual_sentiment).lower() +\
        #            "     predicted:" + str(predicted_sentiment).lower()

    #print "actual sentiment = %s, predicted sentiment = %s, testTweet = %s,\n" % (row[0], predicted_sentiment,testTweet)


accuracy =  (correct / float(total)) * 100
#print accuracy

print "Accuracy : %.2f" % accuracy
print  "(" + str(correct) + "/" + str(total) + ")"
