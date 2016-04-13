library(RTextTools)
library(e1071)


NBUsingTextTools <- function()
{ pos_tweets =  rbind(
  c('I love this car', 'positive'),
  c('This view is amazing', 'positive'),
  c('I feel great this morning', 'positive'),
  c('I am so excited about the concert', 'positive'),
  c('He is my best friend', 'positive')
)

neg_tweets = rbind(
  c('I do not like this car', 'negative'),
  c('This view is horrible', 'negative'),
  c('I feel tired this morning', 'negative'),
  c('I am not looking forward to the concert', 'negative'),
  c('He is my enemy', 'negative')
)

test_tweets = rbind(
  c('feel happy this morning', 'positive'),
  c('larry friend', 'positive'),
  c('not like that man', 'negative'),
  c('house not great', 'negative'),
  c('your song annoying', 'negative')
)

tweets = rbind(pos_tweets, neg_tweets, test_tweets)

# build document term matrix
matrix= create_matrix(tweets[,1], language="english", 
                      removeStopwords=FALSE, removeNumbers=TRUE, 
                      stemWords=FALSE)

# train the model
mat = as.matrix(matrix)
classifier = naiveBayes(mat[1:10,], as.factor(tweets[1:10,2]) )


  #test
  predicted = predict(classifier, mat[11:15,]); predicted
  table(tweets[11:15, 2], predicted)
  recall_accuracy(tweets[11:15, 2], predicted)
}

# Note - this is not runnable as a function yet, mostly
# a container for cut and pasting into console to run.
tryAugTweets <- function(trainRows=1:1000, testRows=1001:1100, sentimentAug, docTerms)
{

  print('training range')
  print(min(trainRows))
  print(max(trainRows))
  
  print('test range')
  print(min(testRows))
  print(max(testRows))
  
  # Read in data if necessary
  if (dim(sentimentAug)[1] != 13871) 
  { 
    print('Reading in august tweets')
    sentimentAug <- read.csv("c:\\users\\doylerav\\onedrive\\cs6220\\project\\SentimentforR.csv", header=TRUE)
  }
  
  # need to prune term matrix down somehow - 10k tweets gets some errors.
  # figure out how to customize stopwords?
  if (dim(docTerms)[1] != 13871) 
  {
    print('Creating term matrix')
    
        docTerms <- create_matrix(sentimentAug$text, language="english", 
                            removeStopwords=TRUE, 
                            removeNumbers=TRUE, 
                            stemWords=FALSE, 
                            toLower=TRUE,
                            removePunctuation = TRUE,
                            minWordLength = 3)
  }
  
  # build the data to specify response variable, training set, testing set.
  # TODO - check on virgin settings
  print('Creating container')
  container = create_container(docTerms, as.numeric(as.factor(sentimentAug$sentiment)),
                               trainSize=trainRows, testSize=testRows,virgin=FALSE)

  # For each model, train and get test results and accuracy
  # You can lump these together, but they take a while to run.
  models = train_models(container, algorithms=c("MAXENT", "SVM"))
  results = classify_models(container, models)  
  
  table(as.numeric(as.factor(sentimentAug$sentiment[testRows])), results[,"MAXENTROPY_LABEL"])
  recall_accuracy(as.numeric(as.factor(sentimentAug$sentiment[testRows])), results[,"MAXENTROPY_LABEL"])
  
  table(as.numeric(as.factor(sentimentAug$sentiment[testRows])), results[,"SVM_LABEL"])
  recall_accuracy(as.numeric(as.factor(sentimentAug$sentiment[testRows])), results[,"SVM_LABEL"])
  

  # model summary
  analytics = create_analytics(container, results)
  summary(analytics)
  head(analytics@document_summary)
  analytics@ensemble_summary
  
  # 5 fold cross validation
  N=5
  set.seed(2014)
  cross_validate(container,N,"MAXENT")
  cross_validate(container,N,"SVM")


  }

buildTermMatrix <- function(trainTweets,testTweets)
{
  allTweets <- rbind(trainTweets, testTweets)
  
  allTweetsTermMatrix = create_Matrix(allTweets[,1], language="english", minDocFreq=1, maxDocFreq=Inf, 
  minWordLength=3, maxWordLength=Inf, ngramLength=1, originalMatrix=NULL, 
  removeNumbers=FALSE, removePunctuation=TRUE, removeSparseTerms=0, 
  removeStopwords=FALSE,  stemWords=FALSE, stripWhitespace=TRUE, toLower=TRUE)

  allTweetsTermMatrix
}

tryAugTweetsNB <- function()
{
  sentimentAug <- read.csv("c:\\users\\doylerav\\onedrive\\cs6220\\Sentiment.csv", header=TRUE)
  
  docTerms <- create_matrix(sentimentAug$text[1:100], language="english", removeStopwords=TRUE, removeNumbers=TRUE, stemWords=FALSE)
  featureMatrix <- as.matrix(docTerms)
  
  #classifier <- naiveBayes(featureMatrix[1:90], as.factor(sentimentAug$sentiment[1:90]))
  #predicted <- predict(classifier, featureMatrix[90:100])
  #featureMatrix <- as.matrix(docTerms)
  #classifier <- naiveBayes(featureMatrix[1:5000,], 
  #                         as.factor(sentimentAug$sentiment[1:5000]))
  
  predicted <- predict(classifier, featureMatrix[5001:6000,])
  predicted
  
  tempTable <- table(sentimentAug$sentiment[5001:6000], predicted)
  recallStats <- recall_accuracy(sentimentAug$sentiment[5001:6000], predicted)
  recallStats
}

