library(RTextTools)
library(e1071)


tryAugTweetsRun <- function(sentimentAug=NULL, verbose=FALSE)
{
  
  rebuildDocTerms <- FALSE
  
  # Read in data if necessary
  if (is.null(sentimentAug) || dim(sentimentAug)[1] != 13871)
  {
    print('Reading in august tweets')
    sentimentAug <-
      read.csv(
        "c:\\users\\doylerav\\onedrive\\cs6220\\project\\SentimentforR.csv",
        header = TRUE
      )
    
    rebuildDocTerms = TRUE
  }
  
  # need to prune term matrix down somehow - 10k tweets gets some errors.
  # figure out how to customize stopwords?
  
  if (rebuildDocTerms || is.null(docTerms) || dim(docTerms)[1] != 13871)
  {
    #print('Creating default term matrix')
    
      #docTerms <- create_matrix(
      #sentimentAug$text,
      #language = "english",
      #removeStopwords = FALSE,
      #removeNumbers = TRUE,
      #stemWords = FALSE,
      #toLower = TRUE,
      #removePunctuation = TRUE,
      #minWordLength = 3,
      #weighting = tm::weightTf,
      #ngramLength = 1
    #)
  }
  
  print('Creating fold list')
  
  # build folds of the data for cross validation
  fold0 <- sentimentAug[sentimentAug$id %% 5 == 0, ]
  fold1 <- sentimentAug[sentimentAug$id %% 5 == 1, ]
  fold2 <- sentimentAug[sentimentAug$id %% 5 == 2, ]
  fold3 <- sentimentAug[sentimentAug$id %% 5 == 3, ]
  fold4 <- sentimentAug[sentimentAug$id %% 5 == 4, ]
  
  # Build containers where the last rows are the test fold
  cv1All <- rbind(fold0, fold1, fold2, fold3, fold4)
  cv2All <- rbind(fold1, fold2, fold3, fold4, fold0)
  cv3All <- rbind(fold2, fold3, fold4, fold0, fold1)
  cv4All <- rbind(fold3, fold4, fold0, fold1, fold2)
  cv5All <- rbind(fold4, fold0, fold1, fold2, fold3)
  
  trainRows <- 1:11097
  testRows <-    11098:13871
  
  accSumAcrossFolds <- 0
  
  folds <- list(cv1All, cv2All, cv3All, cv4All, cv5All)
  foldNum <- 1
  for (curFold in folds)
  {
    # build the data to specify response variable, training set, testing set.
    # virgin=FALSE means has a label (TRUE = data we haven't seen/labeled)
    
    cat("  Fold", foldNum, ": ")
    foldNum <- foldNum + 1
    
    cat(" creating term matrix...")
    docTerms <- create_matrix(
      curFold$text,
      language = "english",
      removeStopwords = FALSE,
      removeNumbers = TRUE,
      stemWords = FALSE,
      toLower = TRUE,
      removePunctuation = TRUE,
      minWordLength = 3,
      weighting = tm::weightTf,
      ngramLength = 1
    )
    
    container = create_container(
      docTerms,
      as.numeric(as.factor(curFold$sentiment)),
      trainSize = trainRows,
      testSize = testRows,
      virgin = FALSE
    )
    
    
    # For each model, train and get test results and accuracy
    # You can lump these together to run as an ensemble, but they take a while to run.
    algos = c("MAXENT") #, "SVM")
    
    cat("Running algorithm", algos, "...")
    
    models = train_models(container, algorithms = algos)
    results = classify_models(container, models)
    
    #table(as.numeric(as.factor(curFold$sentiment[testRows])), results[,"MAXENTROPY_LABEL"])
    accuracyForFold <-
      recall_accuracy(as.numeric(as.factor(curFold$sentiment[testRows])), results[, "MAXENTROPY_LABEL"])
    foldConfMatrix <-
      confusionMatrix(results$MAXENTROPY_LABEL, as.numeric(as.factor(curFold$sentiment[testRows])))
    accSumAcrossFolds <- accSumAcrossFolds + accuracyForFold
    
    if (verbose)
    {
      print(foldConfMatrix)
    }
    
    print(cat("  Accuracy for fold: ", accuracyForFold, " "))
  }
  
  meanAcc <- accSumAcrossFolds / 5
  
  print(cat("Mean accuracy across 5 folds: ", meanAcc))
  
  #table(as.numeric(as.factor(sentimentAug$sentiment[testRows])), results[,"SVM_LABEL"])
  #recall_accuracy(as.numeric(as.factor(sentimentAug$sentiment[testRows])), results[,"SVM_LABEL"])
  
  #table(as.numeric(as.factor(sentimentAug$sentiment[testRows])), results[,"GLMNET_LABEL"])
  #recall_accuracy(as.numeric(as.factor(sentimentAug$sentiment[testRows])), results[,"GLMNET_LABEL"])
  
  #confusionMatrix(results$MAXENTROPY_LABEL, as.numeric(as.factor(sentimentAug$sentiment[11098:13871])))
  
  
  # model summary
  #analytics = create_analytics(container, results)
  #summary(analytics)
  #head(analytics@document_summary)
  #analytics@ensemble_summary
  
  # 5 fold cross validation
  
  
  #N=5
  #set.seed(2014)
  #cross_validate(container,N,"MAXENT")
  #cross_validate(container,N,"SVM")
  
  
}

buildTermMatrix <- function(trainTweets, testTweets)
{
  allTweets <- rbind(trainTweets, testTweets)
  
  allTweetsTermMatrix = create_Matrix(
    allTweets[, 1],
    language = "english",
    minDocFreq = 1,
    maxDocFreq = Inf,
    minWordLength = 3,
    maxWordLength = Inf,
    ngramLength = 1,
    originalMatrix = NULL,
    removeNumbers = FALSE,
    removePunctuation = TRUE,
    removeSparseTerms = 0,
    removeStopwords = FALSE,
    stemWords = FALSE,
    stripWhitespace = TRUE,
    toLower = TRUE
  )
  
  allTweetsTermMatrix
}

tryAugTweetsNB <- function()
{
  sentimentAug <-
    read.csv("c:\\users\\doylerav\\onedrive\\cs6220\\Sentiment.csv",
             header = TRUE)
  
  docTerms <-
    create_matrix(
      sentimentAug$text[1:100],
      language = "english",
      removeStopwords = TRUE,
      removeNumbers = TRUE,
      stemWords = FALSE
    )
  featureMatrix <- as.matrix(docTerms)
  
  #classifier <- naiveBayes(featureMatrix[1:90], as.factor(sentimentAug$sentiment[1:90]))
  #predicted <- predict(classifier, featureMatrix[90:100])
  #featureMatrix <- as.matrix(docTerms)
  #classifier <- naiveBayes(featureMatrix[1:5000,],
  #                         as.factor(sentimentAug$sentiment[1:5000]))
  
  predicted <- predict(classifier, featureMatrix[5001:6000, ])
  predicted
  
  tempTable <- table(sentimentAug$sentiment[5001:6000], predicted)
  recallStats <-
    recall_accuracy(sentimentAug$sentiment[5001:6000], predicted)
  recallStats
}
