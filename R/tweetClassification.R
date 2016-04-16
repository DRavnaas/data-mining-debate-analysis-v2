library(RTextTools)
library(caret)
library(RTextTools)
library(tm)

tryAugTweetsRun <- function(sentimentAug=NULL, verbose=FALSE, doJustOneFold=TRUE)
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
    
  }

  
  print('Creating fold list')
  
  # build folds of the data for cross validation
  # the id happens to be a row number
  # TODO: make sure we have this column for march csv for R
  fold0 <- sentimentAug[sentimentAug$id %% 5 == 0, ]
  fold1 <- sentimentAug[sentimentAug$id %% 5 == 1, ]
  fold2 <- sentimentAug[sentimentAug$id %% 5 == 2, ]
  fold3 <- sentimentAug[sentimentAug$id %% 5 == 3, ]
  fold4 <- sentimentAug[sentimentAug$id %% 5 == 4, ]
  
  # Build containers where the last rows are the test fold
  #cv1All <- rbind(fold0, fold1, fold2, fold3, fold4)
  cv1All <- sentimentAug
  cv2All <- rbind(fold1, fold2, fold3, fold4, fold0)
  cv3All <- rbind(fold2, fold3, fold4, fold0, fold1)
  cv4All <- rbind(fold3, fold4, fold0, fold1, fold2)
  cv5All <- rbind(fold4, fold0, fold1, fold2, fold3)
  
  trainRows <- 1:11097
  testRows <-    11098:13871
  
  accSumAcrossFolds.maxEnt <- 0
  accSumAcrossFolds.svm <- 0
  
  folds <- list(cv1All, cv2All, cv3All, cv4All, cv5All)
  foldNum <- 1
  nGramLength <- 1 # run 1/2/3 = unigrams
  
  useCreateMatrix = FALSE
  
  if (nGramLength > 1)
  {
    # Create matrix doesn't work with ngram > 1
    useCreateMatrix = FALSE
  }
  
  for (curFold in folds)
  {
    # build the data to specify response variable, training set, testing set.
    # virgin=FALSE means has a label (TRUE = data we haven't seen/labeled)
    
    cat("  Fold", foldNum, ": ")
    foldNum <- foldNum + 1

    if (useCreateMatrix ==TRUE)
    {
      cat("Creating term matrix1...")
      
      docTerms <- create_matrix(
        curFold$text,
        language = "english",
        removeStopwords = FALSE,  # run2 = false
        minWordLength = 3,
        ngramLength = nGramLength,  # run 1/2/3 = unigrams
        weighting = tm::weightTfIdf,  # run1/2 = weightTf
        removeNumbers = TRUE,
        stemWords = FALSE,
        toLower = TRUE,
        removePunctuation = TRUE
      )
    }
    if (useCreateMatrix == FALSE)
    {
    
      # nGramLength > 1 doesn't work, so use Weka to build term matrix.
      # Note - need to keep this in sync with create_matrix above
      cat("Creating term matrix2... ")
      
      corpus <- Corpus(VectorSource(curFold$text))
     
      corpus <- tm_map(corpus, removePunctuation)
    
      corpus <- tm_map(corpus, stripWhitespace)
    
      corpus <- tm_map(corpus, removeNumbers)
    
      xgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = nGramLength, max = nGramLength))
      docTerms <- DocumentTermMatrix(corpus,
                              control=list(weighting=weightTfIdf, 
                                           tokenize = xgramTokenizer))

    }  
  
    # Want to see what the terms ended up being?
    # inspect(docTerms[1,])
    
    # build container for this fold = train versus test rows and label
    container = create_container(
      docTerms,
      as.numeric(as.factor(curFold$sentiment)),
      trainSize = trainRows,
      testSize = testRows,
      virgin = FALSE
    )
    
    
    # For each model, train and get test results and accuracy
    # You can lump these together to run as an ensemble, but they take a while to run.
    algos = c("MAXENT", "SVM")
    
    cat("Running ", algos, "...")
    
    models = train_models(container, algorithms = algos)
    results = classify_models(container, models)
    
    # Get maxEnt results for this fold
    accuracyForFold.maxEnt <-
      recall_accuracy(as.numeric(as.factor(curFold$sentiment[testRows])), results[, "MAXENTROPY_LABEL"])
    accSumAcrossFolds.maxEnt <- accSumAcrossFolds.maxEnt + accuracyForFold.maxEnt
    
    if (verbose)
    {
      confusionMatrix(results$MAXENTROPY_LABEL, as.numeric(as.factor(curFold$sentiment[testRows])))
    }
    
    accuracyForFold.svm = "NA"
    if (length(algos) > 1)
    {
      # Get svm results for this fold
      accuracyForFold.svm <-
        recall_accuracy(as.numeric(as.factor(curFold$sentiment[testRows])), results[, "SVM_LABEL"])
      accSumAcrossFolds.svm <- accSumAcrossFolds.svm + accuracyForFold.svm
    
      if (verbose)
      {
        confusionMatrix(results$SVM_LABEL, as.numeric(as.factor(curFold$sentiment[testRows])))
      }
    }
    
    print(cat("  Fold accuracy: ", accuracyForFold.maxEnt, " maxent, ", accuracyForFold.svm, " svm "))
  }

  # model summary - work out how to use/aggregate this for 5 folds?
  if (verbose)
  {
    print("Analytics for last fold: ")
    analytics = create_analytics(container, results)
    summary(analytics)
    head(analytics@document_summary)
    analytics@ensemble_summary

    if (doJustOneFold == TRUE)
    {
      # Useful when testing out some new code.
      break
    }
    
  }
    
  meanAcc.maxEnt <- accSumAcrossFolds.maxEnt / 5
  
  print(cat("Mean accuracy across 5 folds, MAXENT: ", meanAcc.maxEnt, " "))
  
  meanAcc.svm <- accSumAcrossFolds.svm / 5
  
  print(cat("Mean accuracy across 5 folds, svm: ", meanAcc.svm, " "))


  
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
