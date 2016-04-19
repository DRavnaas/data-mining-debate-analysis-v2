library(RTextTools)
library(caret)
library(RWeka)
library(tm)

# wordcloud example: has info on exploring terms
#http://faculty.washington.edu/jwilker/CAP/R_Sample_Script.R

# cleaning tips
#https://sites.google.com/site/miningtwitter/questions/talking-about/given-users

# Drop neutral labels, just train/test o on positive/negative
tryAugNoNeutral <- function(verbose=FALSE, doJustOneFold=TRUE)
{
  # This file is in github: 
  # https://github.com/yogimiraje/data-mining-debate-analysis/tree/master/R
  print('Reading in august tweets (and removing neutrals)')
  sentiment <-
    read.csv(
      "c:\\users\\doylerav\\onedrive\\cs6220\\project\\SentimentforR.csv",
      header = TRUE
    )
  
  sentAugNoNeutral <- sentiment[sentimentAug$sentiment!="Neutral",]
  
  tryAugTweetsRun(sentiment=sentAugNoNeutral, verbose, doJustOneFold)
}

tryAugTweetsRun <- function(sentiment=NULL, verbose=FALSE, doJustOneFold=TRUE)
{

  # Read in data if necessary
  if (is.null(sentiment))
  {
    # This file is in github: 
    # https://github.com/yogimiraje/data-mining-debate-analysis/tree/master/R
    print('Reading in august tweets (including neutrals)')
    sentiment <-
      read.csv(
        "c:\\users\\doylerav\\onedrive\\cs6220\\project\\SentimentforR.csv",
        header = TRUE
      )
    
  }

  
  print('Creating fold list')
  
  # build folds of the data for cross validation
  # the id happens to be a row number
  # TODO: make sure we have this column for march csv for R
  fold0 <- sentiment[sentiment$id %% 5 == 0, ]
  fold1 <- sentiment[sentiment$id %% 5 == 1, ]
  fold2 <- sentiment[sentiment$id %% 5 == 2, ]
  fold3 <- sentiment[sentiment$id %% 5 == 3, ]
  fold4 <- sentiment[sentiment$id %% 5 == 4, ]
  
  # Build containers where the last rows are the test fold
  #cv1All <- rbind(fold0, fold1, fold2, fold3, fold4)
  cv1All <- sentiment
  cv2All <- rbind(fold1, fold2, fold3, fold4, fold0)
  cv3All <- rbind(fold2, fold3, fold4, fold0, fold1)
  cv4All <- rbind(fold3, fold4, fold0, fold1, fold2)
  cv5All <- rbind(fold4, fold0, fold1, fold2, fold3)
  
  numRows <- as.matrix(dim(sentiment))[1,1]
  endTrain <- as.integer(.8 * numRows)
  trainRows <- 1:endTrain
  testRows <-    (endTrain+1):numRows
  
  accSumAcrossFolds.maxEnt <- 0
  accSumAcrossFolds.svm <- 0
  accSumAcrossFolds.glmnet <- 0
  
  folds <- list(cv1All, cv2All, cv3All, cv4All, cv5All)
  foldNum <- 0
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
    
    foldNum <- foldNum + 1
    cat("  Fold", foldNum, ": ")

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
      
      # Want to see what the terms ended up being?
      # inspect(docTerms[1,])
    }
    if (useCreateMatrix == FALSE)
    {
    
      # nGramLength > 1 doesn't work, so use Weka to build term matrix.
      # Note - need to keep this in sync with create_matrix above
      cat("Creating term matrix2... ")
      
      corpus <- Corpus(VectorSource(curFold$text))
    
      # do a variety of transformations that are intended to 
      # separate/normalize words
      toSpace <- content_transformer(function(x,pattern)
        gsub(pattern," ", x))
      
      removeIt <- content_transformer(function(x, pattern) 
        gsub(pattern, "", x))
      
      # Force certain word separators to a space
      # so we can extract words on either side
 
      corpus <- tm_map(corpus,toSpace, "\n")
      corpus <- tm_map(corpus,toSpace,"\t")
      corpus <- tm_map(corpus,toSpace,"\r")
      corpus <- tm_map(corpus, removeIt, "RT @")

      # Turn the ... character into a space for
      # word separation
      corpus <- tm_map(corpus, toSpace, " .")
      corpus <- tm_map(corpus, toSpace, ". ")
      
      # Collapse whitespace and remove punc & numbers
      corpus <- tm_map(corpus, removePunctuation)
      
      corpus <- tm_map(corpus, stripWhitespace)
    
      corpus <- tm_map(corpus, removeNumbers)
    
      # Remove links (assumed to be relatively unique)
      corpus <- tm_map(corpus, removeIt, "http\\w+")
      
      # Remove any word at the end of the string that 
      # ends with the truncation character
      corpus <- tm_map(corpus,removeIt,"\\s*\\w*\\.$")
      
      # You can examine the resulting tweet text like so:
      #as.character(as.character(corpus[[4]]))
      
      xgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = nGramLength, max = nGramLength))
      docTerms <- DocumentTermMatrix(corpus,
                              control=list(tolower=tolower,
                                           weighting=weightTfIdf, 
                                           tokenize = xgramTokenizer))

      # Take out extremely sparse terms to reduce term matrix
      docTerms <- removeSparseTerms(docTerms, sparse=0.9999)
      
      if (verbose == TRUE)
      {
        print(docTerms)
      }
      
      # inspect(docTerms[1:3, 20:30])
      # findFreqTerms(docTerms,2)
    }
    
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
    #algos = c("GLMNET", "MAXENT") # this runs relatively quick (though SVM is usually better)
    #algos = c("GLMNET", "SVM")  #SVM and GLMNET have the edge usually over MAXENT for accuracy
    algos = c("MAXENT", "GLMNET", "SVM")
    
    cat("Running ", algos, "...")
    
    models = train_models(container, algorithms = algos)
    results = classify_models(container, models)
    
    # Get maxEnt results for this fold
    accuracyForFold.maxEnt = "NA"
    accuracyForFold.maxEnt <-
      recall_accuracy(as.numeric(as.factor(curFold$sentiment[testRows])), results[, "MAXENTROPY_LABEL"])
    accSumAcrossFolds.maxEnt <- accSumAcrossFolds.maxEnt + accuracyForFold.maxEnt
    
    if (verbose == TRUE)
    {
      confusionMatrix(results$MAXENTROPY_LABEL, as.numeric(as.factor(curFold$sentiment[testRows])))
    }
 
    accuracyForFold.glmnet = "NA"
    if (length(algos) > 1)
    {
      # Get svm results for this fold
      accuracyForFold.glmnet <-
        recall_accuracy(as.numeric(as.factor(curFold$sentiment[testRows])), results$GLMNET_LABEL)
      accSumAcrossFolds.glmnet <- accSumAcrossFolds.glmnet + accuracyForFold.glmnet
      
      if (verbose == TRUE)
      {
        confusionMatrix(results$GLMNET_LABEL, as.numeric(as.factor(curFold$sentiment[testRows])))
      }
    }
       
    accuracyForFold.svm = "NA"
    if (length(algos) > 2)
    {
      # Get svm results for this fold
      accuracyForFold.svm <-
        recall_accuracy(as.numeric(as.factor(curFold$sentiment[testRows])), results[, "SVM_LABEL"])
      accSumAcrossFolds.svm <- accSumAcrossFolds.svm + accuracyForFold.svm
    
      if (verbose == TRUE)
      {
        confusionMatrix(results$SVM_LABEL, as.numeric(as.factor(curFold$sentiment[testRows])))
      }
    }

    print(cat("  Fold accuracy: maxent=", accuracyForFold.maxEnt, ", svm=", accuracyForFold.svm, ", glmnet=",
              accuracyForFold.glmnet, " "))

    analytics = create_analytics(container, results)
    
    if (length(algos) > 1)
    {
      print(analytics@ensemble_summary)
    }
    
    # Idea - keep just sum of ensemble summary matrix (k algos x 2)
    # Save as files the various per fold summaries?
    results_ensemble <- as.matrix(analytics@ensemble_summary)
    results_document <- as.matrix(analytics@document_summary)
    results_label <- as.matrix(analytics@label_summary)
    results_algorithm <- as.matrix(analytics@algorithm_summary)

    #write.csv("")
    
    if (doJustOneFold == TRUE)
    {
      # Useful when testing out some new code.
      break
    }
  }

  # model summary - work out how to use/aggregate this for 5 folds?
  
  if (verbose == TRUE && length(algos) > 1)
  {
    #print("Analytics for last fold: ")
    #analytics = create_analytics(container, results)
    #print(summary(analytics))
    #print(head(analytics@document_summary))

  }
    
  meanAcc.maxEnt <- accSumAcrossFolds.maxEnt / foldNum
  
  print(cat("Mean accuracy across folds, MAXENT: ", meanAcc.maxEnt, " "))
  
  meanAcc.glmnet <- accSumAcrossFolds.glmnet / foldNum
  
  print(cat("Mean accuracy across folds, glmnet: ", meanAcc.glmnet, " "))

  meanAcc.svm <- accSumAcrossFolds.svm / foldNum
  
  print(cat("Mean accuracy across folds, svm:    ", meanAcc.svm, " "))
  
  # return the list of fold analytics
  #perFoldAnalytics
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
