library(RTextTools)
library(caret)
library(RWeka)
library(tm)
library(e1071)

#Finish cleaning up AugAndMarchLabeled in excel
#tweet 836 - quotes? How is that processed.
# case issue
# what to do with blanks in quote sentiment

# TODO: try on "fresh" machine to ensure readme is right (use R version 3.2.4?)
# method - trainAndEvaluate - August only or August and March
#        - trainAndPredict - trains on full set, then makes predictions
#        - predictFromTrainedModel - takes a trained model and docTermMatrix and makes predictions
# add march capability
# save algorithm probabilities and run through SVM for prediction, save trained svm model

# TODO: make sure we have id column for march csv for R


# see github for various data files
# https://github.com/yogimiraje/data-mining-debate-analysis/tree/master/R

# Primary references used for transforming tweets for R:
# 
#http://faculty.washington.edu/jwilker/CAP/R_Sample_Script.R
#https://sites.google.com/site/miningtwitter/questions/talking-about/given-users

# Train and evaluate an ensemble 
# optionally does 5 fold cross validation and saves the trained model and results
trainAndEvaluate <- function(csvPath="AugSentiment.csv", 
                             verbose=FALSE, 
                             doJustOneFold=FALSE,
                             saveToFolder=NULL)
{
  tryAugNoNeutral(csvPath, verbose, doJustOneFold, saveToFolder)
}


# Drop neutral labels, just train/test on positive/negative
tryAugNoNeutral <- function(csvPath="AugSentiment.csv", 
                            verbose=FALSE, 
                            doJustOneFold=FALSE,
                            saveToFolder=NULL)
{
  print('Reading in tweets (and removing neutrals)')
  sentiment <-
    read.csv(
      csvPath,
      header = TRUE
    )
  
  sentAugNoNeutral <- sentiment[sentiment$sentiment!="Neutral",]
  
  tryAugTweetsRun(sentiment=sentAugNoNeutral, verbose, doJustOneFold, saveToFolder)
}

tryAugTweetsWithNeutral<- function(csvPath="AugSentiment.csv", 
                                   verbose=FALSE, 
                                   doJustOneFold=TRUE,
                                   saveToFolder=NULL)
{

  print('Reading in tweets (with neutrals)')
  sentiment <-
    read.csv(
      csvPath,
      header = TRUE
    )
  
  tryAugTweetsRun(sentiment=sentAugNoNeutral, verbose, doJustOneFold, saveToFolder)
}

buildFolds <- function(sentiment)
{
  # build folds of the data for cross validation
  # the id happens to be a row number
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
  
  folds <- list(cv1All, cv2All, cv3All, cv4All, cv5All)
  folds
}

buildDocTermMatrix <- function(curFold, verbose=FALSE)
{
  nGramLength <- 1 # run 1/2/3 = unigrams
  
  # This toggles between two implementations of the doc term matrix builder
  # I liked the tm version better in the end.
  useCreateMatrix = FALSE
  
  if (nGramLength > 1)
  {
    # Create matrix doesn't work with ngram > 1
    useCreateMatrix = FALSE
  }
    
  if (useCreateMatrix ==TRUE)
  {
    cat("Creating term matrix (old)...")
    
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
    cat("Creating term matrix... ")
    
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
      print("Summary for doc term matrix:")
      
      print(docTerms)
    }
    
    # inspect(docTerms[1:3, 20:30])
    # findFreqTerms(docTerms,2)
  }
  
  docTerms
}

# Main helper function to train and evaluate input tweets
tryAugTweetsRun <- function(sentiment=NULL, 
                            verbose=FALSE, 
                            doJustOneFold=TRUE, 
                            saveToFolder=NULL)
{
  if (verbose == TRUE)
  {
    print(Sys.time())
  }
  
  print('Creating fold list')
  
  folds <- buildFolds(sentiment)
  foldNum <- 0
    
  numRows <- as.matrix(dim(sentiment))[1,1]
  endTrain <- as.integer(.8 * numRows)
  trainRows <- 1:endTrain
  testRows <-    (endTrain+1):numRows
  
  accSumAcrossFolds.maxEnt <- 0
  accSumAcrossFolds.svm <- 0
  accSumAcrossFolds.glmnet <- 0
  ensembleResults <- matrix(c(0,0,0,0,0,0), nrow=3, ncol=2,
                            dimnames = list(c("n >= 1", "n >= 2", "n >=3"),
                                            c("mean coverage", "mean accuracy")))
  
  
  # Loop through the folds of tweets
  for (curFold in folds)
  {
    
    foldNum <- foldNum + 1
    cat("  Fold", foldNum, ": ")

    docTerms <- buildDocTermMatrix(curFold, verbose)
    
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
      confusionDetails <- confusionMatrix(results$MAXENTROPY_LABEL, as.numeric(as.factor(curFold$sentiment[testRows])))
      print("  Confusion matrix:")
      print(confusionDetails)
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
        confusionDetails <- confusionMatrix(results$GLMNET_LABEL, as.numeric(as.factor(curFold$sentiment[testRows])))
        print("  Confusion matrix:")
        print(confusionDetails)
        
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
        confusionDetails <- confusionMatrix(results$SVM_LABEL, as.numeric(as.factor(curFold$sentiment[testRows])))
        print("  Confusion matrix:")
        print(confusionDetails)
      }
    }

    print(cat("  Fold accuracy: maxent=", accuracyForFold.maxEnt, ", svm=", accuracyForFold.svm, ", glmnet=",
              accuracyForFold.glmnet, " "))

    analytics = create_analytics(container, results)
    ensembleResults <- ensembleResults + as.matrix(analytics@ensemble_summary)
    
    
    if (length(algos) > 1)
    {
      print(analytics@ensemble_summary)
    }
    
    # Save results to the file system if specified
    if (!is.null(saveToFolder) && length(saveToFolder) > 0)
    {
      print("Saving model and results...")
      
      if (!dir.exists(saveToFolder))
      {
        dir.create(saveToFolder)
      }
      
      # Save results if requested
      if (doJustOneFold == FALSE)
      {
        analyticsFilePath = paste(saveToFolder, "\\", "analytics_", foldNum, ".RData", sep="" )
        trainedModelAndResultsPath = paste(saveToFolder, "\\", "modelsAndLabels_", foldNum, ".RData", sep="")
      }
      else {
        analyticsFilePath <- paste(saveToFolder, "\\", "analytics.RData", sep="" )
        trainedModelAndResultsPath = paste(saveToFolder, "\\", "modelsAndLabels.RData", sep="")
      }
      
      save(analytics, results, file = analyticsFilePath)
      save(models, container, file = trainedModelAndResultsPath)
    }
    
    if (doJustOneFold == TRUE)
    {
      # Useful when testing out some new code.
      print("doJustOneFold == TRUE, skipping other 4 folds")
      break
    }
  }

  # model summary - print out more details when we are running 
  # one algorithm and one fold (and asked for verbose output)
  
  if (verbose == TRUE && length(algos) == 1 && doJustOneFold)
  {
    #print("Analytics: ")
    #analytics = create_analytics(container, results)
    #print(summary(analytics))
    #print(head(analytics@document_summary))
  }
    
  meanAcc.maxEnt <- accSumAcrossFolds.maxEnt / foldNum
  
  print(cat("Mean accuracy across folds, MAXENT: ", meanAcc.maxEnt, " "))
  
  meanAcc.glmnet <- accSumAcrossFolds.glmnet / foldNum
  
  print(cat("Mean accuracy across folds, glmnet: ", meanAcc.glmnet, " "))

  meanAcc.svm <- accSumAcrossFolds.svm / foldNum
  
  print(as.character(cat("Mean accuracy across folds, svm:    ", meanAcc.svm, " ")))
  
  ensembleResults <- ensembleResults / foldNum
  
  if (doJustOneFold == FALSE)
  {
    print(ensembleResults)
  }
  
  if (verbose == TRUE)
  {
    print(Sys.time())
  }
  
  # save off trained classifier, container, labels and results
  if (!is.null(saveToFolder) && length(saveToFolder) > 0)
  {
  }

}


trainAndPredict <- function(sentimentTrain=NULL, predictSet=NULL, verbose=FALSE)
{
  if (verbose == TRUE)
  {
    print(Sys.time())
  }
  
  numRows <- as.matrix(dim(sentimentTrain))[1,1]
  endTrain <- as.integer(numRows)
  trainRows <- 1:endTrain
  testRows <-    (endTrain+1):(endTrain+dim(predictRows[1,1])+1)
  
  curFold <- c(sentimentTrain$text, predictSet$text)
    
  docTerms <- buildDocTermMatrix(curFold, verbose)
    
  # build container for this fold = train versus test rows and label
  container = create_container(
      docTerms,
      as.numeric(as.factor(curFold$sentiment)),
      trainSize = trainRows,
      testSize = testRows,
      virgin = TRUE
    )
    
  # For each model, train and get test results and accuracy
  # You can lump these together to run as an ensemble, but they take a while to run.
  #algos = c("GLMNET", "MAXENT") # this runs relatively quick (though SVM is usually better)
  #algos = c("GLMNET", "SVM")  #SVM and GLMNET have the edge usually over MAXENT for accuracy
  algos = c("MAXENT", "GLMNET", "SVM")
    
  cat("Running ", algos, "...")
    
  models = train_models(container, algorithms = algos)
  results = classify_models(container, models)
 
  results
  
  # return the list of fold analytics
  #perFoldAnalytics
}


tryAugTweetsNB <- function()
{
  sentimentAug <-
    read.csv("c:\\users\\doylerav\\onedrive\\cs6220\\Sentiment.csv",
             header = TRUE)

  docTerms <- buildDocTermMatrix(sentimentAug)
  featureMatrix <- as.matrix(docTerms)

  cat("Running Naive Bayes...")
  
  numRows <- as.matrix(dim(sentiment))[1,1]
  endTrain <- as.integer(.8 * numRows)
  trainRows <- 1:endTrain
  testRows <-    (endTrain+1):numRows
  
  classifier <- naiveBayes(featureMatrix[trainRows], as.factor(sentimentAug$sentiment[trainRows]))
  predicted <- predict(classifier, featureMatrix[testRows])

  recallStats <-
    recall_accuracy(sentimentAug$sentiment[testRows], predicted)
  
  cat("Prediction accuracy: ", recallStats)
}
