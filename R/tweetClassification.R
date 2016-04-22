library(RTextTools)
library(caret)
library(RWeka)
library(tm)
library(e1071)

# method - trainAndEvaluate - August and March, or a particular CSV

# TODO: try on "fresh" machine to ensure readme is right (use R version 3.2.4?)
#   TODO - trainAndPredict - trains on full set, then makes predictions
#   TODO - predictFromTrainedModel - takes a trained model and docTermMatrix and makes predictions
# save algorithm probabilities and run through SVM for prediction, save trained svm model

# see github for various data files
# https://github.com/yogimiraje/data-mining-debate-analysis/tree/master/R
#  AugSentiment.csv = the original sentiment file from Kaggle/Crowdflower
#  March10th_before_labeled.csv.txt = hand labeled sample of March tweets before the debate
#  March10th_all_labeled.csv.txt = hand labeled sample of March tweets before the debate
#  AllLabeledMini.csv - August and March labeled tweets, "common" 12 columns
#  AllLabeledQuoteMini.csv - August with sentiment, March with quote sentiment, labeled tweets

# Primary references used for transforming tweets for R:
# 
# http://faculty.washington.edu/jwilker/CAP/R_Sample_Script.R
# https://sites.google.com/site/miningtwitter/questions/talking-about/given-users

# Train and evaluate an ensemble (default = on both Aug and March labeled data)
# optionally does 5 fold cross validation and saves the trained model and results
trainAndEvaluate <- function(csvPath="AllLabeledQuoteMini.csv", 
                             verbose=FALSE, 
                             doJustOneFold=FALSE,
                             saveToFolder=NULL)
{
  tryTweetsNoNeutral(csvPath, verbose, doJustOneFold, saveToFolder)
}


# Drop neutral labels, just train/test on positive/negative
tryTweetsNoNeutral <- function(csvPath="AugSentiment.csv", 
                            verbose=FALSE, 
                            doJustOneFold=FALSE,
                            saveToFolder=NULL)
{
  print('Reading in tweets (and removing neutrals)')
  tweetRows <-
    read.csv(
      csvPath,
      header = TRUE,
      encoding = "UTF-8"
    )
  
  print(paste("# rows before removing neutrals = ", dim(tweetRows)[1]))
  numPositive <- dim(tweetRows[tweetRows$sentiment=="Positive",])[1]
  numNeutral <- dim(tweetRows[tweetRows$sentiment=="Neutral",])[1]
  numNegative <- dim(tweetRows[tweetRows$sentiment=="Negative",])[1]
  print(paste("# positive = ", numPositive, ", # neutral = ", numNeutral, ", # negative =", numNegative))

  tweetsNoNeutral <- tweetRows[tweetRows$sentiment!="Neutral",]
  print(paste("# rows after removing neutrals = ", dim(tweetsNoNeutral)[1]))
 
  if (!is.null(saveToFolder) && length(saveToFolder) > 0)
  {
    print("Saving filtered data to folder...")
    
    tweetsNoNeutral$text <- replaceLineFeedsFromColumn(tweetsNoNeutral$text)
    
    if (!dir.exists(saveToFolder))
    {
      dir.create(saveToFolder)
    }
    
    # Save filtered data?
    filteredDataPath = paste(saveToFolder, "\\FilteredData.csv", sep="")
  
    write.csv(tweetsNoNeutral, filteredDataPath, fileEncoding="UTF-8")
    
  }
    
  tryTweetsRun(tweetRows=tweetsNoNeutral, verbose, doJustOneFold, saveToFolder)
}

tryTweetsWithNeutral<- function(csvPath="AugSentiment.csv", 
                                   verbose=FALSE, 
                                   doJustOneFold=TRUE,
                                   saveToFolder=NULL)
{

  print('Reading in tweets (with neutrals)')
  tweetRows <-
    read.csv(
      csvPath,
      header = TRUE,
      encoding = "UTF-8"
    )

  print(paste("# rows read in = ", dim(tweetRows)[1]))
  numPositive <- dim(tweetRows[tweetRows$sentiment=="Positive",])[1]
  numNeutral <- dim(tweetRows[tweetRows$sentiment=="Neutral",])[1]
  numNegative <- dim(tweetRows[tweetRows$sentiment=="Negative",])[1]
  print(paste("# positive = ", numPositive, ", # neutral = ", numNeutral, ", # negative =", numNegative))
    
  tryTweetsRun(tweetRows, verbose, doJustOneFold, saveToFolder)
}

buildFolds <- function(tweetRows)
{
  # build folds of the data for cross validation
  # the id happens to be a row number
  fold0 <- tweetRows[tweetRows$id %% 5 == 0, ]
  fold1 <- tweetRows[tweetRows$id %% 5 == 1, ]
  fold2 <- tweetRows[tweetRows$id %% 5 == 2, ]
  fold3 <- tweetRows[tweetRows$id %% 5 == 3, ]
  fold4 <- tweetRows[tweetRows$id %% 5 == 4, ]
  
  # Build containers where the last rows are the test fold
  
  cv1All <- rbind(fold0, fold1, fold2, fold3, fold4)
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
    # word separation - 
    # BE SURE TO SAVE THIS R FILE AS UTF-8!!
    corpus <- tm_map(corpus, toSpace, " …")
    corpus <- tm_map(corpus, toSpace, "… ")
    
    # Collapse whitespace and remove punc & numbers
    corpus <- tm_map(corpus, removePunctuation)
    
    corpus <- tm_map(corpus, stripWhitespace)
    
    corpus <- tm_map(corpus, removeNumbers)
    
    # Remove links (assumed to be relatively unique)
    corpus <- tm_map(corpus, removeIt, "http\\w+")
    
    # Remove any word at the end of the string that 
    # ends with the truncation character
    corpus <- tm_map(corpus,removeIt,"\\s*\\w*\\…$")
    
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
tryTweetsRun <- function(tweetRows=NULL, 
                            verbose=FALSE, 
                            doJustOneFold=TRUE, 
                            saveToFolder=NULL)
{
  if (verbose == TRUE)
  {
    print(Sys.time())
  }
  
  print('Creating fold list')
  
  folds <- buildFolds(tweetRows)
  foldNum <- 0
    
  numRows <- dim(tweetRows)[1]
  endTrain <- as.integer(.8 * numRows)
  trainRows <- 1:endTrain
  testRows <-    (endTrain+1):numRows
  
  if (length(trainRows) + length(testRows) != numRows)
  {
    print("WARNING: # training + # test != total!!!")
  }
  
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
    #algos = c("GLMNET", "MAXENT") # this runs relatively quick (SVM needs a lot of iterations)
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

}


trainAndPredict <- function(tweetRowsTrain, predictRows, verbose=FALSE, saveToFolder=TRUE)
{
  if (verbose == TRUE)
  {
    print(Sys.time())
  }
  
  numRows <- dim(tweetRowsTrain)[1]
  endTrain <- numRows
  trainRows <- 1:endTrain
  testRows <-    endTrain+1:endTrain + dim(predictRows)[1]
  
  # Build one set with both train and predict row text
  curFold <- rbind(tweetRowsTrain, predictRows)
  
  
  if ((length(trainRows) + length(testRows)) != dim(curFold)[1])
  {
    print("WARNING: # training + # test != total!!!")
  }  
  
  
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
  #algos = c("GLMNET", "MAXENT") # this runs relatively quick (SVM needs a lot of iterations)
  #algos = c("GLMNET", "SVM")  #SVM and GLMNET have the edge usually over MAXENT for accuracy
  algos = c("MAXENT", "GLMNET", "SVM")
    
  cat("Running ", algos, "...")
    
  models = train_models(container, algorithms = algos)
  results = classify_models(container, models)

  # save off trained classifier, container, labels and results
  if (!is.null(saveToFolder) && length(saveToFolder) > 0)
  {
    print("Saving model and results...")
    
    if (!dir.exists(saveToFolder))
    {
      dir.create(saveToFolder)
    }
    
    trainedModelAndResultsPath = paste(saveToFolder, "\\", "modelsAndLabels.RData", sep="")
    
    save(models, container, results, file = trainedModelAndResultsPath)
  }
  
  results
  
  # return the list of fold analytics
  #perFoldAnalytics
}

buildAllMiniFromCsvs <- function(marchSentimentColumnName = "sentiment", saveMiniFile = NULL)
{
  # These files much have the columns listed in readMiniDataFrame,
  # and the column values must be similar (ie: "neutral" != "Neutral", 
  # "Trump" != "trump", etc)
  AugCsvPath <- "AugSentiment.csv"
  marchB4Path <- "March10th_before_labeled.csv.txt"
  marchAfterPath <- "March10th_after_labeled.csv.txt"
  
  augSentiment <- readMiniDataFrame(AugCsvPath)
  
  # The march csvs have three sentiment columns, and the UTF-8 BOM ends up in the first column name
  marchb4sentiment <- readMiniDataFrame(marchB4Path, "X.U.FEFF.id", marchSentimentColumnName)
  marchAfterSentiment <- readMiniDataFrame(marchAfterPath, "X.U.FEFF.id", marchSentimentColumnName)
  
  # We now have three data frames with consistent column names
  allMini <- rbind(augSentiment, marchb4sentiment, marchAfterSentiment)
  
  if (!is.null(saveMiniFile) && length(saveMiniFile))
  {
    write.csv(allMini, saveMiniFile, fileEncoding = "UTF-8")
  }
  
  allMini  
}

readMiniDataFrame <- function(csvPath, idColumn = "id", sentimentColumnName = "sentiment")
{
  fulldataFrame <- read.csv(
    csvPath,
    header = TRUE,
    encoding = "UTF-8"
  )

  print(paste("# rows read = ", dim(fulldataFrame)[1]))
    
  if (dim(fulldataFrame)[2] < 20)
  {
    # March files have a couple columns we have to shift around.
    fulldataFrame$id <- fulldataFrame[,idColumn]
    fulldataFrame$sentiment <- fulldataFrame[,sentimentColumnName]
  
    #print(fulldataFrame$id[1])
    #print(fulldataFrame$sentiment[1])
  }
  
  
  miniDataFrame <- cbind.data.frame(fulldataFrame$id, 
                                      fulldataFrame$tweet_id,
                                      fulldataFrame$candidate, 
                                      fulldataFrame$tweet_created,
                                      fulldataFrame$sentiment,
                                      fulldataFrame$tweet_location,
                                      fulldataFrame$user_timezone,  
                                      fulldataFrame$text
                                      )
 
  
  colnames(miniDataFrame) <- c("id", "tweet_id", "candidate", "tweet_created", 
                              "sentiment", "tweet_location", "user_timezone", "text")
  
  
  miniDataFrame
}

tryTweetsNB <- function(csvPath="AugSentiment.csv")
{
  tweetRows <-
    read.csv(csvPath,
             header = TRUE)
  print(paste("# rows read = ", dim(tweetRows)[1]))
  
  docTerms <- buildDocTermMatrix(tweetRows)
  featureMatrix <- as.matrix(docTerms)

  cat("Running Naive Bayes...")
    
  numRows <- as.matrix(dim(tweetRows))[1,1]
  endTrain <- as.integer(.8 * numRows)
  trainRows <- 1:endTrain
  testRows <-    (endTrain+1):numRows
  
  classifier <- naiveBayes(featureMatrix[trainRows], as.factor(tweetRows$sentiment[trainRows]))
  predicted <- predict(classifier, featureMatrix[testRows])

  recallStats <-
    recall_accuracy(tweetRows$sentiment[testRows], predicted)
  
  cat("Prediction accuracy: ", recallStats)
}

# For easy use of excel with our output csvs, turn linefeeds
# in the text column into spaces
replaceLineFeedsFromColumn <- function(columnOfText)
{
  gsub("\n", " ", columnOfText)
}

#  A collection of work in progress / commands for easy cut/paste
testAndDebug <- function()
{
  allMini <- read.csv("AllLabeledQuoteMini2.csv", header=TRUE, encoding="UTF-8", fileEncoding="UTF-8")
  
  allMiniNoNeutral <- tweetRows[allMini$sentiment!="Neutral",]
  
  # Combine so we train on full test and predict on full set
  allMiniNoNeutralx2 <- rbind(allMiniNoNeutral, allMiniNoNeutral)
  
  numRows <- dim(allMiniNoNeutralx2)[1]
  endTrain <- as.integer(.5 * numRows)
  trainRows <- 1:endTrain
  testRows <-    (endTrain+1):numRows
  
  docTerms <- buildDocTermMatrix(allMiniNoNeutralx2, verbose=FALSE)
  
  # TODO: This is training on the first half and then testing on those
  # same rows we copied to the second half, which is a bit of a hack
  # Rework to separate train and test data 
  # http://www.inside-r.org/packages/cran/RTextTools/docs/create_container
  container = create_container(
    docTerms,
    as.numeric(as.factor(allMiniNoNeutralx2$sentiment)),
    trainSize = trainRows,
    testSize = testRows,
    virgin = FALSE
  )

  algos = c("MAXENT", "GLMNET", "SVM")
  
  cat("Running ", algos, "...")
  
  models = train_models(container, algorithms = algos)
  results = classify_models(container, models)  
  
  analytics = create_analytics(container, results)
  docResults <- analytics@document_summary
  
  predictedSentiment <- docResults$CONSENSUS_CODE
  
  allMiniNoNeutralWithPrediction <- allMiniNoNeutral
  allMiniNoNeutralWithPrediction$predictedLabel <- docResults$CONSENSUS_CODE
  allMiniNoNeutralWithPrediction$actualLabel <- docResults$MANUAL_CODE
  
  write.csv(allMiniNoNeutralWithPrediction, "LabeledAndPredictedQuoteMini.csv", fileEncoding="UTF-8")
  
}
