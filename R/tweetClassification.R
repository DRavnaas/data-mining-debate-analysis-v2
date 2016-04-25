library(RTextTools)
library(caret)
library(RWeka)
library(tm)
library(e1071)

# method - trainAndEvaluate - August and March, or a particular CSV

# TODO: try on "fresh" machine to ensure readme is right (use R version 3.2.4?)
# save algorithm probabilities and run through SVM for prediction, save trained svm model

#   trainAndEvaluate - trains on a set and does 5 fold validation - good for evaluating tuning
#   trainAndPredictOnAllLabeled - trains on the Aug and March labeled set, then makes predictions on that set
#   predictLabelsAfterTraining - trains on a labeled set, then makes predictions on unlabelled set

# see github for various data files
# https://github.com/yogimiraje/data-mining-debate-analysis/tree/master/R
#  AugSentiment.csv = the original sentiment file from Kaggle/Crowdflower
#  March10th_before_labeledForR.csv = hand labeled sample of March tweets before the debate
#  March10th_all_labeledForR.csv = hand labeled sample of March tweets before the debate
#  AugAndMarchLabeled.csv - August and March labeled tweets, "common" columns
#  AugAndMarchLabeledQuote.csv - August with sentiment, March with quote sentiment, labeled tweets

# Primary references used for transforming tweets for R:
# 
# http://faculty.washington.edu/jwilker/CAP/R_Sample_Script.R
# https://sites.google.com/site/miningtwitter/questions/talking-about/given-users

# General help for RTextTools
# https://github.com/timjurka/RTextTools/tree/master/RTextTools/inst/examples
# http://www.inside-r.org/packages/cran/RTextTools/docs/create_container


# Train and evaluate an ensemble (default = on both Aug and March labeled data)
# optionally does 5 fold cross validation and saves the trained model and results
trainAndEvaluate <- function(csvPath="AugAndMarchLabeledQuote.csv", 
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
  
  # When these go through as.factor, 1 = Negative, 3 = Positive
  print(paste("# rows before removing neutrals = ", dim(tweetRows)[1]))
  numPositive <- dim(tweetRows[tweetRows$sentiment=="Positive",])[1]
  numNeutral <- dim(tweetRows[tweetRows$sentiment=="Neutral",])[1]
  numNegative <- dim(tweetRows[tweetRows$sentiment=="Negative",])[1]
  print(paste("# positive = ", numPositive, ", # neutral = ", numNeutral, ", # negative =", numNegative))

  tweetsNoNeutral <- tweetRows[tweetRows$sentiment!="Neutral",]
  print(paste("# rows after removing neutrals = ", dim(tweetsNoNeutral)[1]))
 
  if (is.null(saveToFolder) == FALSE && length(saveToFolder) > 0)
  {
    print("Saving filtered data to folder...")
    
    tweetsNoNeutral$text <- replaceLineFeedsFromColumn(tweetsNoNeutral$text)
    
    if (dir.exists(saveToFolder) == FALSE)
    {
      dir.create(saveToFolder)
    }
    
    # Save filtered data?
    filteredDataPath = paste(saveToFolder, "\\FilteredData.csv", sep="")
  
    write.csv(tweetsNoNeutral, filteredDataPath, fileEncoding="UTF-8")
    
  }
    
  tryTweetsRun(tweetRows=tweetsNoNeutral, verbose=verbose, doJustOneFold=doJustOneFold, saveToFolder=saveToFolder)
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
  print(paste("# positive =", numPositive, ", # neutral =", numNeutral, ", # negative =", numNegative))
    
  tryTweetsRun(tweetRows=tweetsRows, verbose=verbose, doJustOneFold=doJustOneFold, saveToFolder=saveToFolder, testRows=NULL)
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
  nGramLength <- 1 # run with unigrams for speed, bigrams for slightly better accuracy
  
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
    # This is a bit buggy (ngramLength > 1 doesn't work at the very least)
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

trainOnAugTestOnMarch <- function(verbose=FALSE, dropNeutral=TRUE, sentColumnName="sentiment")
{
  allLabeled <- buildAllLabeledFromCsvs(marchSentimentColumnName="sentiment")
  
  # TODO: Could likely use tweet_created data or id ranges to figure
  # this out dynamically, but with August data being fixed, hardcoding isn't so bad.
  AugEnd <- 13871
  AugEndNoNeutral <- 10729
  
  if (dropNeutral == TRUE)
  {
    allLabeled <- dropNeutrals(allLabeled)
    AugEnd <- AugEndNoNeutral
  }
  
  testRows <- AugEnd:dim(allLabeled)[1]
  
  tryTweetsRun(tweetRows=allLabeled, verbose=verbose, doJustOneFold=TRUE, testRows=testRows)
  
}

# Main helper function to train and evaluate input tweets
tryTweetsRun <- function(tweetRows, 
                            verbose=FALSE, 
                            doJustOneFold=TRUE, 
                            saveToFolder=NULL,
                            testRows=NULL)
{
  
  if (verbose == TRUE)
  {
    print(Sys.time())
  }
  
  # If the caller specified testRows, but it doesn't look valid... 
  if ((is.null(testRows) == FALSE) && (length(testRows) <= 1))
  {
    print("Invalid testRows value (null or length <=1)")
    return
  }
  
  if ((is.null(testRows) == TRUE) || (length(testRows) <= 1))
  {
    print(paste("Assigning 20% of rows randomly to test fold"))
    
    numRows <- dim(tweetRows)[1]
    endTrain <- as.integer(.8 * numRows)
    trainRows <- 1:endTrain
    testRows <-    (endTrain+1):numRows
    
  } else
  {
    # This is useful for running on March only (w doJustOneFold=TRUE)
    print(paste("Fixing test rows to input set (rather than random)"))
    
    numRows <- dim(tweetRows)[1]
    endTrain <- numRows - length(testRows)
    trainRows <- 1:endTrain
    
    # Force this to true, use to force test data at end (Aug/March)
    # ie: we assume numTestRows means all at the end of the input rows.
    doJustOneFold <- TRUE
    
  }
  
  if (doJustOneFold == FALSE)
  {
    print(paste('Creating fold list for ', dim(tweetRows)[1], "rows, # train =", length(trainRows), "/ # test =", length(testRows)))
    
    folds <- buildFolds(tweetRows)
  }
  else {
    print(paste('Creating one time run fold for ', dim(tweetRows)[1], "rows, # train =", length(trainRows), "/ # test =", length(testRows)))
    
    folds <- list(tweetRows)
  }
  
  foldNum <- 0
  
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
    print(paste("  Fold", foldNum, ": of", dim(curFold)[1], "rows"))

    docTerms <- buildDocTermMatrix(curFold, verbose=FALSE)
    #inspect(docTerms)
    
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
    if (is.null(saveToFolder) == FALSE && length(saveToFolder) > 0)
    {
      print("Saving model and results...")
      
      if (dir.exists(saveToFolder) == FALSE)
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
  
  if (verbose == TRUE && length(algos) == 1 && doJustOneFold==TRUE)
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


# Trains on a labeled set and then makes predictions on a labeled set
# using the trained model - returns results 
trainAndPredict <- function(tweetRowsTrain, predictRows, verbose=FALSE, saveToFolder=NULL)
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
  if (is.null(saveToFolder) == FALSE && length(saveToFolder) > 0)
  {
    print("Saving model and results...")
    
    if (dir.exists(saveToFolder) == FALSE)
    {
      dir.create(saveToFolder)
    }
    
    trainedModelAndResultsPath = paste(saveToFolder, "\\", "modelsAndLabels.RData", sep="")
    
    save(models, container, results, file = trainedModelAndResultsPath)
  }
  
  results
  
}

buildAllLabeledFromCsvs <- function(marchSentimentColumnName = "sentiment", saveToCsvPath = NULL)
{
  
  # These files much have the columns listed in readLabeledDataFrame,
  # and the column values must be similar (ie: "neutral" != "Neutral", 
  # "Trump" != "trump", etc)
  AugCsvPath <- "AugSentiment.csv"
  marchB4Path <- "March10th_before_labeledforR.csv"
  marchAfterPath <- "March10th_after_labeledforR.csv"

  print(paste("Reading labeled rows from", AugCsvPath, ",", marchB4Path, ",", marchAfterPath))
  
  augSentiment <- readLabeledDataFrame(AugCsvPath)
  marchb4sentiment <- readLabeledDataFrame(marchB4Path, sentimentColumnName=marchSentimentColumnName)
  marchAfterSentiment <- readLabeledDataFrame(marchAfterPath, sentimentColumnName=marchSentimentColumnName)
  
  # We now have three data frames with consistent column names
  allLabeled <- rbind(augSentiment, marchb4sentiment, marchAfterSentiment)

  if (is.null(saveToCsvPath)== FALSE && length(saveToCsvPath) > 0)
  {
    # August data has some linefeeds that causes excel problems
    # NOTE: August also is a UTF-8 file and March isn't - so we save as UTF-8
    allLabeled$text <- replaceLineFeedsFromColumn(allLabeled$text) 
    write.csv(allLabeled, saveToCsvPath, fileEncoding = "UTF-8")
  }
  
  allLabeled  
}

readLabeledDataFrame <- function(csvPath, idColumn = "id", sentimentColumnName = "sentiment")
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

# Build and optionall save off the unlabeled tweet data
# unLabeledTweets <- biuldAllUnlabeledFromCsvs("UnlabeledMarchForR.csv")
buildAllUnlabeledFromCsvs <- function(saveToCsvPath = NULL)
{
  
  # These files much have the columns listed in readLabeledDataFrame,
  # and the column values must be similar (ie: "neutral" != "Neutral", 
  # "Trump" != "trump", etc)
  marchB4Path <- "March10th_before_allunlabeled_forR.csv"
  marchAfterPath <- "March10th_after_allunlabeled_forR.csv"
  
  print(paste("Reading unlabeled rows from", marchB4Path, ",", marchAfterPath))
  
  # unlabeled data
  allMarchUnlabeledB4 <- readUnlabeledDataFrame(marchB4Path)
  allMarchUnlabeledAfter <- readUnlabeledDataFrame(marchAfterPath)
  
  allMarchUnlabeled <- rbind(allMarchUnlabeledB4, allMarchUnlabeledAfter)
  
  if (is.null(saveToCsvPath) == FALSE && length(saveToCsvPath) > 0)
  {
    # August data has some linefeeds that causes excel problems
    # NOTE: August also is a UTF-8 file and March isn't - so we save as UTF-8
    allMarchUnlabeled$text <- replaceLineFeedsFromColumn(allMarchUnlabeled$text) 
    write.csv(allMarchUnlabeled, saveToCsvPath, fileEncoding = "UTF-8")
  }
  
  allMarchUnlabeled  
  
 
   
}

# Read in an unlabeled file 
# There's probably a smart way to refactor with with readLabeledDataFrame?
readUnlabeledDataFrame <- function(csvPath, idColumn = "id")
{
  fulldataFrame <- read.csv(
    csvPath,
    header = TRUE,
    encoding = "UTF-8"
  )
  
  print(paste("# rows read = ", dim(fulldataFrame)[1]))
  
  if (dim(fulldataFrame)[2] < 20)
  {
    # March files can have a couple columns we have to shift around.
    # (TODO: this might not be needed now?)
    fulldataFrame$id <- fulldataFrame[,idColumn]
    
    #print(fulldataFrame$id[1])
    #print(fulldataFrame$sentiment[1])
  }
  
  miniDataFrame <- cbind.data.frame(fulldataFrame$id, 
                                    fulldataFrame$tweet_id,
                                    fulldataFrame$candidate, 
                                    fulldataFrame$tweet_created,
                                    fulldataFrame$tweet_location,
                                    fulldataFrame$user_timezone,  
                                    fulldataFrame$text
  )
  
  
  colnames(miniDataFrame) <- c("id", "tweet_id", "candidate", "tweet_created", 
                               "tweet_location", "user_timezone", "text")
  
  
  miniDataFrame
}


# Read in the labeled set, train on the entire set, then predict on that set.
# Output a file with tweet data and predictions (UTF-8)
trainAndPredictOnAllLabeled <- function(verbose=FALSE, marchSentimentColumnName = "quote_sentiment", dropNeutrals=TRUE, saveToCsvPath="LabeledWithPredictionsQuote.csv")
{
  allLabeled <- buildAllLabeledFromCsvs(marchSentimentColumnName)
  
  if (dropNeutrals == TRUE)
  {
    allLabeled <- dropNeutrals(allLabeled)
  }
  
  predictOnTrainingSet(allLabeled, verbose, saveToCsvPath)
}

# Outputs a csv with predictions on a given training set
# Since training = the prediction set, the accuracy for this is 
# misleadingly high - but used for initial success metric work.
predictOnTrainingSet <- function(tweetRows, verbose=FALSE, saveToCsvPath)
{
  # Our train and test rows are exactly the same.
  
  numRows <- dim(tweetRows)[1]
 
  trainRows <- 1:numRows
  testRows <- 1:numRows
  
  docTerms <- buildDocTermMatrix(tweetRows, verbose=FALSE)
  
  container = create_container(
    docTerms,
    as.numeric(as.factor(tweetRows$sentiment)),
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
  
  print("done!")
  print(analytics@ensemble_summary)
  
  # Build an output data frame with all the various ensemble label results
  predictedSentiment <- docResults$CONSENSUS_CODE
  
  tweetRowsWithPrediction <- tweetRows
  tweetRowsWithPrediction$predictedLabel <- docResults$CONSENSUS_CODE
  tweetRowsWithPrediction$actualLabel <- docResults$MANUAL_CODE
  tweetRowsWithPrediction$maxEntPrediction <- docResults$MAXENTROPY_LABEL
  tweetRowsWithPrediction$maxEntProbability <- docResults$MAXENTROPY_PROB
  tweetRowsWithPrediction$svmPrediction <- docResults$SVM_LABEL
  tweetRowsWithPrediction$svmProbability <- docResults$SVM_PROB
  tweetRowsWithPrediction$glmnetPrediction <- docResults$GLMNET_LABEL
  tweetRowsWithPrediction$glmnetProbability <- docResults$GLMNET_PROB
  tweetRowsWithPrediction$consensusCount <- docResults$CONSENSUS_AGREE
  tweetRowsWithPrediction$probabilityLabel <- docResults$PROBABILITY_CODE
  tweetRowsWithPrediction$consensusLabel <- docResults$CONSENSUS_CODE
  
  # At the moment, we are going with the consensus label
  tweetRowsWithPrediction$predictedLabel <- docResults$CONSENSUS_CODE

  tweetRowsWithPrediction$text <- replaceLineFeedsFromColumn(tweetRowsWithPrediction$text) 
  
  print(paste("Saving results to ", saveToCsvPath))
  write.csv(tweetRowsWithPrediction, saveToCsvPath, fileEncoding="UTF-8")
  
}

# Warning - this takes a long time for the gopdebate data set.
# This method trains on the labeled set and then makes predictions on the unlabeled set
predictLabelsAfterTraining <- function(dropNeutrals = TRUE, csvOutputPath = "UnLabeledWithPredictions.csv")
{
  # Read in and train our model
  
  allLabeled <- buildAllLabeledFromCsvs(marchSentimentColumnName)
  
  if (dropNeutrals == TRUE)
  {
    allLabeled <- dropNeutrals(allLabeled)
  }
  
  allMarchUnlabeled <- buildAllUnlabeledFromCsvs()

  allMarchUnlabeled$sentiment <- c("Positive",rep("Negative",c(nrow(allMarchUnlabeled)-1)))

  allTrainAndUnlabeled <- rbind(allLabeled, allMarchUnlabeled)
      
  numRows <- dim(allLabeled)[1]
  endTrain <- numRows
  trainRows <- 1:endTrain
  testRows <- (endTrain+1):(endTrain+dim(allTest)[1])
  
  docTerms <- buildDocTermMatrix(allTrainAndUnlabeled, verbose=FALSE)
  
  container = create_container(
    docTerms,
    as.numeric(as.factor(allTrainAndUnlabeled$sentiment)),
    trainSize = trainRows,
    testSize = testRows,
    virgin = FALSE
  )
  
  algos = c("MAXENT", "GLMNET", "SVM")
  
  cat("Running ", algos, "...")
  
  models = train_models(container, algorithms = algos)
  
  print("Training done!")
  
  # Now predict labels for the test rows
  # Note that any accuracy reported is bogus
  # since we faked the sentiment for the unlabeled rows.
  # But we will now have per row/ per algo prediction info.
  results = classify_models(container, models)  

  print("Prediction done!")
  
  # This all took forever, so let's save it.
  save(models, container, file = "AugAndMarchUnlabeledRun/trainedModels.RData")
  
  analytics = create_analytics(container, results)
  
  save(analytics, results, file = "AugAndMarchUnlabeledRun/analytics.RData")
  
  docResults <- analytics@document_summary
  
  analytics = create_analytics(container, results)
  docResults <- analytics@document_summary
  
  predictedSentiment <- docResults$CONSENSUS_CODE
  
  # Output the results.  Notice there's no point outputting the 
  # label since it isn't a real value for that tweet.
  allPredictions <- allTest
  allPredictions$predictedLabel <- docResults$CONSENSUS_CODE
  #allPredictions$actualLabel <- docResults$MANUAL_CODE
  allPredictions$maxEntPrediction <- docResults$MAXENTROPY_LABEL
  allPredictions$maxEntProbability <- docResults$MAXENTROPY_PROB
  allPredictions$svmPrediction <- docResults$SVM_LABEL
  allPredictions$svmProbability <- docResults$SVM_PROB
  allPredictions$glmnetPrediction <- docResults$GLMNET_LABEL
  allPredictions$glmnetProbability <- docResults$GLMNET_PROB
  allPredictions$consensusCount <- docResults$CONSENSUS_AGREE
  allPredictions$probabilityLabel <- docResults$PROBABILITY_CODE
  allPredictions$consensusLabel <- docResults$CONSENSUS_CODE
  
  # At the moment, we are going with the consensus label
  allPredictions$predictedLabel <- docResults$CONSENSUS_CODE

  if (is.null(csvOutputPath) == FALSE)
  {
    allPredictions$text <- replaceLineFeedsFromColumn(allPredictions$text)  
    
    print(paste("Saving data and predictions to", csvOutputPath))
    # this is march only for unlabeled so no need to save as utf-8
    write.csv(allPredictions, csvOutputPath)
  }
  
  # Can sample the returned results with
  # numRows <- dim(allPredictions)[1]
  # predictSample <- allPredictions[sample(1:numRows,size=100,replace=FALSE),]
  
  allPredictions
}

# A quick Naive Bayes test
tryTweetsNB <- function(csvPath="AugAndMarchLabeledQuote.csv")
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


dropNeutrals <- function(tweetRows)
{
  # When these go through as.factor, 1 = Negative, 3 = Positive
  print(paste("# rows before removing neutrals = ", dim(tweetRows)[1]))
  numPositive <- dim(tweetRows[tweetRows$sentiment=="Positive",])[1]
  numNeutral <- dim(tweetRows[tweetRows$sentiment=="Neutral",])[1]
  numNegative <- dim(tweetRows[tweetRows$sentiment=="Negative",])[1]
  print(paste("# positive = ", numPositive, ", # neutral = ", numNeutral, ", # negative =", numNegative))
 
  tweetsNoNeutral <- tweetRows[tweetRows$sentiment!="Neutral",]
  print(paste("# rows after removing neutrals = ", dim(tweetsNoNeutral)[1]))
  
  tweetsNoNeutral
  
}


tryLabellingJustNeutral <- function(tryJustNeutralOrNot=TRUE)
{
  allLabeled <- read.csv("labeledWithPredictionsQuoteEdit.csv", sep=",")
  
  y <- as.factor(allLabeled$sentiment)

  # sentiment,predictedLabel,actualLabel,maxEntPrediction,maxEntProbability,svmPrediction,svmProbability,glmnetPrediction,glmnetProbability,consensusCount,probabilityLabel,consensusLabel,python_sentiment,ensemble_pos,ensemble_neg,lexicon pos,lexicon_neu,ensemble_neg,lexicon_compound
  sampleColNames <- c(
                  "maxEntPrediction", "maxEntProbability", 
                   "svmPrediction", "svmnetProbability", 
                   "glmnetPrediction", "glmnetProbability",
                   "consensusCount", "probabilityLabel", "consensusLabel",
                    "ensemble_pos", "ensemble_neg", "lexicon_pos",
                    "lexicon_neu"  , "ensemble_neg", "lexicon_compound")
  
  if (tryJustNeutralOrNot == TRUE)
  {
    # Basically turn this into an 'is neutral' flag
    first <-  gsub("Negative", "NotNeutral", allLabeled$sentiment)
    second <- gsub("Positive", "NotNeutral", first)
    #y <- as.numeric(as.factor(third))
    y <- as.factor(second)
  }
  
  
  print(paste("# rows read in = ", dim(allLabeled)[1]))
  
  x <- cbind.data.frame(allLabeled$maxEntPrediction,
                        allLabeled$maxEntProbability,
                        allLabeled$svmPrediction,
                        allLabeled$svmProbability,
                        allLabeled$glmnetPrediction,
                        allLabeled$glmnetProbability,
                        allLabeled$consensusCount,
                        allLabeled$probabilityLabel,
                        allLabeled$consensusLabel,
                        
                        #allLabeled$python_sentiment,
    
                        allLabeled$ensemble_pos,
                        allLabeled$ensemble_neg,
                        allLabeled$lexicon_pos,
                        allLabeled$lexicon_neu,
                        allLabeled$ensemble_neg,
                        allLabeled$lexicon_compound)
  
  colnames(x) <- sampleColNames
    
  svmModel <- svm(x, y) 
  
  pred <- predict(svmModel, x)
  
  predAndY <- cbind.data.frame(pred, y, second)
  
  print(paste("Predictions on labeled august and march:"))
  
  print( table(pred, y) )
  
  
  sample <- read.csv("Sample_CombinedResultsWithAllStats.csv")
  sample <- sample[1:100,]
  
  sampx <- cbind.data.frame(sample$maxEntPrediction,
                            sample$maxEntProbability,
                            sample$svmPrediction,
                            sample$svmProbability,
                            sample$glmnetPrediction,
                            sample$glmnetProbability,
                            sample$consensusCount,
                            sample$probabilityLabel,
                            sample$consensusLabel,
  
                        #allLabeled$python_sentiment,
  
                        sample$ensemble_pos,
                        sample$ensemble_neg,
                        sample$lexicon_pos,
                        sample$lexicon_neu,
                        sample$ensemble_neg,
                        sample$lexicon_compound)
  
  colnames(x) <- sampleColNames
  
  print(paste("Predictions on labeled sample from March:"))
  
  pred <- predict(svmModel, sampx)
  
  sampy <- sample$X.correct..human.label
  
  if (tryJustNeutralOrNot == TRUE)
  {
    firsty <- gsub("1", "NotNeutral", as.character(sampy))
    firsty <- gsub("3", "NotNeutral", as.character(firsty))
    firsty <- gsub("2", "Neutral", as.character(firsty))
    newsampy <- as.factor(firsty)
  }
  
  if (tryJustNeutralOrNot == FALSE)
  {
    firsty <- gsub("1", "Negative", as.character(sampy))
    firsty <- gsub("3", "Positive", as.character(firsty))
    firsty <- gsub("2", "Neutral", as.character(firsty))
    newsampy <- as.factor(firsty)
  }
  
  print( table(pred, newsampy) )
  
  save(svmModel, file="svmModel.RData")
  #predAndY
}