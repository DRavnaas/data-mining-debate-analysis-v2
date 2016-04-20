
To run the R algorithms, please do the following:

All code was run using R 3.2.3 and RStudio 0.99.891 on a Windows system.

Download links:

R: 3.2.3 (or higher):
https://cran.r-project.org/bin/windows/base/old/3.2.3/ 
Other platform downloads available here:  https://cran.r-project.org/

RStudio:  0.99.891 (or higher):
https://www.rstudio.com/products/rstudio/download/ 

(On Windows, you'll want to start RStudio in admin mode so it can install packages required by the code)

Download the data file and the R file from github:
- AugSentimentForR.csv
- tweetClassification.R

For the following, let's assume you downloaded the files to the folder C:\gopfolder\R

There's two sections - the "quick" way and the "longer" way, depending on if you want
to see the full train/evaluate and then predict flow or just go straight to predict.

THE QUICK WAY (use a previously trained classifier):
=====================================================
# Set the working directory to where you downloaded the files
# NOTE: replace the path with your download directory.
setwd("C:/gopfolder/R")  

# Load the code from the working directory
source('tweetClassification.R')



THE LONGER WAY (see all training stats and then use classifier):
================================================================

Open RStudio and run the following commands in the console:

# Set the working directory to where you downloaded the files
# NOTE: replace the path with your download directory.
setwd("C:/gopfolder/R")  

# Load the code from the working directory
source('tweetClassification.R')

# Run the evaluation on the training data, see the statistics
# Note: this will take ~ ten minutes since it trains 3 algorithms
# on five folds of evaluation on the data.  
# tip 1: trainAndTestNoNeutral(doJustOneFold=TRUE) does just one fold
# tip 2: you can save (and later reload) a classifier with the saveToFolder param
classifier <- trainAndEvaluate()

# Make predictions using the trained classifier
results <- predictFromTrainedClassifier(classifier)

# summarize results
summary(results)

