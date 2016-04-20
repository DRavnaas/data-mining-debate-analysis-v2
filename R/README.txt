
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

Open RStudio and run the following commands:

# Set the working directory to where you downloaded the files
setwd("C:/Users/doylerav/OneDrive/r")  

# Load the code
source('tweetClassification.R')

# Run the evaluation on the training data, see the statistics
# Note: this will take ~ five minutes since it does 3 algorithms
# for five fold evaluation on the data
classifier <- trainAndTestNoNeutral()

# Make predictions using the trained classifier
results <- predictFromTrainedClassifier(classifier)



