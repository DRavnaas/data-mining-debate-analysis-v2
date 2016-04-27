
To run the R algorithms, please do the following:

All code was run using R 3.2.3 and RStudio 0.99.891 on a Windows system.

Download links:

R: 3.2.3 (or higher):
https://cran.r-project.org/bin/windows/base/old/3.2.3/ 
Other platform downloads available here:  https://cran.r-project.org/

RStudio:  0.99.891 (or higher):
https://www.rstudio.com/products/rstudio/download/ 

(On Windows, you'll want to start RStudio in admin mode so it can install packages required by the code)

Download the data files and the R file from github (data files are also on google drive):
- tweetClassification.R

- AugSentiment.csv
- AugAndMarchLabeledQuote.csv
- UnlabeledMarchForR.csv

For the following, let's assume you downloaded the files to the folder C:\gopfolder\R

Open RStudio and run the following commands in the console:

# Set the working directory to where you downloaded the files
# NOTE: replace the path with your download directory.
setwd("C:/gopfolder/R")  

# Load the code from the working directory
source('tweetClassification.R', encoding = 'UTF-8')

# Run the evaluation on the training data, see the statistics
trainAndEvaluate(doJustOneFold=TRUE)

# Other methods are documented in the R file, happy R'ing!

