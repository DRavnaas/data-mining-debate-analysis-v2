import csv
import re

## This
records = []
num = 0
headers = {}

with open('../data/gop/august/august_full.csv', 'rU') as csvFile:
#with open('data/gop/march/combined_sample_unique.csv', 'rU') as csvFile:
    reader = csv.DictReader(csvFile)

    for row in reader:
        item = []
        row["text"] = re.sub('[^a-zA-Z0-9 -!%+#?$;@()&:\'"]',"",row["text"]);
        records.append(row)

#with open('data/gop/march/combined_sample_unique_quote_form.csv',"wb+") as ofile:
with open('../data/gop/august/august_full_clean.csv',"wb+") as ofile:
    writer = csv.writer(ofile,delimiter=',')
    fieldnames = ['id','candidate','candidate_confidence','relevant_yn','relevant_yn_confidence','sentiment',
                  'sentiment_confidence','subject_matter','subject_matter_confidence','candidate_gold','name',
                  'relevant_yn_gold','retweet_count','sentiment_gold','subject_matter_gold','text','tweet_coord',
                  'tweet_created','tweet_id','tweet_location','user_timezone']

    writer = csv.DictWriter(ofile, fieldnames=fieldnames)

    writer.writeheader()
    for record in records:
        writer.writerow(record)

