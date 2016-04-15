import csv
import re

## This
records = []
num = 0
headers = {}

with open('data/gop/august/august_full.csv', 'rU') as csvFile:
    reader = csv.DictReader(csvFile)

    for row in reader:
        item = []
        sentiment =  str(row["sentiment"]).lower().replace(' ','').replace('\r','').replace('\n', '')

        if(sentiment == 'positive' or sentiment == 'negative' or sentiment == 'neutral'):
            item.append('|' + str(row["candidate"]) + '|')
            item.append('|' + sentiment + '|')
            item.append('|' + str(row["text"]).replace('\r',' ').replace('\n', ' ') + '|')
            records.append(item)
            num = num + 1
        else:
            print "skipped as sentiment : %s" % sentiment

with open('data/gop/august/august_full_form.csv',"wb+") as ofile:
    writer = csv.writer(ofile,delimiter=',')
    writer.writerows(records)

