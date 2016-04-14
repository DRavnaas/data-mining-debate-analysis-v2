import csv
import re

## This
records = []
num = 0
headers = {}


with open('data/gop/march/after_sample.csv', 'rU') as csvFile:
    reader = csv.DictReader(csvFile)

    for row in reader:
        item = []
        item.append('|' + str(row["sentiment"]) + '|')
        item.append('|' + str(row["text"]).replace('\r',' ').replace('\n', ' ') + '|')
        records.append(item)
        num = num + 1

with open('data/gop/march/after_sample_form.csv',"wb+") as ofile:
    writer = csv.writer(ofile,delimiter=',')
    writer.writerows(records)

