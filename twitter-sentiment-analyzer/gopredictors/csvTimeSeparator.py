import csv
import re

records = []
num = 0

headers = []

#with open('../data/gop/UnlabeledWithPredictions_v1.csv', 'rU') as csvFile:
with open('/Users/Yogi/Study/4_Spring_16/DM/project/EVALUATION/predicted/unlabeled_python_ensemble.csv', 'rU') as csvFile:
    reader = csv.DictReader(csvFile)


    for row in reader:
        timestamp =  row["tweet_created"]

        date = timestamp[0:10]
        hour = timestamp[11:13]
        minute = timestamp[14:16]

        row["date"]   = date
        row["hour"]   = hour
        row["minute"] = minute

        del row[""]

        records.append(row)

    all_fields = reader._fieldnames

    for key in all_fields:
        if (key != ''):
            headers.append(key)

        if (key == "tweet_created"):
            headers.append("date")
            headers.append("hour")
            headers.append("minute")



#with open('../data/gop/UnlabeledWithPredictions_v1_ts_separated.csv',"wb+") as ofile:
with open('/Users/Yogi/Study/4_Spring_16/DM/project/EVALUATION/predicted/predicted_march_debate_v2.csv',"wb+") as ofile:

    writer = csv.DictWriter(ofile, fieldnames=headers)
    #print headers
    writer.writeheader()
    for record in records:
        try:
            #print record
            writer.writerow(record)
        except ValueError as e:
            print '{} missing'.format(record)

