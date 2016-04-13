import csv
import re

records = []
num = 0

reader = csv.reader(open('data/gop/august/august_full.csv', 'rU'))
ofile = open('data/gop/august/august_full_form.csv',"wb+")
#reader = csv.reader(open('data/gop/march/before_sample.csv', 'rU'))
#ofile = open('data/gop/march/before_sample_form.csv',"wb+")
writer = csv.writer(ofile,delimiter=',')

for record in reader:
    item = []
    item.append('|' + str(record[0]) + '|')
    item.append('|' + str(record[1]).replace('\r',' ').replace('\n', ' ') + '|')
    records.append(item)
    num = num + 1

writer.writerows(records)
