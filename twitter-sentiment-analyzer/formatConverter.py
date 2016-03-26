import csv
import re

records = []
num = 0
reader = csv.reader(open('data/gop/august_full_test.csv', 'rU'))
ofile = open('data/gop/august_full_test_form.csv',"wb+")
writer = csv.writer(ofile,delimiter=',')

for record in reader:
    item = []
    item.append('|' + str(record[0]) + '|')
    item.append('|' + str(record[1]).replace('\r',' ').replace('\n', ' ') + '|')
    records.append(item)
    num = num + 1

writer.writerows(records)
