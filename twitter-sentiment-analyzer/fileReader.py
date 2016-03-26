import csv

ifile = open('data/augustNew.csv',"rU")

reader = csv.reader(ifile)

num=0
for row in reader:
    for  col in row:
       print col

    if (num ==50):
        break
    num += 1