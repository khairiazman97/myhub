#import csv
#import numpy
#
#file ="pima-indians-diabetes.csv"
#rawdata = open(file,"rt")
#reader = csv.reader(rawdata,delimiter = ",", quoting = csv.QUOTE_NONE)
#x = list(reader)
#data = numpy.array(x).astype("float")
#print(data)

#or

#with numpy
#import numpy
#
#file ="pima-indians-diabetes.csv"
#rawdata = open(file,"rt")
#data = numpy.loadtxt(rawdata,delimiter=",")
#print(data.shape)==> for reshaping the data.

# data from online site
import numpy as np
import pandas as pd
from numpy import loadtxt
from urllib.request import urlopen
url = ""
raw_data = urlopen(url)
dataset = pd.read_csv(raw_data)
print(dataset.shape)