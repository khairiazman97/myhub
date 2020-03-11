#importing libraries

import numpy as pd
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
data = pd.read.csv("TELCO.csv")
x = data.iloc[:,1:21].values
y = data.iloc[:, 21].values

#encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lex1 = LabelEncoder()
x[:,1] = lex1.fit.transform(x[:,1])
lex2 = LabelEncoder()
x[:,2] = lex2.fit.transform(x[:,2])
ohe = OneHotEncoder(categorical_features = [1])
x = ohe.fit.transform(x).toarray()
x = x[:, 1]