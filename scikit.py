#regression

import pandas as pd
import quandl
import math

df = quandl.get("WIKI/Googl")
df=df[["Adj.Open","Adj.High", "Adj.Close","Adj.Volume"]]

df["HL_%"] = (df["Adj.High"]-df["Adj.Close"]/df["Adj.Close"])* 100.0
df["% change"] = (df["Adj.Close"]-df["Adj.Open"]/df["Adj.Open"])* 100.0

df = df[["Adj.Close","HL_% ", "% change", "Adj.Volume"]]
print(df)

forecast_col = "Adj.Close"
df.fillna(-99999,inplace =True)

forecast_out = int(math.ceil(0.1 * len(df)))
df["label"] = df[forecast_col].shift (-forecast_out)

df.dropna(inplace = True)
print(df.tail())
