import numpy as np
import pandas as pd
from datetime import datetime as dt

df = pd.read_csv('data/houseprice.csv', usecols=["SalePrice", "MSSubClass", "MSZoning", "LotFrontage", "LotArea",
                                                 "Street", "YearBuilt", "LotShape", "1stFlrSF", "2ndFlrSF"]).dropna()
print(df.shape)

for i in df.columns:
    print(f"Column name {i} and unique values are {len(df[i].unique())}")

# replace YearBuult with TotalYear

df['Total_Year'] = dt.now().year - df['YearBuilt']
# delete YearBuilt
df.drop('YearBuilt', axis=1, inplace=True)
print(df.head())
