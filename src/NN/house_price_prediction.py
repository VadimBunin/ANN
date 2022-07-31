import torch
from sklearn.preprocessing import LabelEncoder
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
# print(df.head())

# Use LabelEncoder to convert cat_features into 0 ...n_classes - 1
cat_features = ["MSSubClass", "MSZoning", "Street", "LotShape"]
out_feature = "SalePrice"
lbl_encoders = {}
for features in cat_features:
    lbl_encoders[features] = LabelEncoder()
    df[features] = lbl_encoders[features].fit_transform(df[features])

# print(df.head())

# Stacking cat_features

cat_features = np.stack(
    [df['MSSubClass'], df['MSZoning'], df['Street'], df['LotShape']], 1)

# Convert cat_features_numpy into tensors
cat_features = torch.tensor(cat_features, dtype=torch.int64)

# Create continuos features
cont_features = []

for i in df.columns:
    if i in ["MSSubClass", "MSZoning", "Street", "LotShape", "SalePrice"]:
        pass
    else:
        cont_features.append(i)

# Stack cont_features
cont_values = np.stack([df[i].values for i in cont_features], 1)

# Convert cont_values into tensors
cont_values = torch.tensor(cont_values, dtype=torch.float)
print(cont_values.dtype)
