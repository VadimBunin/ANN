import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Create Dependent Feature
y = torch.tensor(df['SalePrice'].values, dtype=torch.float).reshape(-1, 1)

# Appendix A: Create Embedding Size for Categorical Columns
cat_featuresz = cat_features[:4]
print(cat_featuresz)
# compute numbers of the unique values for every cat_columns
cat_dims = [len(df[col].unique())
            for col in ["MSSubClass", "MSZoning", "Street", "LotShape"]]
# print(cat_dims)
# compute outpute dims of the embedding
embedding_dim = [(x, min(50, (x+1)//2)) for x in cat_dims]
# print(embedding_dim)

embed_representation = nn.ModuleList(
    [nn.Embedding(inp, out) for inp, out in embedding_dim])
# print(embed_representation)


embed_val = []

for i, e in enumerate(embed_representation):
    embed_val.append(e(cat_featuresz[:, i]))

print(embed_val)

z = torch.cat(embed_val, 1)
print(z)
