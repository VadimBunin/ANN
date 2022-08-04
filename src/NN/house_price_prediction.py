import matplotlib.pyplot as plt
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

# Embedding Size For Categorical columns

cat_dims = [len(df[col].unique())
            for col in ["MSSubClass", "MSZoning", "Street", "LotShape"]]
print(cat_dims)

embedding_dim = [(x, min(50, (x+1)//2)) for x in cat_dims]
print(embedding_dim)


# Create a Feed Forward Neural Network

class FeedForwardNN(nn.Module):

    def __init__(self, embedding_dim, n_cont, out_sz, layers, p=0.4):
        super().__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(inp, out) for inp, out in embedding_dim])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        # total input_dims
        n_emb = sum((out for inp, out in embedding_dim))
        n_in = n_emb + n_cont

        # create sequence of layers

        layerlist = []

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))

        self.layers = nn.Sequential(*layerlist)

        # create foreard method

    def forward(self, x_cat, x_cont):
        embeddings = []

        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x


torch.manual_seed(100)
model = FeedForwardNN(embedding_dim, len(cont_features), 1, [100, 50], p=0.4)
print(model)

# Define Loss fx and opimizer

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

batch_size = 1200
test_size = int(batch_size*0.15)
train_categorical = cat_features[:batch_size-test_size]
test_categorical = cat_features[batch_size-test_size:batch_size]
train_cont = cont_values[:batch_size-test_size]
test_cont = cont_values[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]

# Train the model

epochs = 6000
final_losses = []
for i in range(epochs):
    i = i+1
    y_pred = model(train_categorical, train_cont)
    loss = torch.sqrt(loss_function(y_pred, y_train))
    final_losses.append(loss.item())
    if i % 100 == 1:
        print("Epoch number: {} and the loss : {}".format(i, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), final_losses)
plt.ylabel('RMSE Loss')
plt.xlabel('epoch')
plt.show()
# Appendix A: Create Embedding Size for Categorical Columns
"""
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

"""
