# Import some of the Track Features from tar.gz file into pandas dataframe -->

# import packages
import pandas as pd
import numpy
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# import own functions

from CSV_categorize import sns_plot, sns_plot_count, filter_df, convert_df_to_array, convert_df_to_numeric, convert_array_to_torch

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# settings matrixrange (plot view)

pd.options.display.max_columns = None
pd.options.display.max_rows = 20

# Track Features

path1 = r'/Users/moritzdeecke/Desktop/Spotify_Dataset/track_features'
feature_files = glob.glob(path1 + "/*.csv")

lis1 = []

for filename in feature_files:
    df1 = pd.read_csv(filename, index_col=None, header=0)
    lis1.append(df1)

track_features = pd.concat(lis1, axis=0, ignore_index=True)

# Test Data

path2 = r'/Users/moritzdeecke/Desktop/Spotify_Dataset/test_set1'
test_files = glob.glob(path2 + "/*.csv")

lis2 = []

for filename in test_files:
    df2 = pd.read_csv(filename,usecols=['track_id_clean', 'not_skipped'], index_col=None, header=0)
    lis2.append(df2)

test_data = pd.concat(lis2, axis=0, ignore_index=True)
test_data = test_data.rename(columns={'track_id_clean': 'track_id'})

# Train Data

path3 = r'/Users/moritzdeecke/Desktop/Spotify_Dataset/training_set'
train_files = glob.glob(path3 + "/*.csv")

lis3 = []

for filename in train_files:
    df3 = pd.read_csv(filename, usecols=['track_id_clean', 'not_skipped'], index_col=None, header=0)
    lis3.append(df3)

train_data = pd.concat(lis3, axis=0, ignore_index=True)
train_data = train_data.rename(columns={'track_id_clean': 'track_id'})

ds_train = pd.merge(train_data, track_features, on="track_id", how="inner")
ds_test = pd.merge(test_data, track_features, on="track_id", how="inner")
print(ds_test['track_id'].nunique())
print(ds_train['track_id'].nunique())
print(ds_test['not_skipped'].value_counts())
print(ds_train['not_skipped'].value_counts())

# Y-Value
sns_plot_count(track_features)
print(track_features.info())
print(ds_test.info())
print(ds_train.info())
labels_te = convert_df_to_array(ds_test["not_skipped"])
labels_tr = convert_df_to_array(ds_train["not_skipped"])
labels_torch_te = convert_array_to_torch(labels_te)
labels_torch_tr = convert_array_to_torch(labels_tr)


# Drop labels from Dataframe

ds_train = ds_train.drop(['not_skipped', 'track_id'], axis=1)
ds_test = ds_test.drop(['not_skipped', 'track_id'], axis=1)

# Train Data
musicdata_tr = convert_df_to_array(convert_df_to_numeric(ds_train))
musicdata_tr = (musicdata_tr - np.mean(musicdata_tr, axis=0)) / (np.std(musicdata_tr, axis=0) + 1.0e-6)
musicdata_torch_train = convert_array_to_torch(musicdata_tr)

# Test Data

musicdata_te = convert_df_to_array(convert_df_to_numeric(ds_test))
musicdata_te = (musicdata_te - np.mean(musicdata_te, axis=0)) / (np.std(musicdata_te, axis=0) + 1.0e-6)
musicdata_torch_test = convert_array_to_torch(musicdata_te)

# Logistic Regression with Torch

class LogisticRegression(torch.nn.Module):
  def __init__(self, n):
    super(LogisticRegression, self).__init__()
    self.linear = torch.nn.Linear(n, 1)

  def forward(self, x):
    logits = self.linear(x)
    return torch.sigmoid(logits).squeeze()


data_torch_train = musicdata_torch_train
data_torch_test = musicdata_torch_test
model = LogisticRegression(data_torch_train.shape[1])  # Instantiate model w/ number of features in data (e.g. 14 for musicdata)

print(model.linear.weight.data)
print('b0:', model.linear.bias.data)

dataset_tr = torch.utils.data.TensorDataset(data_torch_train, labels_torch_tr)
dataset_te = torch.utils.data.TensorDataset(data_torch_test, labels_torch_te)

# n = len(dataset)  # Number of examples
# dataset_tr, dataset_te = torch.utils.data.random_split(dataset, [int(n * 0.75), int(n * 0.25)])
# print("Split training data and testing data: %s vs. %s samples." % (len(dataset_tr), len(dataset_te)))

loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=32, shuffle=True)
loader_te = torch.utils.data.DataLoader(dataset_te, batch_size=32, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
loss = torch.nn.BCELoss()
obs = {"l": []}

for x, y in loader_tr:
  y_hat = model(x)

  l = loss(y_hat, y)
  obs["l"].append(l.item())

  l.backward()

  optimizer.step()
  optimizer.zero_grad()

y_hats = torch.Tensor([])  # Leerer Tensor
ys = torch.Tensor([])
with torch.no_grad():
  for x, y in loader_te:
    y_hat = model(x)
    y_hats = torch.cat((y_hats, y_hat), dim=0)  # cat = concatenate (Verkettung)
    ys = torch.cat((ys, y), dim=0)

# Calculate ERR (Total Number of Incorrect Predictions divided by the total Number of the Dataset) / ACC (1 - ERR) / Use confusion Matrix
print(model.linear.weight)
print('b0:', model.linear.bias)
print(model.parameters)
print("ROC-AUC score: %2.5f." % roc_auc_score(ys, y_hats))

for x in range(y_hats.size(0)):
  if y_hats[x] > 0.5:
    y_hats[x] = 1
  else:
    y_hats[x] = 0

#print(y_hats)

# Build the Confusion Matrix: 1. ERR-Rate (Incorrect Pred / Total Number of Dataset), ACC-Rate (Correct Pred / Total..)
# 1 - ERR |
ACC = 0
ERR = 0
# TRP -> Sensitivity (SN) or True Positive Rate 1 and 1
TPR = 0
# TNR -> Specificity (SP) or True Negative Rate 0 and 0
TNR = 0
# FPR -> False Positive Rate 1 and 0
FPR = 0
# FNR -> False Negative Rate 0 and 1
FNR = 0
# PPV -> Precision, Correct Positive Predictions divided by total Positive Predictions
PPV = 0

for x in range(y_hats.size(0)):
  if y_hats[x] == 1 and ys[x] == 1:
    TPR = (TPR + 1)
  elif y_hats[x] == 0 and ys[x] == 0:
    TNR = TNR + 1
  elif y_hats[x] == 1 and ys[x] == 0:
    FPR = FPR + 1
  elif y_hats[x] == 0 and ys[x] == 1:
    FNR = FNR + 1

print(confusion_matrix(ys, y_hats))
print("Negative / positive examples in testing data: %s vs. %s." % ((ys == 0).sum(), (ys == 1).sum()))
print(TPR, TNR, FPR, FNR)

ACC = (TPR + TNR) / y_hats.size(0)
print('The Accuracy of the Predicted Values is (Best Accuracy is 1): ', ACC)
print('Therefor the Error of the Predicted Values is 1 - ACC (Best Error is 0): ', 1 - ACC)

SN = TPR / (y_hats.size(0) - (TNR + FNR))
print('The Sensitivity of the Predicted Values is (Best Sensitivity is 1): ', SN)

SP = TNR / (y_hats.size(0) - (TPR + FPR))
print('The Specificity of the Predicted True Negative Values is: ', SP)

PPV = TPR / (TPR + FPR)
print('The Precision (positive predictive value) is (Best Precision is 1): ', PPV)

Falsepositive = FNR / (y_hats.size(0) - (TPR + FPR))
print('The False Positive Rate is: ', Falsepositive)

raise ValueError
