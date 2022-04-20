import pandas as pd
import numpy as np
import glob
import torch

from data import load_test_data, load_track_features
from CSV_categorize import convert_df_to_array, convert_df_to_numeric, convert_array_to_torch
from logistic_regression import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from train import MODEL_SAVE_PATH

# Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = 20

track_features = load_track_features()
test_data = load_test_data()
ds_test = pd.merge(test_data, track_features, on="track_id", how="inner")

print(ds_test['track_id'].nunique())
print(ds_test['not_skipped'].value_counts())

# Extract labels
labels_te = convert_df_to_array(ds_test["not_skipped"])
labels_torch_te = convert_array_to_torch(labels_te)

# After extracting labels, drop label (and track_id) from dataset
ds_test = ds_test.drop(['not_skipped', 'track_id'], axis=1)
musicdata_te = convert_df_to_array(convert_df_to_numeric(ds_test))
musicdata_te = (musicdata_te - np.mean(musicdata_te, axis=0)) / (np.std(musicdata_te, axis=0) + 1.0e-6)

musicdata_torch_test = convert_array_to_torch(musicdata_te)
dataset_te = torch.utils.data.TensorDataset(musicdata_torch_test, labels_torch_te)

model = torch.load(MODEL_SAVE_PATH)
print("Model parameters for testing:")
print('Betas:', model.linear.weight.data)
print('Bias:', model.linear.bias.data)

loader_te = torch.utils.data.DataLoader(dataset_te, batch_size=32, shuffle=False)

# Leerer Tensor
y_hats = torch.Tensor([])
ys = torch.Tensor([])

with torch.no_grad():
  for x, y in loader_te:
    y_hat = model(x)
    y_hats = torch.cat((y_hats, y_hat), dim=0)  # cat = concatenate (Verkettung)
    ys = torch.cat((ys, y), dim=0)

# Calculate ERR (Total Number of Incorrect Predictions divided by the total Number of the Dataset) / ACC (1 - ERR) / Use confusion Matrix
print("ROC-AUC score: %2.5f." % roc_auc_score(ys, y_hats))
#
# for x in range(y_hats.size(0)):
#   if y_hats[x] > 0.5:
#     y_hats[x] = 1
#   else:
#     y_hats[x] = 0

y_hats = (y_hats > 0.4).int()

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
print('Therefore the Error of the Predicted Values is 1 - ACC (Best Error is 0): ', 1 - ACC)

SN = TPR / (y_hats.size(0) - (TNR + FNR))
print('The Sensitivity of the Predicted Values is (Best Sensitivity is 1): ', SN)

SP = TNR / (y_hats.size(0) - (TPR + FPR))
print('The Specificity of the Predicted True Negative Values is: ', SP)

PPV = TPR / (TPR + FPR)
print('The Precision (positive predictive value) is (Best Precision is 1): ', PPV)

Falsepositive = FNR / (y_hats.size(0) - (TPR + FPR))
print('The False Positive Rate is: ', Falsepositive)
