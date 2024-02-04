import xgboost as xgb
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch

elr = torch.load("./tensors/elr.pt") # (368,176)
first_sample = elr[0].transpose(0,1) # (176,368)
print(first_sample.shape)

labels = torch.load("./tensors/emotion_labels.pt")
print(labels[0].shape) # (176)

# train/test splits
train_ratio = 0.75
test_ratio = 1 - train_ratio
num_samples_train = int(len(first_sample) * train_ratio) # get first 75% samples as training
num_samples_test = len(first_sample) - num_samples_train # get last 25% samples as testing

X_train = first_sample[:num_samples_train]
X_test = first_sample[num_samples_train:]
y_train = labels[0][:num_samples_train]
y_test = labels[0][num_samples_train:]

# Preparing the data for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Setting up the parameters for XGBoost
# For a multi-class classification problem, you need to set 'objective' to 'multi:softprob'
# and 'num_class' to the number of classes
params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 4,
    'eta': 0.001,
    'eval_metric': 'mlogloss',
}

num_rounds = 1000
bst = xgb.train(params, dtrain, num_rounds)

preds = bst.predict(dtest)
print(y_test)
print(preds)
predictions = np.asarray([np.argmax(line) for line in preds])

# Evaluating the predictions
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")



# for i in range(1, elr.shape[0]-2):
#     sample = elr[i].transpose(0,1) # (176,368)
#     print(sample.shape)
#     label = labels[i]

#     X_train = sample[:num_samples_train]
#     X_test = sample[num_samples_train:]
#     y_train = label[:num_samples_train]
#     y_test = label[num_samples_train:]

#     # Preparing the data for XGBoost
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dtest = xgb.DMatrix(X_test, label=y_test)

#     # update
#     bst_updated = xgb.Booster()
#     bst_updated.load_model("tree_model")

#     # Making predictions
#     preds = bst.predict(dtest)
#     # print(y_test)
#     # print(preds)
#     predictions = np.asarray([np.argmax(line) for line in preds])

#     # Evaluating the predictions
#     accuracy = accuracy_score(y_test, predictions)
#     print(f"Accuracy: {accuracy * 100:.2f}%")