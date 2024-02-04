import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from binary_xgb import reshape_labels, reshape_data
import torch

DATA = "/Users/.../Projects/Dalhousie_MRI_Neural_Data_Hackathon/tensors"

elr = torch.load(DATA + "/elr.pt") # (904, 368,176)
e_labels = torch.load(DATA + "/emotion_labels.pt")
elr_data = reshape_data(elr)
emotion_labels = reshape_labels(elr, e_labels)

train_ratio = 0.70
test_ratio = 1 - train_ratio
num_samples_train = int(len(elr_data) * train_ratio) # get first 70% samples as training
num_samples_test = len(elr_data) - num_samples_train # get last 30% samples as testing
X_train = elr_data[:num_samples_train]
X_test = elr_data[num_samples_train:]
y_train = emotion_labels[:num_samples_train]
y_test = emotion_labels[num_samples_train:]

xgb_model = xgb.XGBClassifier()
params = {
        'objective': ['binary:logistic'],
        'max_depth': [5, 6],
        'eta': [0.1, 0.2, 0.3],
        'n_estimators': [100, 150, 200, 250, 300]
    }

grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, scoring='accuracy', cv=5, n_jobs=-1,
                               verbose=2)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)
