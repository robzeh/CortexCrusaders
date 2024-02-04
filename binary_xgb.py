import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch


# reshapes and converts all F/R to 0, T=1
def reshape_labels(og_data, labels):
    labels = labels[0] # (368,TIME_STEPS) -> (TIME_STEPS)
    labels = ((labels == 2).float()).type(torch.LongTensor) # convert all F/R to 0, T=1
    labels = labels.expand((len(og_data),labels.size(0))) # (NUM_SAMPLES,TIMESTEPS)
    labels = labels.reshape((-1,)) # (NUM_SAMPLES*TIMESTEPS)
    return labels

def reshape_data(data):
    # case of emtion: (904,368,176) -> (904*176, 368)
    # each time step of each subject becomes an individual sample
    return data.view((data.size(0)*data.size(2), data.size(1)))

def train_xgboost(data, labels):
    # time based spltting
    train_ratio = 0.70
    num_samples_train = int(len(data) * train_ratio) # get first 70% samples as training
    X_train = data[:num_samples_train]
    X_test = data[num_samples_train:]
    y_train = labels[:num_samples_train] # last 30% samples as testing
    y_test = labels[num_samples_train:]

    # sample splitting
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    # Preparing the data for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # hyperparamter tuning
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'eta': 0.03,
        'n_estimators': 300
    }
    num_rounds = 1000
    bst = xgb.train(params, dtrain, num_rounds)

    preds = bst.predict(dtest)
    predictions = np.asarray([np.argmax(line) for line in preds])

    # Evaluating the predictions
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")


### Configure data to load here, uncomment desired training data

# emotion
# elr = torch.load("./tensors/elr.pt") # (904, 368,176)
# erl = torch.load("./tensors/erl.pt") # (904, 368,176)
# e_labels = torch.load("./tensors/emotion_labels.pt")
# elr_data = reshape_data(elr)
# erl_data = reshape_data(erl)
# emotion_labels = reshape_labels(elr, e_labels)

# print("training on EMOTION_LR")
# train_xgboost(elr_data, emotion_labels)
# print("training on EMOTION_RL")
# train_xgboost(erl_data, emotion_labels)
# torch.cuda.empty_cache()


# # motor
# mlr = torch.load("./tensors/mlr.pt")
# mrl = torch.load("./tensors/mrl.pt")
# m_labels = torch.load("./tensors/motor_labels.pt")
# mlr_data = reshape_data(mlr)
# mrl_data = reshape_data(mrl)
# motor_labels = reshape_labels(mlr, m_labels)

# print("training on MOTOR_LR")
# train_xgboost(mlr_data, motor_labels)
# print("training on MOTOR_RL")
# train_xgboost(mrl_data, motor_labels)
# torch.cuda.empty_cache()


# language
llr = torch.load("./tensors/llr.pt")
lrl = torch.load("./tensors/lrl.pt")
l_labels = torch.load("./tensors/language_labels.pt")
llr_data = reshape_data(llr)
lrl_data = reshape_data(lrl)
language_labels = reshape_labels(llr, l_labels)

print("training on LANGUAGE_LR")
train_xgboost(llr_data, language_labels)
print("training on LANGUAGE_RL")
train_xgboost(lrl_data, language_labels)


# relational
rlr = torch.load("./tensors/rlr.pt")
rrl = torch.load("./tensors/rrl.pt")
r_labels = torch.load("./tensors/relational_labels.pt")
rlr_data = reshape_data(rlr)
rrl_data = reshape_data(rrl)
relational_labels = reshape_labels(rlr, r_labels)

print("training on RELATIONAL_LR")
train_xgboost(rlr_data, relational_labels)
print("training on RELATIONAL_RL")
train_xgboost(rrl_data, relational_labels)
torch.cuda.empty_cache()


# social
slr = torch.load("./tensors/slr.pt")
srl = torch.load("./tensors/srl.pt")
s_labels = torch.load("./tensors/social_labels.pt")
slr_data = reshape_data(slr)
srl_data = reshape_data(srl)
social_labels = reshape_labels(slr, s_labels)

print("training on SOCIAL_LR")
train_xgboost(slr_data, social_labels)
print("training on SOCIAL_RL")
train_xgboost(srl_data, social_labels)
torch.cuda.empty_cache()


# gambling
glr = torch.load("./tensors/glr.pt")
grl = torch.load("./tensors/grl.pt")
g_labels = torch.load("./tensors/gambling_labels.pt")
glr_data = reshape_data(glr)
grl_data = reshape_data(grl)
gambling_labels = reshape_labels(glr, g_labels)

print("training on GAMBLING_LR")
train_xgboost(glr_data, gambling_labels)
print("training on GAMBLING_RL")
train_xgboost(grl_data, gambling_labels)
torch.cuda.empty_cache()


# wm
wlr = torch.load("./tensors/wlr.pt")
wrl = torch.load("./tensors/wrl.pt")
w_labels = torch.load("./tensors/wm_labels.pt")
wlr_data = reshape_data(wlr)
wrl_data = reshape_data(wrl)
wm_labels = reshape_labels(wlr, w_labels)

print("training on WM_LR")
train_xgboost(wlr_data, wm_labels)
print("training on WM_RL")
train_xgboost(wrl_data, wm_labels)
torch.cuda.empty_cache()

### pass data and train here







