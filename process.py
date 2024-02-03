import os 
from pathlib import Path
import torch

# path to data files
#DATA = "../neural_data"
DATA = "projects/def-sponsor00/hcp1200/data"

def get_tensor_from(folder, split):
    rows = []
    if Path(f"{DATA}/{folder}/tfMRI_{split}/timeseries").is_file():
        with open(f"{DATA}/{folder}/tfMRI_{split}/timeseries", "r") as d:
            for l in d:
                row = l.strip().split(" ")
                row = [float(s) for s in row]
                rows.append(row)
        return torch.tensor(rows)
    else:
        pass


# for each subject, append tensors to array. then build stack at very end
elr_stack = erl_stack = []
glr_stack = grl_stack = []
llr_stack = lrl_stack = []
mlr_stack = mrl_stack = []
rlr_stack = rrl_stack = []
slr_stack = srl_stack = []
wlr_stack = wrl_stack = []

# example first 5 
for subject in os.listdir(DATA)[:5]:
    elr = get_tensor_from(subject, "EMOTION_LR")
    #erl = get_tensor_from(subject, "EMOTION_RL")
    elr_stack.append(elr)
    #erl_stack.append(erl)

    # glr = get_tensor_from(subject, "GAMBLING_LR")
    # grl = get_tensor_from(subject, "GAMBLING_RL")
    # glr_stack.append(glr)
    # grl_stack.append(grl)

    # llr = get_tensor_from(subject, "LANGUAGE_LR")
    # lrl = get_tensor_from(subject, "LANGUAGE_RL")
    # llr_stack.append(llr)
    # lrl_stack.append(lrl)

    # mlr = get_tensor_from(subject, "MOTOR_LR")
    # mrl = get_tensor_from(subject, "MOTOR_RL")
    # mlr_stack.append(mlr)
    # mrl_stack.append(mrl)


    # rlr = get_tensor_from(subject, "RELATIONAL_LR")
    # rrl = get_tensor_from(subject, "RELATIONAL_RL")
    # rlr_stack.append(rlr)
    # rrl_stack.append(rrl)

    # slr = get_tensor_from(subject, "SOCIAL_LR")
    # srl = get_tensor_from(subject, "SOCIAL_RL")
    # slr_stack.append(slr)
    # srl_stack.append(srl)

    # wlr = get_tensor_from(subject, "WM_LR")
    # wrl = get_tensor_from(subject, "WM_RL")
    # wlr_stack.append(wlr)
    # wrl_stack.append(wrl)

elr_tensors = torch.stack(elr_stack) # (NUM, 368,176)
#erl_tensors = torch.stack(erl_stack)

# 0=F, 1=R, 2=T
# emotion
ranges = [(0,12,0), (12,16,1), (16,40,2), (40,44,1), (44,68,2), (68,72,1), (72,96,2), (100,124,1), (128,152,2), (152,156,1),(156,176,2)]
emotion_labels = torch.empty((368,176))
for start, end, label in ranges:
    emotion_labels[:, start:end] = label

print(emotion_labels)

# Get the indices where the labels are 2 (T), 1 (R), and 0 (F)
indices_T = torch.eq(emotion_labels, 2)
indices_R = torch.eq(emotion_labels, 1)
indices_F = torch.eq(emotion_labels, 0)

print("Indices for T:")
print(indices_T)

print("Indices for R:")
print(indices_R)

print("Indices for F:")
print(indices_F)

# Perform the calculation (2-0) - (1-0) on the actual values
result_tensors = (elr_tensors) * indices_T - (elr_tensors) * indices_R
print(result_tensors)
