import os 
from pathlib import Path
import torch

# path to data files
DATA = "../neural_data"

# goes into specified folder, and extracts timeseries data to tensor
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


def get_emotion_data():
    elr_stack = []
    erl_stack = []
    for subject in os.listdir(DATA):
        elr = get_tensor_from(subject, "EMOTION_LR")
        erl = get_tensor_from(subject, "EMOTION_RL")

        # validate
        if elr.shape == erl.shape == (368, 176): 
            elr_stack.append(elr)
            erl_stack.append(erl)

    # labels
    ranges = [(0,12,0), (12,16,1), (16,40,2), (40,44,1), (44,68,2), (68,72,1), (72,96,2), (100,124,1), (128,152,2), (152,156,1),(156,176,2)]
    emotion_labels = torch.empty((368,176))
    for start, end, label in ranges:
        emotion_labels[:, start:end] = label

    elr_tensors = torch.stack(elr_stack) # (NUM, 368,176)
    erl_tensors = torch.stack(erl_stack)

    return elr_tensors, erl_tensors, emotion_labels


# (368, 284)
def get_motor_data():
    mlr_stack = []
    mrl_stack = []
    for subject in os.listdir(DATA):
        mlr = get_tensor_from(subject, "MOTOR_LR")
        mrl = get_tensor_from(subject, "MOTOR_RL")
        # validate
        if mlr.shape == mrl.shape == (368, 284): 
            mlr_stack.append(mlr)
            mrl_stack.append(mrl)

    ranges = [
        (0, 12, 0), 
        (12, 16, 1),
        (16, 32, 2),
        (32, 36, 1),
        (36, 52, 2),
        (52, 56, 1),
        (56, 72, 2),
        (72, 76, 1),
        (76, 92, 2),
        (92, 96, 1),
        (96, 112, 2),
        (112, 116, 1),
        (116, 132, 2),
        (132, 152, 0),
        (152, 156, 1),
        (156, 172, 2),
        (172, 176, 1),
        (176, 192, 2),
        (192, 212, 0),
        (212, 216, 1),
        (216, 232, 2),
        (232, 236, 1),
        (236, 252, 2),
        (252, 284, 0)
    ]
    motor_labels = torch.empty((368,284))
    for start, end, label in ranges:
        motor_labels[:, start:end] = label
    
    mlr_tensors = torch.stack(mlr_stack)
    mrl_tensors = torch.stack(mrl_stack)
    return mlr_tensors, mrl_tensors, motor_labels


def get_language_data():
    llr_stack = []
    lrl_stack = []
    for subject in os.listdir(DATA):
        llr = get_tensor_from(subject, "LANGUAGE_LR")
        lrl = get_tensor_from(subject, "LANGUAGE_RL")
        # validate
        if llr.shape == lrl.shape == (368, 316): 
            llr_stack.append(llr)
            lrl_stack.append(lrl)

    ranges = [
        (0, 36, 2),
        (36, 40, 1),
        (40, 76, 2),
        (76, 80, 1),
        (80, 116, 2),
        (116, 120, 1),
        (120, 156, 2),
        (156, 160, 1),
        (160, 196, 2),
        (196, 200, 1),
        (200, 236, 2),
        (236, 240, 1),
        (240, 276, 2),
        (276, 280, 1),
        (280, 316, 2)
    ]
    language_labels = torch.empty((368, 316))
    for start, end, label in ranges:
        language_labels[:, start:end] = label

    llr_tensors = torch.stack(llr_stack) 
    lrl_tensors = torch.stack(lrl_stack)
    return llr_tensors, lrl_tensors, language_labels


def get_relational_data():
    rlr_stack = []
    rrl_stack = []
    for subject in os.listdir(DATA):
        rlr = get_tensor_from(subject, "RELATIONAL_LR")
        rrl = get_tensor_from(subject, "RELATIONAL_RL")
        # validate
        if rlr.shape == rrl.shape == (368, 232):
            rlr_stack.append(rlr)
            rrl_stack.append(rrl)

    ranges = [
        (0, 12, 0),
        (12, 18, 1),
        (18, 38, 2),
        (38, 44, 1),
        (44, 64, 2),
        (64, 84, 0),
        (84, 90, 1),
        (90, 110, 2),
        (110, 116, 1),
        (116, 136, 2),
        (136, 156, 0),
        (156, 162, 1),
        (162, 168, 2),
        (168, 188, 1),
        (188, 208, 2),
        (208, 232, 0)
    ]
    relational_labels = torch.empty((368,232))
    for start, end, label in ranges:
        relational_labels[:, start:end] = label

    rlr_tensors = torch.stack(rlr_stack) 
    rrl_tensors = torch.stack(rrl_stack)
    return rlr_tensors, rrl_tensors, relational_labels


def get_social_data():
    slr_stack = srl_stack = []
    for subject in os.listdir(DATA):
        slr = get_tensor_from(subject, "SOCIAL_LR")
        srl = get_tensor_from(subject, "SOCIAL_RL")
        if slr.shape == srl.shape == (368, 274):
            slr_stack.append(slr)
            srl_stack.append(srl)

    ranges = [
        (0, 12, 0),
        (12, 44, 2),
        (44, 64, 1),
        (64, 96, 2),
        (96, 116, 1),
        (116, 148, 2),
        (148, 168, 1),
        (168, 200, 2),
        (200, 220, 1),
        (220, 252, 2),
        (252, 274, 1)
    ]
    social_labels = torch.empty((368, 274))
    for start, end, label in ranges:
        social_labels[:, start:end] = label
    
    slr_tensors = torch.stack(slr_stack)
    srl_tensors = torch.stack(srl_stack)
    return slr_tensors, srl_tensors, social_labels

def get_gambling_data():
    glr_stack = grl_stack = []
    for subject in os.listdir(DATA):
        glr = get_tensor_from(subject, "GAMBLING_LR")
        grl = get_tensor_from(subject, "GAMBLING_RL")
        if glr.shape == grl.shape == (368, 253):
            glr_stack.append(glr)
            grl_stack.append(grl)

    ranges = [
        (0, 12, 0),
        (12, 52, 2),
        (52, 72, 1),
        (72, 112, 2),
        (112, 132, 1),
        (132, 172, 2),
        (172, 192, 1),
        (192, 232, 2),
        (232, 253, 1)
    ]
    gambling_labels = torch.empty((368, 253))
    for start, end, label in ranges:
        gambling_labels[:, start:end] = label
    
    glr_tensors = torch.stack(glr_stack)
    grl_tensors = torch.stack(grl_stack)
    return glr_tensors, grl_tensors, gambling_labels


def get_wm_data():
    wlr_stack = wrl_stack = []
    for subject in os.listdir(DATA):
        wlr = get_tensor_from(subject, "WM_LR")
        wrl = get_tensor_from(subject, "WM_RL")
        if wlr.shape == wrl.shape == (368, 405):
            wlr_stack.append(wlr)
            wrl_stack.append(wrl)

    ranges = [
        (0, 12, 0),
        (12, 50, 2),
        (50, 88, 2),
        (88, 110, 1),
        (110, 148, 2),
        (148, 186, 2),
        (186, 208, 1),
        (208, 246, 2),
        (246, 284, 2),
        (284, 306, 1),
        (306, 344, 2),
        (344, 382, 2),
        (382, 405, 1)
    ]
    wm_labels = torch.empty((368, 405))
    for start, end, label in ranges:
        wm_labels[:, start:end] = label

    wlr_tensors = torch.stack(wlr_stack)
    wrl_tensors = torch.stack(wrl_stack)
    return wlr_tensors, wrl_tensors, wm_labels