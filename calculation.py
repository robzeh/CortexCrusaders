import torch
from process import get_emotion_data, get_motor_data, get_language_data, get_relational_data, get_social_data, get_gambling_data, get_wm_data

fixation_state_tensors = {
    "emotion": get_emotion_data()[0],  # Assuming the first returned value is the LR tensor
    "motor": get_motor_data()[0],
    "language": get_language_data()[0],
    "relational": get_relational_data()[0],
    "social": get_social_data()[0],
    "gambling": get_gambling_data()[0],
    "wm": get_wm_data()[0]
}

resting_state_tensors = {
    "emotion": get_emotion_data()[1],  # Assuming the second returned value is the RL tensor
    "motor": get_motor_data()[1],
    "language": get_language_data()[1],
    "relational": get_relational_data()[1],
    "social": get_social_data()[1],
    "gambling": get_gambling_data()[1],
    "wm": get_wm_data()[1]
}

task_state_tensors = {
    "emotion": get_emotion_data()[2],  # Assuming the first returned value is the LR tensor
    "motor": get_motor_data()[2],
    "language": get_language_data()[2],
    "relational": get_relational_data()[2],
    "social": get_social_data()[2],
    "gambling": get_gambling_data()[2],
    "wm": get_wm_data()[2]
}

#for task_key in resting_state_tensors.keys():
#    # Access and print the value for the current task key
#    print(f"Values for task '{task_key}':")
#    print(resting_state_tensors[task_key])
#    print()  # Add a newline for better readability


# Calculate (T-F) - (R-F) for each task
result_tensors = {}
for task in resting_state_tensors.keys():
    result_tensors[task] = (task_state_tensors[task] - fixation_state_tensors[task]) - (resting_state_tensors[task] - fixation_state_tensors[task])
    
    # Save the result tensors
    torch.save(result_tensors[task], f"/home/user002/projects/def-sponsor00/user002/tensors/{task}_result.pt")

