import torch
import matplotlib.pyplot as plt
import numpy as np

tasks = ["emotion", "motor", "language", "relational", "social", "gambling", "wm"]

# Dictionary to store mean values for each task
mean_values_dict = {}

for task in tasks:
    result_tensor = torch.load(f"/home/user002/projects/def-sponsor00/user002/tensors/{task}_result.pt")
    mean_values = torch.mean(result_tensor, dim=0).numpy()
    mean_values_dict[task] = mean_values

    # Print basic statistics
    print(f"Task: {task}")
    print(f"Mean: {np.mean(mean_values)}")
    print(f"Standard Deviation: {np.std(mean_values)}")
    print(f"Min: {np.min(mean_values)}")
    print(f"Max: {np.max(mean_values)}")
    print()
    plt.plot(mean_values)
    plt.xlabel('Brain Areas')
    plt.ylabel('Mean Result Value')
    plt.title(f"Mean Result Tensor Across Brain Areas for Task")
    plt.savefig(f"/home/user002/projects/def-sponsor00/user002/plots/{task}_mean_values.png")

# Plot the mean values for each brain area
#result_tensor = torch.load("/home/user002/projects/def-sponsor00/user002/tensors/emotion_result.pt")

# Calculate mean values across the first dimension (905 samples)
#mean_values = torch.mean(result_tensor, dim=0).numpy()

#plt.savefig('/home/user002/projects/def-sponsor00/user002/plots/emotion_mean_values.png')
