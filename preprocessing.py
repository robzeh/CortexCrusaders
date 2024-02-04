import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import detrend
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from collections import Counter

tasks = ["emotion", "motor", "language", "relational", "social", "gambling", "wm"]

# Dictionary to store mean values for each task
mean_values_dict = {}

def detrend(data):
    detrended_data = np.apply_along_axis(lambda x: np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x))), axis=0, arr=data)
    return detrended_data

def smooth(data, window_size=5):
    smoothed_data = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='valid'), axis=0, arr=data)
    return smoothed_data

mean_values_list = []

# Dictionary to store indices of the top 10 values for each task
top_10_indices_dict = {}

for task in tasks:
    result_tensor = torch.load(f"/home/user002/projects/def-sponsor00/user002/tensors/{task}_result.pt")
    
    scaler = MinMaxScaler()
    flattened_data = result_tensor.view(-1, result_tensor.shape[-1])
    flattened_data_2d = flattened_data.reshape(-1, flattened_data.shape[-1])
    scaled_data_2d = scaler.fit_transform(flattened_data_2d)
    scaled_data = scaled_data_2d.reshape(result_tensor.shape[:-1] + (scaled_data_2d.shape[-1],))
    scaled_result_tensor = torch.tensor(scaled_data)

    detrended_data = detrend(scaled_result_tensor.numpy())
    detrended_result_tensor = torch.tensor(detrended_data)      

    window_length = 5 
    polyorder = 2  
    smoothed_data = savgol_filter(detrended_result_tensor.numpy(), window_length, polyorder, axis=-1)
    smoothed_result_tensor = torch.tensor(smoothed_data)

    mean_values = torch.mean(smoothed_result_tensor, dim=0).numpy()
    mean_values_dict[task] = mean_values
    mean_values_list.append(mean_values)

    min_length = min(len(arr) for arr in mean_values_list)
    trimmed_mean_values = [arr[:min_length] for arr in mean_values_list]

    sorted_indices = np.argsort(mean_values)[::-1]
    top_10_indices = sorted_indices[:10]
    top_10_indices_dict[task] = top_10_indices

    flat_indices = [index for sublist in top_10_indices for index in sublist]
    index_counter = Counter(flat_indices)
    top_5_indices = index_counter.most_common(5)

    print(f"Task: {task}")
    print(f"Mean: {np.mean(mean_values)}")
    print(f"Standard Deviation: {np.std(mean_values)}")
    print(f"Min: {np.min(mean_values)}")
    print(f"Max: {np.max(mean_values)}")
    print(f"Top 5 most common indices: {top_5_indices}")
    print()
    plt.plot(mean_values, label=task)
    plt.xlabel('Brain Areas')
    plt.ylabel('Mean Result Value')
    plt.title(f"Mean Result Tensor Across Brain Areas for Task")
    plt.savefig(f"/home/user002/projects/def-sponsor00/user002/standardizedplots/{task}_mean_values.png")

# Perform t-tests between pairs of tasks
for i in range(len(tasks)):
    for j in range(i + 1, len(tasks)):
        task_1 = tasks[i]
        task_2 = tasks[j]
        print(mean_values_dict[task_1][:min_length].shape, mean_values_dict[task_2][:min_length].shape)
        # Flatten the arrays into one dimension
        array1 = np.array(mean_values_dict[task_1]).flatten()
        array2 = np.array(mean_values_dict[task_2]).flatten()
        # Run the t-test
        t_statistic, p_value = ttest_ind(array1, array2)
        print(f’Two-sample t-test for {task_1} vs {task_2}:‘)
        print(f’T-statistic: {t_statistic}‘)
        print(f’P-value: {p_value}’)
