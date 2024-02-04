### For testing only

import torch
import numpy as np
from sklearn.decomposition import PCA

DATA = "/Users/.../Projects/Dalhousie_MRI_Neural_Data_Hackathon/tensors"

elr = torch.load(DATA + "/elr.pt") # (904, 368,176)
elr_array = elr.numpy()

elrStack = []
for layer in range(0, 905):
    pca = PCA(n_components=1)
    components = pca.fit_transform(elr_array[layer])
    elrStack.append(components)

elrStack_array = np.array(elrStack)
