# CortexCrusaders

**Directory Structure**

```
- binary_xgb.py - XGBoost binary classifier training code
- calculation.py - calculate and plot means
- gridsearchcv.py - run gridsearch params for single XGBoost classifier
- pca.py - PCA
- plot.py - calculate and plot means
- preprocessing.py - find top 5 brain areas
- process.py - converting data files to torch tensors
- save_tensors.py - save data files as torch tensors
- toke.py - old processing code, use save_tensors.py
- xgb.py - single XGBoost classifier for single task
```


**Steps**

In `process.py`, change `DATA=...` to point to the location of your data.

Ensure the `tensors` folder exists
```
mkdir tensors
```

Then run `save_tensors.py` to save the data files in torch tensor format, which can be used as input for models.
