# Deep_Learning_CNN

How to set up the CNN implementation in python.

1. Install required packages
- These are the imports used for this implementation, install those that you have not already installed using "pip install" for each package.

IMPORTS
- import time
- import glob
- import numpy as np
- import os.path as path
- import matplotlib.pyplot as plt
- import tensorflow as tf
- from scipy import misc
- from matplotlib.pyplot import imread
- from datetime import datetime
- from sklearn.metrics import accuracy_score, f1_score
- from keras.models import Sequential
- from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
- from keras.callbacks import EarlyStopping, TensorBoard
- from sklearn.metrics import confusion_matrix
- from sklearn.metrics import classification_report

2. Adjust the parameters in the settings.py file to be the same as the test you wish to run.
- Remember to change the folder path to the training and data set.

3. To get deterministic results similar to what we have done, use a seed of 7. This should be the standard seed already set.

4. Run the python file and the results should come after some time depending on what settings are used. To remove the progress bar during training, set the verbose parameter in model.fit() to be 0.

(OPTIONAL)
5. To view the training progress in tensorboard, install CUDA developer kit on your computer (https://developer.nvidia.com/cuda-downloads)
- Create a log directory (empty folder)
- Set the LOG_DIRECTORY_ROOT in the settings.py file to the newly created directory
- Run this line in your command terminal: tensorboard --logdir "path/to/directory"
- Open tensorboard on http://localhost:6006/
