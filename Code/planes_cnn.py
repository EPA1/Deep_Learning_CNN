# Imports
import time
import glob
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import misc
from matplotlib.pyplot import imread
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Imports from settings
from settings import TRAIN_TEST_SPLIT, N_LAYERS, EPOCHS, BATCH_SIZE, START_STEP, STEPS, KERNEL, PATIENCE, SOURCE_DIR, TRAIN_DIR, TEST_DIR, LOG_DIRECTORY_ROOT, IS_SHUFFLED, print_setting

# Set seed to get deterministic results
tf.random.set_seed(7) 

# Load training data
train_paths = glob.glob(path.join(TRAIN_DIR, "*.png"))
training_images = [imread(path) for path in train_paths]
training_images = np.asarray(training_images, dtype="float32")

# Load test data
test_paths = glob.glob(path.join(TEST_DIR, "*.png"))
test_images = [imread(path) for path in test_paths]
test_images = np.asarray(test_images, dtype="float32")

# Read the labels from the filenames in the training data set
train_n = training_images.shape[0]
train_labels = np.zeros(train_n)
for i in range(train_n):
    filename = path.basename(train_paths[i])[0]
    train_labels[i] = int(filename[0])

# Read the labels from the filenames in the test data set
test_n = test_images.shape[0]
test_labels = np.zeros(test_n)
for i in range(test_n):
    filename = path.basename(test_paths[i])[0]
    test_labels[i] = int(filename[0])

# Create indicies for training and testing data
train_indicies = np.arange(train_n)
test_indicies = np.arange(test_n)

# Set training and testing variables to be sent into the model
x_train = training_images[train_indicies]
y_train = train_labels[train_indicies]
x_test = test_images[test_indicies]
y_test = test_labels[test_indicies]

# Sequential model for linear stack of layers
model = Sequential()

# Define input shape
shape = (20, 20, 3)

# Convolutional layers
layer_index = 0
first_layer = True
while layer_index <= N_LAYERS:
    if first_layer:
        model.add(Conv2D(START_STEP, KERNEL, input_shape=shape, padding="valid"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        first_layer = False
    else:
        model.add(Conv2D(START_STEP + STEPS * layer_index, KERNEL, padding="valid"))
        model.add(Activation("relu"))
    layer_index += 1

# Max pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=120))
model.add(Activation('relu'))

# Output layer
model.add(Dense(units=1))
model.add(Activation('sigmoid')) 

# Compile the model
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

# Early stopping callback
# Checks in on the model as it trains
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# Create logs for TensorBoard visualization
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)  
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

# Place the callbacks in a list
callbacks = [early_stopping, tensorboard]

# Train the model
start_time = time.time()
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)
total_time = time.time() - start_time

# Make a prediction on the test set
test_predictions = model.predict(x_test)
test_predictions = np.round(test_predictions)
y_pred_bool = np.argmax(test_predictions, axis=1)

# Summary of the model
model.summary()

# Accuracy score
accuracy_frac = accuracy_score(y_test, test_predictions)
accuracy_count = accuracy_score(y_test, test_predictions, normalize=False)
print("____ACCURACY____")
print(str(accuracy_frac))
print("-------------------------")
print(str(accuracy_count), "/", str(len(y_test)))
print("-------------------------")

# Time
print("____TIME____")
print(str(total_time))
print("-------------------------")

# Print settings
print_setting()

# precision, recall and f1score
print(classification_report(y_test, np.asarray(test_predictions).ravel(), labels=[0, 1]))

