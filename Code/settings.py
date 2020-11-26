### SETTINGS ###

TRAIN_TEST_SPLIT = 0.7  # Split into test and training sets
N_LAYERS = 3            # Used to tweak the models performance. Defines how many convolutional layers the CNN will have
EPOCHS = 100             # Each epoch iterate over all images in the training set
BATCH_SIZE = 200        # Batches are needed because all the data cannot fit into the GPU memory at once
                        # If the total number of images is 1000 and the batch size is 100, this requires 10 iterations per epoch
START_STEP = 20         # Starting neurons
STEPS = 30              # Additional neurons per step
KERNEL = (3, 3)        

PATIENCE = 10           # Tells the model to stop training if no improvement is made after PATIENCE amount of epochs

# Directory of the images
SOURCE_DIR = 'C:\\Users\\epa13\\Documents\\NTNU\\TDT4173_Maskinlæring\\Project\\Datasett\\planesnet\\planesnet\\planesnet'
TRAIN_DIR = "C:\\Users\\epa13\\Documents\\NTNU\\TDT4173_Maskinlæring\\Project\\Datasett\\planesnet\\planesnet\\training_set"
TEST_DIR = "C:\\Users\\epa13\\Documents\\NTNU\\TDT4173_Maskinlæring\\Project\\Datasett\\planesnet\\planesnet\\test_set"
# TensorBoard callback
LOG_DIRECTORY_ROOT = 'C:\\Users\\epa13\\Documents\\NTNU\\TDT4173_Maskinlæring\\Project\\Code\\log_dir'

IS_SHUFFLED = False

def print_setting():
    print("____Settings____")
    print("Train Test Split: ", TRAIN_TEST_SPLIT)
    print("-------------------------")
    print("N Layers: ", N_LAYERS)
    print("-------------------------")
    print("Epochs: ", EPOCHS)
    print("-------------------------")
    print("Batch Size: ", BATCH_SIZE)
    print("-------------------------")
    print("Start Step: ", START_STEP)
    print("-------------------------")
    print("Steps: ", STEPS)
    print("-------------------------")
    print("Kernel: ", KERNEL)
    print("-------------------------")
    print("Patience: ", PATIENCE)
    print("-------------------------")