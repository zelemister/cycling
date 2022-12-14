from transformations import get_transformer
import torch.optim as optim
# Expermentname
EXPERMIMENT_NAME = "RIM_512_Test"
NUM_EPOCHS = 100
OVERSAMPLING_RATE = 10

#image resolution
RESOLUTION = 256

# oversampling transformation
TRANSFORMATION = get_transformer("rotations", resolution=RESOLUTION)
#"bikelane" or "rim
TASK = "bikelane"

#transformer or resnet (resnet version maybe?
MODEL = "resnet"
PRETRAINED = True
#"full" or "head"
PARAMS = "full"

#validation set ratio
VAL_SET_RATIO = 0.2
#weights
WEIGHTS = [1,1] # as class distribution. 1879 negativs, 110 positives.
#optimizer function
OPTIMIZER = optim.RMSprop
#repetitions

# Learning rate parameters
BASE_LR = 0.001
EPOCH_DECAY = NUM_EPOCHS +1 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1 # factor by which the learning rate is reduced.


