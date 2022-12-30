from transformations import get_transformer
import torch.optim as optim
# Expermentname
EXPERMIMENT_NAME = "Test_Config_Overhall"
NUM_EPOCHS = 80
OVERSAMPLING_RATE = 10

#image resolution
RESOLUTION = 256
BATCH_SIZE = 64 # 64 for 256 works, for 512 probably lower.
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
EPOCH_DECAY = 40 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1 # factor by which the learning rate is reduced.

