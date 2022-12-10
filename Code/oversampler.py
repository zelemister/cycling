import os
import torch.nn as nn
import torch
import PIL.Image
from torchvision import transforms
import numpy.random as random
from torchvision.utils import save_image
image_folder = "../Data/bikelane"

seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
rotate = transforms.Compose([transforms.ToTensor(),
                             #generate a random number between 0 and 1, then multiply by 360 for a random number between 0 and 360
                             transforms.RandomRotation(degrees=random.random(1)[0]*360),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip()])


factor = 10
for root, dirs, files in os.walk(image_folder):
    if root.endswith("train/1"):
        for file in files:
            image = PIL.Image.open(os.path.join(root, file))
            oversampled_list = list(map(rotate, [image]*factor))
            for i in range(factor):
                save_image(oversampled_list[i],  os.path.join(root, file[0:-4] + "_" + str(i) + ".png"))
            os.remove(os.path.join(root, file))
    elif root.endswith("0") or root.endswith("val/1"):
        for file in files:
            image = PIL.Image.open(os.path.join(root, file))
            alt_image = rotate(image)
            save_image(alt_image, os.path.join(root, file[0:-4] + "_" + "1" + ".png"))
            os.remove(os.path.join(root, file))





