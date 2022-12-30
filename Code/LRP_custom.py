#Resources
#https://github.com/sebastian-lapuschkin/lrp_toolbox
#https://github.com/chr5tphr/zennit
#Tutorial: https://zennit.readthedocs.io/en/latest/tutorial/image-classification-vgg-resnet.html

#Aim: create heatmap of model predicitions and analyze false positives

import torch
import numpy as np
import matplotlib.pyplot as plt
#import cv2
from PIL._imaging import display
from torch.nn import Linear
from PIL import Image
#import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms import ToTensor, Normalize, ToPILImage
from torchvision.models import resnet18
from zennit.attribution import Gradient, SmoothGrad
from zennit.core import Stabilizer
from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat
from zennit.composites import SpecialFirstLayerMapComposite, NameMapComposite
from zennit.image import imgify, imsave
from zennit.rules import Epsilon, ZPlus, ZBox, Norm, Pass, Flat
from zennit.types import Convolution, Activation, AvgPool, Linear as AnyLinear
from zennit.types import BatchNorm, MaxPool
from zennit.torchvision import ResNetCanonizer
from model_loaders import get_model

"""
torch.hub.download_url_to_file(
    'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/2006_09_06_180_Leuchtturm.jpg/640px-2006_09_06_181_Leuchtturm.jpg',
    '../Example Images/dornbusch-lighthouse.jpg',
)
Testbild Leuchtturm. Beispiel RIM: x_18030025
"""
"""
def show(imgs):
    #function to plot a list of tensor images from pytorch tutorial
    #unused
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
"""

# define the base image transform
transform_img = Compose([
    Resize(256),
    CenterCrop(224),
])
# define the normalization transform
transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# define the full tensor transform
transform = Compose([
    transform_img, #resize and crop
    ToTensor(),
    transform_norm, #normalize
])

# load the image
# image = Image.open('../Example Images/dornbusch-lighthouse.jpg')
# #image = Image.open('../Example Images/x_18040012.png')
# image = Image.open('../Images/x_18030025.png')
image = Image.open('../Images/x_23040044.png')
# image = Image.open('../Images/x_77030014.png')
"""
0,x_18020087_1.png
1,x_23040044_1.png
2,x_10040055_1.png
3,x_77030014_1.png
4,x_15030078_1.png
5,x_23030052_1.png
"""

# transform the PIL image and insert a batch-dimension
data = transform(image)[None]
#print(data.shape)
#print(data_PIL)
# display the resized and cropped image
#data_PIL = ToPILImage()(data) #only works with unbatched data = transform(image)
#data_PIL.show()

# load the model and set it to evaluation mode
# model = resnet18(weights=None).eval()
# model = resnet18().eval()
model = get_model("resnet", pretrained=False)
model.load_state_dict(torch.load("../Models" + "/fine_tuned_best_model.pt", map_location=torch.device('cpu')))
model = model.eval()

# use the ResNet-specific canonizer
canonizer = ResNetCanonizer()

# the ZBox rule needs the lowest and highest values, which are here for
# ImageNet 0. and 1. with a different normalization for each channel

low, high = transform_norm(torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]]))

# create a composite, specifying the canonizers, if any
composite = EpsilonGammaBox(low=low, high=high, canonizers=[canonizer])

# choose a target class for the attribution (label 437 is lighthouse)
target = torch.eye(2)[[1]]

# create the attributor, specifying model and composite
with Gradient(model=model, composite=composite) as attributor:
    # compute the model output and attribution
    output, attribution = attributor(data, target)

print(f'Prediction: {output.argmax(1)[0].item()}')


# sum over the channels
relevance = attribution.sum(1)

# create an image of the visualize attribution
img = imgify(relevance, symmetric=True, cmap='coldnhot')

# show the image
#display(transform_img(img)) #diplay seems to be a jupyter notebook function
img.show()