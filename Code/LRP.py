'''
This file is used to perform Layerwise-Relevance Propagation (LRP)
It uses the zennit implementation in pytorch: https://github.com/chr5tphr/zennit
The code adapts the tutorial: https://zennit.readthedocs.io/en/latest/tutorial/image-classification-vgg-resnet.html
'''

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms import ToTensor, Normalize
from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox
from zennit.image import imgify
from zennit.torchvision import ResNetCanonizer
from model_loaders import get_model

'''
To use the file, the following is necessary:
1) an image  located in a folder Images_256 ('../Images_256/image_name.png')
2) a trained neural network that can be loaded (change in line 61)
Our project used a Resnet34 for the bike lane model and a Resnet18 for the RIM model.
'''
'''
List of images used in our report and the respective model:
image = Image.open('../Images_256/x_23040044.png') # s-shape (resnet34)
image = Image.open('../Images_256/x_26010150.png') # perfect score prediction (resnet34)
image = Image.open('../Images_256/x_18030037.png') # dual rim bike (resnet34)
image = Image.open('../Images_256/x_27020025.png') # white (resnet34)
image = Image.open('../Images_256/x_23010112.png') # misslabelled (resnet34)
image = Image.open('../Images_256/x_18030037.png') # dual rim bike (resnet18)
image = Image.open('../Images_256/x_25020048.png') # not a rim (resnet18)
'''

#specify an image to be analyzed
image = Image.open('../Images_256/x_18030037.png')

'''
The following steps apply typical image transformations and load the image with an additional batch dimension as data.
'''
# define the base image transformation size as 224 pixel
transform_img = Compose([
    Resize(256),
    CenterCrop(224),
])
# define the normalization according ImageNet data
transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# define a composite transformation
transform = Compose([
    transform_img,  # resize and crop
    ToTensor(),
    transform_norm,  # normalize
])
# apply the transformations the PIL image and insert a batch-dimension
data = transform(image)[None]

'''
load the model and set it to evaluation mode
models are rim_model.pt (resnet18) or bike_model.pt (resnet34) located in the folder "Deliverables"
'''
model = get_model("resnet18") #18 for rim and 34 for bike model
model.load_state_dict(torch.load("../Deliverables" + "/rim_model.pt", map_location=torch.device('cpu')))
model = model.eval()

'''
the following steps apply LRP. It requires:
1) a canonizer
2) a composite building upon the canonizer
3) an attributor using the model and the composite
'''
# use the ResNet-specific canonizer
canonizer = ResNetCanonizer()

# the ZBox rule needs the lowest and highest values, which are here for
# ImageNet 0. and 1. with a different normalization for each channel
low, high = transform_norm(torch.tensor([[[[[0.]]] * 3], [[[[1.]]] * 3]]))

# create a composite after specifying the canonizer
composite = EpsilonGammaBox(low=low, high=high, canonizers=[canonizer])
# choose a target class for the attribution (binary classification)
target = torch.eye(2)[[1]]

# create the attributor, specifying model and composite
with Gradient(model=model, composite=composite) as attributor:
    # compute the model output and attribution
    output, attribution = attributor(data, target)
#optionally print a prediction
#print(f'Prediction: {output.argmax(1)[0].item()}')

'''
Finally, calculate the relevance by attribution it to the target class and create the image output.
Img.save() allows to save an image.
Display works only in jupyter notebooks. Use img.show().
'''
# sum over the channels
relevance = attribution.sum(1)
# create an image of the visualize attribution
img = imgify(relevance, symmetric=True, cmap='coldnhot')

# show the image
img.show()


