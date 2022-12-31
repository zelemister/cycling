import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms import ToTensor, Normalize
from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox
from zennit.image import imgify
from zennit.torchvision import ResNetCanonizer
from model_loaders import get_model


def explainme(pil_image: Image, resolution=256, model_name="resnet",
              pretrained_val=False, finetune_path="../Models" + "/fine_tuned_best_model.pt"):
    # define the base image transform
    transform_img = Compose([
            Resize(256),
            CenterCrop(224),
        ])
    # define the normalization transform
    transform_norm = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # define the full tensor transform
    transform = Compose([
            transform_img,  # resize and crop
            ToTensor(),
            transform_norm,  # normalize
        ])
    if not isinstance(pil_image, Image.Image):
            raise ValueError('Input not an Image')
    else:
        # transform the PIL image and insert a batch-dimension
        data = transform(pil_image)[None]
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
        # sum over the channels
        relevance = attribution.sum(1)
        # create an image of the visualize attribution
        img = imgify(relevance, symmetric=True, cmap='coldnhot')
        # return the image
        return img