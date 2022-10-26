from Code.dataset import CyclingData
from Code.model_loaders import get_model
from torchvision import transforms
from PIL import Image
import torch

torch.set_default_dtype(torch.float64)


transformer = transforms.Compose(
    [
     transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(
         [x / 255 for x in [125.3, 123.0, 113.9]],
         [x / 255 for x in [63.0, 62.1, 66.7]]
     )
     ]
)


img = Image.open("../Data/x_18040012.png")
img_transform = transformer(img)

model = get_model()
out = model(img_transform)
print(out)
