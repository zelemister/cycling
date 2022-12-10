#from Code.model_loaders import get_model
from torchvision.models import resnet34
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from PIL import Image
import numpy.random as random
import torch



rotate = transforms.Compose([transforms.ToTensor(),
                             #generate a random number between 0 and 1, then multiply by 360 for a random number between 0 and 360
                             transforms.RandomRotation(degrees=random.random(1)[0]*360),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip(),
                             transforms.ToPILImage()])

normalize= transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

img = [Image.open("../Example Images/x_1010001.png"),
                    Image.open("../Example Images/x_10020034.png"),
                    Image.open("../Example Images/x_10020036.png"),
                    Image.open("../Example Images/x_18040012.png")]
post=[normalize(rotate(item)) for item in img]
nonsense = [item.unsqueeze(0) for item in post]
labels = torch.tensor([0,1,1,1])

model = resnet34()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("../Models/fine_tuned_best_model.pt", map_location=torch.device('cpu')))
model.eval()
out=[]
for item in nonsense:
    with torch.no_grad():
        out.append(model(item))
print(out)

#those are the predicted probabilities for the zero class
#you can use those for roc metrics (maybe?)
zero_preds= [torch.softmax(item,1)[0][0] for item in out]

#those are the predicted probabilities for the one class
one_preds=[torch.softmax(item,1)[0][1] for item in out]

#those are the actual predicted values
pred_labels = torch.tensor([torch.max(item, 1)[1] for item in out])

tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()