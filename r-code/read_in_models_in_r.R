library(torch)
library(torchvision)
library(torchdatasets)
library(luz)
library(magick)

#to read in a model, adjust the paths and the model function, based on their 
#inputs_and_results.csv file
#or, if it's in deliverables, rim_model was resnet18, bikelane model resnet34

path = "../Deliverables/"
# model_path = "bikelane_model.pt"
# m = model_resnet34()
model_path = "rim_model.pt"
m = model_resnet18()
#initialize model
m$fc = nn_linear(m$fc$in_features, 2)
state_dict = load_state_dict(paste0(path, model_path))
m$load_state_dict(state_dict)

#then apply transformation to each image in img. This image has to be initialized
#read about this in https://blogs.rstudio.com/ai/posts/2020-09-29-introducing-torch-for-r/

#somhow read in images through the magick package or an array (H x W x C) (height x width x channels) with range ⁠[0, 255]
img
img = transform_to_tensor(img)
img = transform_normalize(img, mean = c(0.485, 0.456, 0.406), 
                                     std = (0.229, 0.224, 0.225))
#then do
output = m(img)