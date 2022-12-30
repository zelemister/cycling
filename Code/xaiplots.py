# Create Plots
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from transformations import get_transformer

'''
We are interested in analyzing false negatives, i.e. bicycle lanes which have not been identified as such.
Images of interest should be saved in "Overlooked Images.
'''
# folder_path = "../Overlooked Images"
folder_path = "../Example Images"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# image transformation can be normalize_size or rotations
transformation_name = "normalize"
image_transformation = get_transformer(transformation_name)

# image lists
images_original = []
images_transformed = []
images_explained = []
# images_test = [Image.open(os.path.join("../Example Images", "dornbusch-lighthouse.jpg"))]

for file in os.listdir(folder_path):
    # ignore files that aren't images
    if file.endswith(".png") or file.endswith(".jpg"):
        # apply a xai function yet to be determined
        # test with transforms
        image = Image.open(os.path.join(folder_path, file))
        images_original.append(image)
        transformed_image = image_transformation(image)  # outputs a tensor
        images_transformed.append(transforms.ToPILImage()(transformed_image))
        transformed_image = transformed_image.permute(1, 2, 0)
        transformed_image = transformed_image.to(dtype=torch.uint8)
        plt.imshow(transformed_image)
        plt.show()

''''
testing
'''

# for file in os.listdir(folder_path):
#     # ignore files that aren't images
#     if file.endswith(".png") or file.endswith(".jpg"):
#         # apply a xai function yet to be determined
#         # test with transforms
#         image = Image.open(os.path.join(folder_path, file))
#         images_original.append(image)
#         images_explained.append(image)
#         images_transformed.append(image)

# fig, axs = plt.subplots(len(images_original), 3, figsize=(10, 15))
# for ax, (image1, image2, image3) in zip(axs, zip(images_original, images_explained, images_transformed)):
#     ax[0].imshow(image1)
#     ax[1].imshow(image2)
#     ax[2].imshow(image3)
#     ax[0].axis('off')
#     ax[1].axis('off')
#     ax[2].axis('off')
# plt.show()


"""
end testing
"""

fig, axs = plt.subplots(len(images_original), 2, figsize=(10, 15))
for ax, (image1, image2) in zip(axs, zip(images_original, images_transformed)):
    ax[0].imshow(image1)
    ax[1].imshow(image2)
    ax[0].axis('off')
    ax[1].axis('off')
plt.show()

