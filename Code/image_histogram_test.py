# Testfile to create a coloured histogram from an image file
# Aim: test how much info colour red alone provides about presence of cycling infrastructure
# To-do: read image, create histogram, define appropriate colour range, count pixels within range

import os
import numpy
import cv2
from PIL import Image
from matplotlib import pyplot as plt

# get image x_18040012.png from example image folder
img = Image.open("../Example Images/x_18040012.png")
print(img.size)

#need image type converted to matrix. channel 0=blue, 1=green, 2=red
img_cv = cv2.imread("../Example Images/x_18040012.png")
hist_example = cv2.calcHist([img_cv],[2],None,[256],[0,256])
#plot histogram for example image
plt.plot(hist_example, color='r')
plt.show()
#relevant pixel range 175-205?
