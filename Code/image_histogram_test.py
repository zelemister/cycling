# Testfile to create a coloured histogram from an image file
# Aim: test how much info colour red alone provides about presence of cycling infrastructure
# Problem: What is red in a picture? only determined by a combination of pixels.

import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


def distance_to_red(color_bgr):
    #curently unused
    #exploring different colour representations according to human eye
    reference = sRGBColor(1.0, 0.0, 0.0)
    reference_lab = convert_color(reference, LabColor)

    color = sRGBColor(color_bgr[2], color_bgr[1], color_bgr[0])
    color_lab = convert_color(color, LabColor)

    # print(type(delta_e_cie2000(reference_lab, color_lab)))
    return delta_e_cie2000(reference_lab, color_lab)


def is_red_enough(color_bgr):
    # print("----", color_bgr, (color_bgr[0] + color_bgr[1] + color_bgr[2]), color_bgr[2] / (color_bgr[0] + color_bgr[1] + color_bgr[2]))
    # return ( color_bgr[2] / (color_bgr[0] + color_bgr[1] + color_bgr[2] + 1) > 0.5 )
    return color_bgr[2] >= max(color_bgr[0], color_bgr[1]) + 30

def is_green_enough(color_bgr):
    return color_bgr[1] >= max(color_bgr[0], color_bgr[2]) + 5
def is_pretty_gray(color_bgr):
    return (max(color_bgr[0], color_bgr[1], color_bgr[2]) - min(color_bgr[0], color_bgr[1], color_bgr[2]) <= 20) or (color_bgr[0] >= max(color_bgr[1], color_bgr[2]) + 30)


# get image x_18040012.png from example image folder
# img = Image.open("../Example Images/x_18040012.png", cv2.IMREAD_COLOR)
# print(img.size)

# need image type converted to matrix. channel 0=blue, 1=green, 2=red
#img_cv = cv2.imread("../Example Images/x_18040012.png")
# img_cv = cv2.imread("../Example Images/x_10020034.png")
#img_cv = cv2.imread("../Example Images/x_1010001.png")
img_cv = cv2.imread('../Images/x_23040044.png')

def classify_colors(img_cv):
    rows, cols, _ = img_cv.shape
    for r in range(rows):
        for c in range(cols):
            color_bgr = img_cv[r, c].astype(float)
            if is_red_enough(color_bgr):
                img_cv[r, c] = [0, 0, 255]
            if is_pretty_gray(color_bgr):
                img_cv[r, c] = [255, 0, 0]
            if is_green_enough(color_bgr):
                img_cv[r, c] = [0, 255, 0]


classify_colors(img_cv)
cv2.imshow("my image", img_cv)
cv2.waitKey(0)

# hist_example = cv2.calcHist([img_cv], [2], None, [256], [0, 256])

# plot histogram for example image
# plt.plot(hist_example, color='r')
# plt.show()
# relevant pixel range 180-205?