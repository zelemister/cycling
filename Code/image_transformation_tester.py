from PIL import Image
from torchvision import transforms
def custom_transformer(img: Image.Image):
    transformer = transforms.Compose(
                        [transforms.PILToTensor(),
                        transforms.CenterCrop(256),
                        transforms.RandomRotation(180),
                        # RandomRotation(180) should rotate the image by a random 360 degrees. There are some filling
                        # options that are important, since RIM would be easily detectable by black corners

                        #transforms.RandomHorizontalFlip,
                        #transforms.RandomAutocontrast,
                        # transforms.functional.adjust_contrast(contrast_factor= 2),
                        # How much to adjust the contrast. Can be any non negative number. 0 gives a solid gray image,
                        # 1 gives the original image while 2 increases the contrast by a factor of 2.
                        #transforms.functional.adjust_brightness,
                        #transforms.functional.adjust_hue,
                        #transforms.functional.adjust_sharpness,
                        #transforms.functional.adjust_saturation
                        transforms.ToPILImage()]
                    )
    return(transformer)


names = ["x_1010001.png", "x_18040012.png"]
#value = int(input("What's the value of the transformation that should take place?"))

#transformer = custom_transformer()
img = Image.open("../Data/x_18040012.png")
img = transforms.functional.adjust_contrast(img, 10)
print(img.__class__)
#img = transformer(img)
img.show()

#print(value)