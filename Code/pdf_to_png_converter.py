import fitz
import io
from PIL import Image
import os


#this code transforms the pdf files to the native file type of the images (this seems to be png files)
#code was used and modified from the internet:
#https://pythonscholar.com/blog/extract-images-from-pdf-in-python/

folder = "../Images_512"
if not os.path.isdir(folder):
    os.mkdir(folder)
os.chdir(folder)

pdfs_folder = "../512_pdfs"
total_num = os.listdir(pdfs_folder).__len__()

for index, name in enumerate(os.listdir(pdfs_folder), start=1):
    #this image is corrupted in the higher resolution data, trying without
    #if not name == "X__9010063.pdf":
    pdf_file = fitz.open(os.path.join(pdfs_folder,name))
    name_raw = name[:-4] # For the small pictures. This line
    #name_raw = "x_" + name[3:-4]
    page = pdf_file[0]

    # get the XREF of the image
    xref = page.get_images()[0][0]
    # extract the image bytes
    base_image = pdf_file.extract_image(xref)
    image_bytes = base_image["image"]
    # get the image extension
    image_ext = base_image["ext"]
    # load it to PIL
    image = Image.open(io.BytesIO(image_bytes))
    # save it to local disk
    image.save(folder + f"/{name_raw}.{image_ext}")
    print(f"Added file {index}/{total_num}; {name_raw}.{image_ext}")
