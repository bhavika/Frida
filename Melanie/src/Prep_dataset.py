import sys, os
from io import StringIO
from PIL import Image
import re
import unicodedata
from os import listdir
from os.path import isfile, join


def resize_photo(file_source, file_dest, newSize):
    try:
        im = Image.open(file_source)

        if im.mode != "RGB":
            im = im.convert("RGB")

        imsmall = im.resize(newSize, Image.ANTIALIAS)
        imsmall.save(file_dest, "JPEG")
    except Exception as exception:
        print("Exception for " + str(file_dest) + " is: " + str(exception))


def create_annotation_file(rootDirectory, query, carIndicator, imageCount):
    with open(rootDirectory + 'annotation.txt', "w") as annotation_file:
        annotation_file.write(query + "\n")
        annotation_file.write(str(carIndicator) + "\n")
        annotation_file.write(str(imageCount) + "\n")

def main():
    imgDir = 'C:\\Users\\Melanie\\Desktop\\impress_data\\'
    imgDirResize = 'C:\\Users\\Melanie\\Desktop\\impress_data_small\\'
    imageList = [f for f in os.listdir(imgDir) if isfile(join(imgDir, f))]    
    #print (imageList)

    for originalImage in imageList:
        resize_photo(join(imgDir, originalImage),join(imgDirResize,originalImage), (210,140))

if __name__ == "__main__":
    main()