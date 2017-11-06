# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:48:00 2017

@author: Melanie
"""

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

def main():
    imgDir = 'C:\\Users\\Melanie\\Desktop\\DEMO_impress_data\\'
    imgDirResize = 'C:\\Users\\Melanie\\Desktop\\DEMO_impress_data_small\\'
    imageList = [f for f in os.listdir(imgDir) if isfile(join(imgDir, f))]    
    #print (imageList)
    c=1
    b=0
        
    with open('C:\\Users\\Melanie\\Desktop\\art_anno\\annotation.txt', "w") as annotation_file:
        for originalImage in imageList:
            
            if c > 40 and c < 81:
                b = 1
            if c > 80 and c < 121:
                b = 2

            o = originalImage.split('_')
            #print(o[0])
            resize_photo(join(imgDir, originalImage),join(imgDirResize,originalImage), (210,140))

            annotation_file.write(o[0] + "\n")
            annotation_file.write(str(b) + "\n")
            #print(b)
            annotation_file.write(str(40) + "\n")
            c=c+1
if __name__ == "__main__":
    main()