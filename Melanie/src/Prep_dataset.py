import os
from PIL import Image
from os.path import isfile, join
import pandas as pd
import shutil
import constants

csv = constants.csv
wiki_art = constants.wiki_art_path
imgDir = constants.original_img_path
imgDirResize = constants.small_img_path

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
    artist_data = pd.read_csv(csv)
    location_of_data = artist_data['location']
    
    i=0;
    for i in location_of_data:
        shutil.copy(join(wiki_art,i), imgDir)
      
    imageList = [f for f in os.listdir(imgDir) if isfile(join(imgDir, f))]    

    for originalImage in imageList:
        resize_photo(join(imgDir, originalImage),join(imgDirResize,originalImage), (210, 140))

if __name__ == "__main__":
    main()