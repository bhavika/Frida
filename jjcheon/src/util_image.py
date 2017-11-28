import sys, os
import pandas as pd
from PIL import Image, ImageStat, ImageFilter
import cv2
import numpy as np

# Via http://stackoverflow.com/a/3498247
from scipy.misc import toimage, fromimage

from Constant import *


def find_brightness( im_file ):
    #img = Image.open(im_file)
    im = im_file.convert('L')
    #im.show()
    stat = ImageStat.Stat(im)
    #print "Read RMS brightness of image: "
    #print stat.rms[0]
    return stat.rms[0]

#if __name__ == "__main__":
#    br  = brightness("images.png")
#    print ("brightness",br)


def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float" )
    return data


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension

    #err = ((np.sum(np.array(imageA.resize((244,244))).astype(np.float) - np.array(imageB.resize((244,244))).astype(np.float)) ** 2))
    err = ((np.sum(np.array(imageA.resize((244,244))).astype(np.float) - np.array(imageB.resize((244,244))).astype(np.float)) ** 2))
    #err = ((np.sum(np.array(imageA).astype(np.float) - np.array(imageB).astype(np.float)) ** 2))
    #err /= float(imageA.shape[0] * imageA.shape[1])
    err /= float(np.array(imageA.resize((244, 244))).astype(np.float).shape[0] * np.array(imageB.resize((244, 244))).astype(np.float).shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def PIL2array(img):
    return np.array(img.get,
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def find_similiar_image(author_class,input_image):
    paintings_by_artist = pd.read_csv(base_address + 'author_image_mapping.csv', names=['class', 'absolute_path'], header=0)
    error =[]
    min_index =0
    mse_value=100000000000000.0
    temp = 0.0

    for i in range(paintings_by_artist.shape[0]):
        img = Image.open(paintings_by_artist.iloc[i]['absolute_path'])
        #img.show()
        if paintings_by_artist.iloc[i]['class'] == author_class :
            temp= mse(input_image, img)
            if(temp <mse_value):
                min_index =i
                mse_value= temp
    similar_image =Image.open(paintings_by_artist.iloc[min_index]['absolute_path'],'r')
    #similar_image.show()
    return similar_image

def find_author_image(author_class):
    paintings_by_artist = pd.read_csv(base_image_file_location + 'author_image_mapping.csv', names=['class', 'absolute_path'], header=0)
    error =[]
    min_index =0
    mse_value=100000000000000.0
    image_data  = []
    temp = 0.0

    for i in range(paintings_by_artist.shape[0]):
        img = Image.open(paintings_by_artist.iloc[i]['absolute_path'])
        #img.show()
        if paintings_by_artist.iloc[i]['class'] == author_class :
            image_data.insert(i,img)
            if(i==5):
                break
            #temp= mse(input_image, img)
            #if(temp <mse_value):
            #    min_index =i
            #    mse_value= temp
    #similar_image =Image.open(paintings_by_artist.iloc[min_index]['absolute_path'],'r')
    #similar_image.show()
    return image_data

def find_edge(im_file) :
    #image = Image.open(im_file)
    #image = im_file.filter(ImageFilter.FIND_EDGES)
    #image.show()

    #stat = ImageStat.Stat(image)
    #print "Read RMS find edge of image: "
    #print stat.rms[0]

    img = cv2.imread(im_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    return dst.size
    #return stat.rms[0]

def find_edge2(image) :
    #image = Image.open(im_file)
    #image = im_file.filter(ImageFilter.FIND_EDGES)
    #image.show()

    #stat = ImageStat.Stat(image)
    #print "Read RMS find edge of image: "
    #print stat.rms[0]

    #img = cv2.imread(im_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    return dst.size
    #return stat.rms[0]


def find_contour(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.CONTOUR)
    #image = image.filter(ImageFilter.CONTOUR)
    #image.show()
    #cv2.imwrite('arkhip-kuindzhi_red-sunset-1-contour.jpg', np.array(image).astype(np.float32))

    stat = ImageStat.Stat(image)
    #print "Read RMS find contour of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_emboss(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.EMBOSS)
    #image.show()
    #cv2.imwrite('arkhip-kuindzhi_red-sunset-1-emboss.jpg', np.array(image).astype(np.float32))
    stat = ImageStat.Stat(image)
    #print "Read RMS find EMBOSS of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_detail(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.DETAIL)
    #image.show()
    #cv2.imwrite('arkhip-kuindzhi_red-sunset-1-detail.jpg', np.array(image).astype(np.float32))
    stat = ImageStat.Stat(image)
    #print "Read RMS find detail(thumbnail) of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_edge_enhance(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.EDGE_ENHANCE)
    #image.show()        image =
    #cv2.imwrite('arkhip-kuindzhi_red-sunset-1-detail-enhance.jpg', np.array(image).astype(np.float32))
    stat = ImageStat.Stat(image)
    #print "Read RMS find EDGE_ENHANCE of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_edge_enhance_more(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.EDGE_ENHANCE_MORE)
    #image.show()
    #cv2.imwrite('arkhip-kuindzhi_red-sunset-1-detail-enhance-more.jpg', np.array(image).astype(np.float32))
    stat = ImageStat.Stat(image)
    #print "Read RMS find EDGE_ENHANCE_MORE of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_smooth(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.SMOOTH)
    #image.show()
    #cv2.imwrite('arkhip-kuindzhi_red-sunset-1-detail-smooth.jpg', np.array(image).astype(np.float32))
    stat = ImageStat.Stat(image)
    #print "Read RMS find SMOOTH of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_smooth_more(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.SMOOTH_MORE)
    #image.show()
    #cv2.imwrite('arkhip-kuindzhi_red-sunset-1-detail-smooth-more.jpg', np.array(image).astype(np.float32))
    stat = ImageStat.Stat(image)
    #print "Read RMS find SMOOTH more of image: "
    #print stat.rms[0]
    return stat.rms[0]


# compute the Laplacian of the image and then return the focus
# measure, which is simply the variance of the Laplacian
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# load the image, convert it to grayscale, and compute the
# focus measure of the image using the Variance of Laplacian
# method
def find_blur_value(im_file) :
    image = cv2.imread(im_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = "Blurry"
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    #cv2.imshow(image)
    key = cv2.waitKey(0)
    return fm

def find_blur_value2(image) :
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = "Blurry"
    cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    #cv2.imshow(image)
    key = cv2.waitKey(0)
    return fm


def imfilter(arr, ftype):
    """
    Simple filtering of an image.

    Parameters
    ----------
    arr : ndarray
        The array of Image in which the filter is to be applied.
    ftype : str
        The filter that has to be applied. Legal values are:
        'blur', 'contour', 'detail', 'edge_enhance', 'edge_enhance_more',
        'emboss', 'find_edges', 'smooth', 'smooth_more', 'sharpen'.

    Returns
    -------
    imfilter : ndarray
        The array with filter applied.

    Raises
    ------
    ValueError
        *Unknown filter type.*  If the filter you are trying
        to apply is unsupported.

    """
    _tdict = {'blur': ImageFilter.BLUR,
              'contour': ImageFilter.CONTOUR,
              'detail': ImageFilter.DETAIL,
              'edge_enhance': ImageFilter.EDGE_ENHANCE,
              'edge_enhance_more': ImageFilter.EDGE_ENHANCE_MORE,
              'emboss': ImageFilter.EMBOSS,
              'find_edges': ImageFilter.FIND_EDGES,
              'smooth': ImageFilter.SMOOTH,
              'smooth_more': ImageFilter.SMOOTH_MORE,
              'sharpen': ImageFilter.SHARPEN
              }

    im = toimage(arr)
    if ftype not in _tdict:
        raise ValueError("Unknown filter type.")
    return fromimage(im.filter(_tdict[ftype]))