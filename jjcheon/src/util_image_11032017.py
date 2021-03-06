import sys, os
from PIL import Image, ImageStat, ImageFilter
import argparse
import cv2
from pathlib import Path

# Via http://stackoverflow.com/a/3498247
from scipy.misc import toimage, fromimage


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


def find_edge(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.FIND_EDGES)
    #image.show()
    stat = ImageStat.Stat(image)
    #print "Read RMS find edge of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_contour(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.CONTOUR)
    #image.show()
    stat = ImageStat.Stat(image)
    #print "Read RMS find contour of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_emboss(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.EMBOSS)
    #image.show()
    stat = ImageStat.Stat(image)
    #print "Read RMS find EMBOSS of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_detail(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.DETAIL)
    #image.show()
    stat = ImageStat.Stat(image)
    #print "Read RMS find detail(thumbnail) of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_edge_enhance(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.EDGE_ENHANCE)
    #image.show()
    stat = ImageStat.Stat(image)
    #print "Read RMS find EDGE_ENHANCE of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_edge_enhance_more(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.EDGE_ENHANCE_MORE)
    #image.show()
    stat = ImageStat.Stat(image)
    #print "Read RMS find EDGE_ENHANCE_MORE of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_smooth(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.SMOOTH)
    #image.show()
    stat = ImageStat.Stat(image)
    #print "Read RMS find SMOOTH of image: "
    #print stat.rms[0]
    return stat.rms[0]

def find_smooth_more(im_file) :
    #image = Image.open(im_file)
    image = im_file.filter(ImageFilter.SMOOTH_MORE)
    #image.show()
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