from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

#iris = load_iris()

import pandas as pd
from PIL import Image, ImageFilter, ImageFile, ImageStat
import os
import numpy as np
import pandas as pd

from sklearn import svm
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.genmod.tests.results.results_glm_poisson_weights import predicted

from util_image import find_blur_value, find_edge_enhance, find_smooth_more, find_brightness, find_edge, find_contour, \
    find_emboss, find_edge_enhance_more, find_detail, find_smooth

#load the impressionist csv and images file
base_address = '/home/jay/PycharmProjects/688-project/wikiart/'

style = 'Impressionism-5a-5p/'
train_file = 'impressionists-5a-5p.csv'
filepath = base_address
unique_artists = set()
unique_link = list()


print("")
print("Load the dataset from csv file(impressionists-3a-3p")
print("")

#file categories are ids,location,artist,class
paintings_by_artist = pd.read_csv(filepath+train_file, names=['ids','location','artist', 'class'], header=0)
paintings_by_artist['absolute_location'] = base_address +style+ paintings_by_artist['location']
#paintings_by_artist = paintings_by_artist.sort_values('class')
ImageFile.LOAD_TRUNCATED_IMAGES = True

rows_count = paintings_by_artist.shape[0]
#cols_count = 7
cols_count = 10
actual_image_count = 0

#investigate the actual number of images
for i in range(paintings_by_artist.shape[0]):
    link = paintings_by_artist.iloc[i]['absolute_location']
    #paintings_by_artist = paintings_by_artist.sort_values('class')
    my_file = Path(link)
    if my_file.is_file():
        actual_image_count += 1
        unique_link.append(link)


feature= [[0 for j in range(cols_count)] for i in range(actual_image_count)]
label_data = [0 for i in range(actual_image_count)]


print("Calculate the feature values of each paintings ")
print()
print("Features are brightness,blur,edge,contour,emboss,smooth, and so on")
print()
#read records one by one
#print(paintings_by_artist.shape[0])
#feature =[paintings_by_artist.shape[0], 10]
#for i in range(paintings_by_artist.shape[0]):
for i in range(len(list(unique_link))):
    #link = paintings_by_artist.iloc[i]['absolute_location']
    link = list(unique_link)[i]
    #paintings_by_artist = paintings_by_artist.sort_values('class')
    global img
    #print label_data
    my_file = Path(link)
    if my_file.is_file():
        img = Image.open(link,'r')
    else :
        continue

    #transform the image value to statistical value
    stat = ImageStat.Stat(img)

    #generate the class data
    label_data[i] = paintings_by_artist.iloc[i]['class']

    #class data
    #label_data[i] = label


    #feature values : brightness,blur, edge, edge_enhance, edge_enhance_more, contour,emboss, detail, smooth, smooth_more
#    feature[i][0] = find_brightness(img)
#    feature[i][1] = find_blur_value(link)
#    feature[i][2] = find_edge(img)
#    feature[i][3] = find_edge_enhance(img)
#    feature[i][4] = find_edge_enhance_more(img)
#    feature[i][5] = find_contour(img)
#    feature[i][6] = find_emboss(img)
#    feature[i][7] = find_detail(img)
#    feature[i][8] = find_smooth(img)
#    feature[i][9] = find_smooth_more(img)
    feature[i][0] = find_brightness(img)
    feature[i][1] = find_blur_value(link)
    feature[i][2] = find_edge(link)
    feature[i][3] = find_edge_enhance(img)
    feature[i][4] = find_edge_enhance_more(img)
    feature[i][5] = find_contour(img)
    feature[i][6] = find_emboss(img)
    feature[i][7] = find_detail(img)
    feature[i][8] = find_smooth(img)
    feature[i][9] = find_smooth_more(img)
    unique_artists.add(label_data[i])



#X_train, X_test, y_train, y_test = train_test_split(feature, label_data, test_size=0.2)


#X, y = iris.data, iris.target
#print (X)
#print (X.shape)

print ("Original all features value - show first row of feature data")
print (feature[0])
print ("")
print("brightness(1),blur(2),edge(3),edge_enhance(4), edge_enhance_more(4), contour(5),emboss(6),detail(7),smooth(9), smooth_more(10)")
print()
clf = ExtraTreesClassifier()
#clf = clf.fit(X, y)
clf = clf.fit(feature, label_data)
print ("Report the importance value per features")

for i in range(len(clf.feature_importances_)) :
    print (i+1, clf.feature_importances_[i])
#print ("edge, detail, emboss, smooth_more")
model = SelectFromModel(clf, prefit=True)
X_new  = model.transform(feature)
print()
print ("Picked features")
print (X_new[0])