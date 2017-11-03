import pandas as pd
from PIL import Image, ImageFilter, ImageFile, ImageStat
import os
import numpy as np
#from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

from sklearn import svm
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
from util_image import find_blur_value, find_edge_enhance, find_smooth_more, find_brightness, find_edge, find_contour, \
    find_emboss, find_edge_enhance_more, find_detail, find_smooth

#load the impressionist csv and images file
base_address = '/Users/User/PycharmProjects/PR-termProject/wikiart/'
style = 'Impressionism/'
train_file = 'impressionists.csv'
filepath = base_address

#file categories are ids,location,artist,class
paintings_by_artist = pd.read_csv(filepath+train_file, names=['ids','location','artist', 'class'], header=0)
paintings_by_artist['absolute_location'] = base_address +style+ paintings_by_artist['location']
paintings_by_artist = paintings_by_artist.sort_values('class')
ImageFile.LOAD_TRUNCATED_IMAGES = True

rows_count = paintings_by_artist.shape[0]
cols_count = 10

feature= [[0 for j in range(cols_count)] for i in range(rows_count)]
label_data = [0 for i in range(rows_count)]


#read records one by one
#print(paintings_by_artist.shape[0])
#feature =[paintings_by_artist.shape[0], 10]
for i in range(paintings_by_artist.shape[0]):
    link = paintings_by_artist.iloc[i]['absolute_location']
    paintings_by_artist = paintings_by_artist.sort_values('class')
    #print label_data
    my_file = Path(link)
    if my_file.is_file():
        img = Image.open(link,'r')
    else :
        continue

    #transform the image value to statistical value
    stat = ImageStat.Stat(img)

    #assign the class data
    label_data[i] = paintings_by_artist.iloc[i]['class']

    #class data
    #label_data[i] = label


    #feature values : blur, brightness, edge, edge_enhance, edge_enhance_more, contour,emboss, detail, smooth, smooth_more
    feature[i][0] = find_brightness(img)
    feature[i][1] = find_blur_value(link)
    feature[i][2] = find_edge(img)
    feature[i][3] = edge_enhance = find_edge_enhance(img)
    feature[i][4] = find_edge_enhance_more(img)
    feature[i][5] = find_contour(img)
    feature[i][6] = find_emboss(img)
    feature[i][7] = find_detail(img)
    feature[i][8] = find_smooth(img)
    feature[i][9] = find_smooth_more(img)
    #print ("blur", blur, "edge", edge,"smooth_more", smooth_more, 'smooth',smooth)
    #s = pd.Series(blur, brightness, edge, edge_enhance, edge_enhance_more, contour,emboss, detail, smooth, smooth_more, label)
    #table = plt.table(blur)

#print(feature)
#print label_data
#print np.shape(label_data)


#df = pd.read_csv('breast-cancer-wisconsin.data.txt')
#df.replace('?',-99999, inplace=True)
#df.drop(['id'], 1, inplace=True)
#X = np.array(df.drop(['class'], 1))
#y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(feature, label_data, test_size=0.2)
#print("X_train", X_train)
#print("X_test", X_test)
#print("y_train", y_train)
#print("y_test", y_test)
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_train, y_train)
x_predict = clf.predict(X_test)
#print "mean_squere_error", mean_squared_error(y_test, x_predict )
print "r2_score",r2_score(y_test, x_predict )

