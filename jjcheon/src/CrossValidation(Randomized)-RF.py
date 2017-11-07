from PIL import ImageStat, Image, ImageFile
from sklearn.model_selection import train_test_split
import pandas as pd

from util_image import find_brightness, find_blur_value, find_edge, find_edge_enhance, find_edge_enhance_more, \
    find_contour, find_emboss, find_detail, find_smooth, find_smooth_more

#print (__doc__)

import numpy as np
from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

#get some data
#load the impressionist csv and images file
base_address = '/home/jay/PycharmProjects/688-project/wikiart/'

style = 'Impressionism-3a-3p/'
train_file = 'impressionists-3a-3p.csv'
filepath = base_address
unique_artists = set()
unique_link = list()

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



print()
print("Calculate the feature values of each paintings ")
print()
print("Features are brightness,blur,edge,contour,emboss,smooth, and so on")
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


    #feature values : blur, brightness, edge, edge_enhance, edge_enhance_more, contour,emboss, detail, smooth, smooth_more
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


print()

print("Start estimating optimal parameters of RandomForest")
print()

#build a classifier
clf = RandomForestClassifier(n_estimators=20)

#Utility function to report best scores
def report(results, n_top=3) :
    for i in range(1, n_top +1):
        candidates = np.flatnonzero(results['rank_test_score'] ==1)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print()

#specify parameters and distributions to sample form
param_dist = {"max_depth": [3,None],
              "max_features": sp_randint(1,11),
              "min_samples_split": sp_randint(2,11),
              "min_samples_leaf": sp_randint(1,11),
              "bootstrap":[True, False],
              "criterion":["gini","entropy"]}

# run randomized search
n_iter_search =20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(feature, label_data)
print("RandomizedSearchCV took %.2f seconds for %d candidates" 
      " parameter settings." %((time()- start), n_iter_search))
report(random_search.cv_results_)




