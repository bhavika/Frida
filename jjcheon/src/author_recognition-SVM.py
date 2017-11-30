#import pandas as pd
import pickle
import numpy as np
from PIL import Image, ImageFilter, ImageFile, ImageStat
import pandas as pd
import time

from sklearn import svm
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from pathlib import Path

from sklearn.svm import LinearSVC

from util_file import make_author_mapping_file
from util_image import find_blur_value, find_edge_enhance, find_smooth_more, find_brightness, find_edge, find_contour, \
    find_emboss, find_edge_enhance_more, find_detail, find_smooth
from Constant import style, base_address, train_file

filepath = base_address
unique_artists = set()
unique_link = list()

print("")
print("Load the dataset from csv file ", train_file)
print("")

#file categories are ids,location,artist,class
paintings_by_artist = pd.read_csv(filepath+train_file, names=['ids','location','artist', 'class'], header=0)
paintings_by_artist['absolute_location'] = base_address +style+ paintings_by_artist['location']
#make the mapping between class and the author
make_author_mapping_file(filepath,train_file)

ImageFile.LOAD_TRUNCATED_IMAGES = True

rows_count = paintings_by_artist.shape[0]
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
image_data= []


print("Calculate the feature values of each paintings ")
print("Features are brightness,blur,edge,contour,emboss,smooth, and so on")
print()

#read records one by one
#for i in range(paintings_by_artist.shape[0]):
for i in range(len(list(unique_link))):
    link = list(unique_link)[i]
    global img
    my_file = Path(link)
    if my_file.is_file():
        img = Image.open(link,'r')
        image = img.resize((30,30))
        #image = img.resize((224,224))
        image = np.array(image).astype(np.float32)
        image_data.append(img)
    else :
        continue
    #transform the image value to statistical value
    stat = ImageStat.Stat(img)
    #generate the class data
    label_data[i] = paintings_by_artist.iloc[i]['class']


    #before feature selection
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

print ("")
print ("Split arrays or matrices into random train and test subsets")
print ("X_train, X_test, y_train, y_test, image_train, image_test  ")
X_train, X_test, y_train, y_test , image_train, image_test = train_test_split(feature, label_data,image_data, test_size=0.40)


print ("")
print ("Calculate feature importance")

#clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(feature,label_data)
clf = LinearSVC(C=0.01, penalty="l1", dual=False)
#clf  = svm.SVC(C=0.01,cache_size=2000).fit(feature,label_data)
#rfecv = RFECV(estimator=clf,  cv=StratifiedKFold(2),n_jobs=3,
rfecv=RFECV(estimator=clf, cv=StratifiedKFold(2),
                          scoring=None)
rfecv.fit(X_train, y_train)

#features importance
#for i in range(len(clf.feature_importances_)) :
#for i in range(len(rfecv.ranking_)):
#        print (i+1, rfecv.ranking_[i])

mask = rfecv.get_support(indices=True)
print("Selected feature indexies", mask)
print ("")

print("Estimate optimal parameters of SVM")
#parameters = {'kernel':'linear', 'C':[1]}
parameters = [{'kernel' : ['linear'], 'C' : [1,10]}]


#search_time = time.time()
scv =svm.SVC()
#scv = LinearSVC()

clf2 = GridSearchCV(scv, parameters, cv=2, scoring=None, n_jobs= 10, verbose=True)
clf2.fit(feature, label_data)
#print()
#print("Best parameters : ", clf2.best_params_)
#print("searching time is ", search_time - time.time())

start_time = time.time()
#svm_clf = svm.SVC(kernel=clf2.best_params_['kernel'], C=clf2.best_params_['C'], cache_size=2000)
svm_clf = svm.SVC(kernel='linear', C=1, cache_size=400)

print()
print("Train with X_train, y_train data by using StratifiedKFold Cross Validation.")
print()

from sklearn.model_selection import StratifiedKFold
#skf = StratifiedKFold(n_splits=2)
#for train, test in skf.split(X_train, y_train):
svm_clf.fit(X_train, y_train)

print("Test with X_test, y_test data by using StratifiedKFold Cross Validation.")
print("Save the trained model")
print()

filename = 'finalized_model_svm.sav'
pickle.dump(svm_clf, open(filename, 'wb'))

print("Predict class for X_test.")
print()
x_predict = svm_clf.predict(X_test)


print()
print("Show the estimation result .")
print()

from sklearn.metrics import confusion_matrix
print ("Confusion Matrix:\n", confusion_matrix(y_test, x_predict))
print ("Accuracy :", accuracy_score(y_test, x_predict)*100)
data = precision_recall_fscore_support(y_test, x_predict, average='micro')
print ("precision :", data[0])
print ("recall:", data[1])
print("--- %s seconds ---" % (time.time() - start_time))