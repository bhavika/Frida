#import pandas as pd
from PIL import Image, ImageFilter, ImageFile, ImageStat
#import os
#import numpy as np
import pandas as pd
import time

from sklearn import svm
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
#import matplotlib.pyplot as plt
from pathlib import Path
#from statsmodels.genmod.tests.results.results_glm_poisson_weights import predicted

from util_image import find_blur_value, find_edge_enhance, find_smooth_more, find_brightness, find_edge, find_contour, \
    find_emboss, find_edge_enhance_more, find_detail, find_smooth

#load the impressionist csv and images file
#base_address = '/Users/User/PycharmProjects/PR-termProject/wikiart/'
base_address = '/home/jay/PycharmProjects/688-project/wikiart/'

style = 'Impressionism-3a-3p/'
train_file = 'impressionists-3a-3p.csv'
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

##before feature selection
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

## 5a-5p
# ##After LinearSVC feature selection : brightness, blur, contour, emboss, smooth_more
    #feature[i][0] = find_brightness(img)
    #feature[i][1] = find_blur_value(link)
    #feature[i][2] = find_contour(img)
    #feature[i][3] = find_smooth(img)
    #feature[i][4] = find_smooth_more(img)


## 5a-5p
##After ExtraForestClassifier feature selection : brightness, blur, edge, contour,
    #feature[i][0] = find_brightness(img)
    #feature[i][1] = find_blur_value(link)
    #feature[i][2] = find_edge_enhance_more(img)
    #feature[i][3] = find_contour(img)
    unique_artists.add(label_data[i])


##After ExtraTreesClassifier feature selection : brightness, blur, contour, emboss, smooth_more


# Create a dataframe with the ten feature variables
#df = pd.DataFrame(feature, columns=['brightness','blur','edge', 'edge_enhance','edge_enhance_more','contour','emboss','detail','smooth','smooth_more'])

#df = pd.DataFrame(feature, columns=['brightness','blur','edge', 'contour','emboss','detail','smooth'])
#df = pd.DataFrame(feature, columns=['brightness','blur','edge', 'contour'])
#print ("feature value:", df)
#remove the first row, because there exists redundant data
#df.drop(df.index[0])

#when dropping the column, use this code
#df.drop(['brightness'], 1, inplace=True)


#s = pd.Series(list(unique_artists), dtype="category")
#print label_data


#df['author'] = pd.Categorical.from_codes(list(unique_artists),list(unique_artists))


#df = pd.read_csv('breast-cancer-wisconsin.data.txt')
#df.replace('?',-99999, inplace=True)
#df.drop(['id'], 1, inplace=True)
#X = np.array(df.drop(['class'], 1))
#y = np.array(df['class'])

print ("")
print ("Split arrays or matrices into random train and test subsets")
print ("X_train, X_test, y_train, y_test ")
X_train, X_test, y_train, y_test = train_test_split(feature, label_data, test_size=0.2)
#print (X_train)
#print (y_train)

print ("")
print ("Fit the SVM model according to the given training data.")
print ("X_train, y_train")
svm_start_time = time.time()
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_train, y_train)
x_predict = clf.predict(X_test)
print ("")
print ("Perform classification on samples in X_test.")
print ("SVM Accuracy :", accuracy_score(y_test, x_predict)*100)
print("--- %s seconds ---" % (time.time() - svm_start_time))


# Create a random forest Classifier. By convention, clf means 'Classifier'
random_start_time = time.time()
clf2 = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
print()
print("Build a forest of trees from the training set (X_train, y_train)")
clf2.fit(X_train, y_train)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
print()
print("Predict class for X_test.")
x_predict_random_forest = clf2.predict(X_test)

# Create confusion matrix
#print "random forest :",r2_score(y_test, x_predict_random_forest)
print ("Random Forest Accuracy:", accuracy_score(y_test, x_predict_random_forest)*100)
print("--- %s seconds ---" % (time.time() - random_start_time))

#print np.reshape(x_predict_random_forest, [1,x_predict_random_forest.shape[0]])
#print (pd.crosstab(y_test, x_predict_random_forest,  rownames=None, colnames=None))