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
base_address = '/Users/User/PycharmProjects/PR-termProject/wikiart/'
style = 'Impressionism/'
train_file = 'impressionists.csv'
filepath = base_address
unique_artists = set()
unique_link = list()

#file categories are ids,location,artist,class
paintings_by_artist = pd.read_csv(filepath+train_file, names=['ids','location','artist', 'class'], header=0)
paintings_by_artist['absolute_location'] = base_address +style+ paintings_by_artist['location']
#paintings_by_artist = paintings_by_artist.sort_values('class')
ImageFile.LOAD_TRUNCATED_IMAGES = True

rows_count = paintings_by_artist.shape[0]
cols_count = 7
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
    feature[i][3] = find_contour(img)
    feature[i][4] = find_emboss(img)
    feature[i][5] = find_detail(img)
    feature[i][6] = find_smooth(img)
    #s = pd.Series(blur, brightness, edge, edge_enhance, edge_enhance_more, contour,emboss, detail, smooth, smooth_more, label)
    #table = plt.table(blur)
    unique_artists.add(label_data[i])


# Create a dataframe with the ten feature variables
#df = pd.DataFrame(feature, columns=['brightness','blur','edge', 'edge_enhance','edge_enhance_more','contour','emboss','detail','smooth','smooth_more'])
df = pd.DataFrame(feature, columns=['brightness','blur','edge', 'contour','emboss','detail','smooth'])
print "feature value:", df
#remove the first row, because there exists redundant data
#df.drop(df.index[0])

#when dropping the column, use this code
#df.drop(['brightness'], 1, inplace=True)

#from sklearn.datasets import load_iris
#iris = load_iris()

# Add a new column with the species names, this is what we are going to try to predict
#print label_data
#print list(unique_artists)
#print array

#s = pd.Series(list(unique_artists), dtype="category")
#print label_data


#df['author'] = pd.Categorical.from_codes(list(unique_artists),list(unique_artists))


#print(feature)
#print label_data
#print np.shape(label_data)

#df = pd.read_csv('breast-cancer-wisconsin.data.txt')
#df.replace('?',-99999, inplace=True)
#df.drop(['id'], 1, inplace=True)
#X = np.array(df.drop(['class'], 1))
#y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(feature, label_data, test_size=0.2)
print("X_train", X_train)
print("X_test", X_test)
print("len(X_train)", len(X_train))
print("len(X_test)", len(X_test))
print("y_train", y_train)
print("y_test", y_test)
print("len(y_train)", len(y_train))
print("len(y_test)", len(y_test))

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_train, y_train)
x_predict = clf.predict(X_test)
print "x_predict", x_predict
print "len(x_predict)",len(x_predict)
#print "mean_squere_error", mean_squared_error(y_test, x_predict )
#print "svm :",r2_score(y_test, x_predict )
print "svm :", accuracy_score(y_test, x_predict)
#print "svm :",r2_score(x_predict , y_test )


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf2 = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf2.fit(X_train, y_train)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
x_predict_random_forest = clf2.predict(X_test)

# Create confusion matrix
#print "random forest :",r2_score(y_test, x_predict_random_forest)
print "random forest:", accuracy_score(y_test, x_predict_random_forest)

print y_test
#print np.reshape(x_predict_random_forest, [1,x_predict_random_forest.shape[0]])
print pd.crosstab(y_test, x_predict_random_forest,  rownames=None, colnames=None)