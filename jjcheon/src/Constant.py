"""
**************PLEASE READ ME*******************

style, train_file variables can be modified for training  by using SVM, Random Frest,
train file location is base_address
If you can change the traning file, you should change style and train_file.
Additionally, you don't need to remove "/". Just leave it


*_explorer variables is for explorer.py.
That file is used for train_file creation
If you want to create the new train file, you can change  top40_location_and_top40_file_exploer variable in the exploerer.py

we will make one train file.
if possible, top40_location_and_top40_file_exploer keep consistency with style, train_file

"""

base_address = 'c:\users\jjche\PycharmProjects\\688-term\wikiart\\'
style = 'Impressionism\\'
train_file = 'top40.csv'



###Normal-Demo
#
normal_demo_csv_file = 'NormalTestDataset.csv'
saved_trained_file = 'finalized_model_random_forest.sav'
base_csv_file_location = 'c:\users\jjche\PycharmProjects\\688-term\\'
loaded_model_filename = base_csv_file_location




###Random-Demo
#randome image
image = 'landscape.jpg'
saved_trained_file = 'finalized_model_random_forest.sav'
base_image_file_location = 'c:\users\jjche\PycharmProjects\\688-term\\'
loaded_model_filename = base_image_file_location+saved_trained_file
absolute_link  = base_image_file_location+image




#style = 'Impressionism/'
#train_file = 'top40.csv'



mapping_file = 'author_mapping.csv'
style_explorer = 'Impressionism'
train_file_explorer = '/artist_train.csv'
top40_location_and_top40_file_exploer = './wikiart/top40.csv'
impressionist_location_and_impressionist_file_exploer = './wikiart/impressionists.csv'