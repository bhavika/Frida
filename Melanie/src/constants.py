# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 01:01:25 2017

@author: Melanie
"""

#Prep_dataset.py 
csv = 'C:\\Users\\Melanie\\Desktop\\csvFiles\\data\\top15.csv'
#csv = 'C:\\Users\\Melanie\\Documents\\GitHub\\Frida\\Bhavika\\data\\top15.csv'
wiki_art_path = 'C:\\Users\\Melanie\\Desktop\\WikiartData\\wikiart\\Impressionism\\'
original_img_path = 'C:\\Users\\Melanie\\Desktop\\impress_data'
small_img_path = 'C:\\Users\\Melanie\\Desktop\\impress_data_small\\'

#LoadDataset.py
img_count = 750
#img_count = 600

#CNN.py
bs = 15
number_of_classes = 15
epochs = 100
model_folder = 'C:/KerasModels/Model'

#PredictArtist.py
model_path = 'C:\KerasModels\Model2017-11-27 23-00-27\ArtistIdentificationModel.h5'
test_samples = 16

#LoadDataset_predictonly
demo_path = 'C:\\Users\\Melanie\\Desktop\\randImg\\'