# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 01:01:25 2017

@author: Melanie
"""

#Prep_dataset.py 
csv = '../data/top15.csv'
wiki_art_path = '/home/bhavika/wikiart/'
original_img_path = '/home/bhavika/wikiart/Impressionism/'
small_img_path = '../data/impress_data_small/'

#LoadDataset.py
img_count = 600

#CNN.py
bs = 15
number_of_classes = 15
epochs = 100
model_folder = '../models/'

#PredictArtist.py
model_path = '../models/ArtistIdentificationModel.h5'
test_samples = 4

#LoadDataset_predictonly
demo_path = '../data/demo/'