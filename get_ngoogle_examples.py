'''
get_ngoogle_examples.py 
'''

import urllib.request
from urllib.parse import quote
import os
import pandas as pd
import numpy as np
import sys
import math
from PIL import Image, ImageFont, ImageDraw, ImageEnhance 

num_imgs = 2000
addresses = 'vancouver.csv'
key = ''

image_lib = 'negative_examples'
myloc = os.getcwd() 



def GetStreet(Loc, SaveLoc, Name):

	
	base = ('https://maps.googleapis.com/maps/api/streetview?size=600x600'
	+ '&key=' + key + '&location=')
	
	fi = Name + "_img" + ".jpg"
	filename = os.path.join(SaveLoc,fi)

	urllib.request.urlretrieve(base + urllib.request.quote('\'') + 
		urllib.request.quote(Loc) + urllib.request.quote('\'') +'/', filename)

address_list = pd.read_csv(addresses, header=None, na_values='nan')


if not(dir in os.listdir(os.getcwd())):
	os.makedirs(image_lib)


for i in range(1, num_imgs):
	if (str(address_list.loc[i,3]) == 'nan'):
		continue

	address = str(address_list.loc[i,2]) + ' ' + str(address_list.loc[i,3]) + ' Vancouver, BC'

	GetStreet(Loc = address, SaveLoc = myloc+'/' + image_lib + '/', Name = str(i))
	print(i)

		







