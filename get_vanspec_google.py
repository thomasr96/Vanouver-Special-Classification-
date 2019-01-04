'''
get_vanspec_google.py downloads images from the Google streetview API (user must specify
API key). 

The program requires a list of addresses. It will ask the user if the image corresponding
to an address is sufficient. All images are saved in the directory specified by the variable
'dir'. If the image is sufficient, then it is saved in the subdirectory 'image_lib'. If not,
then the user specifies 'n'. The program retrieves another image that is rotated 45 
degrees about the address point from the last image. Once the camera has been rotated 315 
degrees, then the image is saved the directory specified by 'qimage_dir'
'''

import shutil
import urllib.request
from urllib.parse import quote
import os
import pandas as pd
import numpy as np
import sys
import math
from PIL import Image, ImageFont, ImageDraw, ImageEnhance  

myloc = os.getcwd() 
addresses = 'address_list.csv'
key = ''
degrees = np.arange(8)*45
# image_dir = 'correct_images'
qimage_dir = 'questionable_images'
dir = 'vanspec_google'
image_lib = 'image_library'

def GetStreet(H, Loc, SaveLoc, Name):
	img_resp = 'n'
	pitch = ''
	lock = 1
	lock_count = 0
	deg_text = ''
	pcount = 0
	Heading = ''
# 	loop as long as image is not correct
	while img_resp == 'n':

		base = ("http://maps.googleapis.com/maps/api/streetview?size=600x600" + Heading + pitch
		+ "&key="+ key +"&location=")

		address = Loc[1:len(Loc)-1]
		

		fi = address + '__'+  Name + ".jpg"
		filename = os.path.join(SaveLoc,fi)
		
# 		retrieve image
		urllib.request.urlretrieve(base + urllib.request.quote('\'') + 
		urllib.request.quote(address) + urllib.request.quote('\'') +'/', filename)
# 		test the image
		im = Image.open(filename)
		im.show()
		img_resp = input("Is this image correct?" + deg_text + "(y/n/?)")
		
		if img_resp == 'n':

			if pcount == len(degrees):
				print('Degree choices have run out. Image saved.')
				shutil.move(filename, dir + '/' + qimage_dir + '/' + fi)		
				img_resp = '?'
			else:
# 				rotate about address
				deg_text = "-- degrees = " + str(degrees[pcount]) + " --"
				Heading = '&heading=' + str(degrees[pcount])
				pcount+=1
				os.remove(filename)		
		elif img_resp == '?':
			shutil.move(filename,  dir + '/' + qimage_dir + '/' + fi)		

	
	

				
city = pd.read_csv(addresses, header=None, na_values='nan')

if not(dir in os.listdir(os.getcwd())):
	os.makedirs(dir)
	os.makedirs(dir + '/' + qimage_dir)
	os.makedirs(dir + '/' + image_lib)

for i in range(0, 1240):
	print('Image number ' + str(i))
	address = str(city.loc[i,0]) 
	GetStreet(H = '-1', Loc = address, SaveLoc = myloc+'/' + dir + '/' + image_lib + '/', Name = str(i))


	
		







