import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import SGD
import image
import sys
import csv
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.imagenet_utils import decode_predictions

# cd streetview/"untitled folder 2"/data/new data/Train
img_width, img_height = 128*2, 128*2

top_model_weights_path = 'bottleneck_fc_model.h5'

# Paths to datasets, must be given in the below form
train_data_dir = 'data/new data/Train'
validation_data_dir = 'data/new data/Validation'
test_cropped_dir = 'data/test/cropped'
test_og_dir = 'data/new data/Test'

test_file_c = 'tesfilec.csv'
test_file_og = 'tesfileog.csv'

# image library sizes must be divisible by batch_size
train_count = 1200
validation_count = 512
cropped_num = 96

batch_size = 16
epochs = [5, 10, 20]


def saveNames(filename, picNames , classes):

	with open(filename,'w') as resultFile:
		wr = csv.writer(resultFile, delimiter=',')
		for ii in range(len(picNames)):
			wr.writerows([[picNames[ii], str(classes[ii]), str(0)]])

def print_results(filename, classes, probs):
	test_labels = []
	e = 0 # additive variable that increases by one with each correct predicted example
	count = 0
	
	with open(filename,'r') as resultFile:
		csv_reader = csv.reader(resultFile, delimiter=',')
		
		
		print(['filename', 'True Value', 'Predicted Value', 'Probability'])
		for row in csv_reader:
			row[0] = test_cropped_dir + '/' + row[0]
			row[2] = classes[count,0]
			test_labels.append(int(row[1]))
			
			if not (test_labels[count] == row[2]):
				e+=1 # e corresponds to the accuracy
				print(str([row[0], row[1], row[2], probs[count, 0]]) + '---')
			else:	
				print([row[0], row[1], row[2], probs[count, 0]])
			count+=1
	return e, count

# Save the weights from the test images from Google Streetview 
def saveTest(model, example_num, example_info, example_dir, feature_file):

# 	test_datagen = ImageDataGenerator(rescale=1. / 255) #normalize and create the data generator 
# 	test_generator = test_datagen.flow_from_directory(
# 		test_cropped_dir,
# 		target_size=(img_width, img_height),
# 		batch_size=batch_size,
# 		class_mode=None,
# 		shuffle=False)
# 	saveNames(test_file_c, test_generator.filenames, test_generator.classes) #save file names for later testing
# 	test_data = model.predict_generator(test_generator, cropped_num // batch_size) #retrieve weights
# 	np.save('bottleneck_features_test_c.npy', test_data) #save weights for prediction
		
		
	test_datagen = ImageDataGenerator(rescale=1. / 255)
	test_generator = test_datagen.flow_from_directory(
		example_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)

	saveNames(example_info, test_generator.filenames, test_generator.classes)
	test_data = model.predict_generator(test_generator, example_num // batch_size)
	np.save(feature_file, test_data)

# Save the weights from the training and validation images
def saveTrain(model):
	datagen = ImageDataGenerator(rescale=1. / 255)
	generator = datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)

	bottleneck_features_train = model.predict_generator(generator, train_count // batch_size)
	
	np.save('bottleneck_features_train.npy',
			bottleneck_features_train)
	
	generator = datagen.flow_from_directory(
		validation_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)
	bottleneck_features_validation = model.predict_generator(generator, validation_count // batch_size)
	np.save('bottleneck_features_validation.npy',
			bottleneck_features_validation)

def save_bottlebeck_features():


# A few different models can be used, here I used the VGG16
	model = applications.VGG16(include_top=False, weights='imagenet')
# 	model = applications.resnet50.ResNet50(include_top=False, weights='imagenet')
# 	model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')

	saveTest(model, cropped_num, test_file_og, test_og_dir, 'bottleneck_features_test_og.npy')	
	
	saveTrain(model)

# Training
def evaluate_model(i, fig):

	train_data = np.load('bottleneck_features_train.npy')	
	train_labels = np.array([0] * 600  + [1] * 600)
	
	validation_data = np.load('bottleneck_features_validation.npy')
	validation_labels = np.array([0] * 256 + [1] * 256)

# Create the fully connected layer
	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='rmsprop',
				  loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the previously computed weights
	hist = model.fit(train_data, train_labels, verbose = 1,
			  epochs=epochs[i],
			  batch_size=batch_size,
			  validation_data=(validation_data, validation_labels))
	model.save_weights(top_model_weights_path)
	
	
	
	print('----')
	print('epoch = ' + str(epochs[i]))
 	

# 	test_data = np.load('bottleneck_features_test_c.npy')
# # 	Compute the predicted classes from the test images
# 	predictions1 = model.predict_classes(test_data)
# # 	Compute the probability that the model thinks an input is a Vancouver Special image
# 	predictions2 = model.predict_proba(test_data)
# 	(e, count) = print_results(test_file_c, predictions1, predictions2)
# 	print('accuracy cropped: ' + str(1-e/count))
	
	test_data = np.load('bottleneck_features_test_og.npy')
	predictions1 = model.predict_classes(test_data)
	predictions2 = model.predict_proba(test_data)
	(e, count) = print_results(test_file_og, predictions1, predictions2)
	print('accuracy og: ' + str(1-e/count))
	
	
	print('----')





# Plot training and validation accuracy across each epoch

	
	ep5 = hist.history
	ax1 = fig.add_axes()
	
	plt.subplot(221+i)

	plt.plot(range(1,epochs[i]+1), ep5['val_acc'], label='Validation')   #top left
	plt.plot(range(1,epochs[i]+1), ep5['acc'], label='Training')   #top left
	if i == 2:
		plt.legend(loc='lower right')
		plt.xlabel(r'epoch',fontsize=16)
		plt.ylabel(r'Accuracy',fontsize=16)


	return fig

if __name__ == '__main__':

	save_bottlebeck_features()
	for i in range(len(epochs)):

		if i == 0:
			fig = plt.figure()
		fig = evaluate_model(i,fig)
	plt.savefig('eplt' +'.eps', format='eps')
