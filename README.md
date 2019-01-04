# Vanouver Special Classification

# Introduction
This project implements a deep learning image classification model to detect if a house matching the Vancouver Special house style is located in an image. he Vancouver Special is an architecture style for homes that was popular between the 1960s and 1980s. They can be characterized by their box-like shape, shallow roof that is normally split through the middle, and some sort of piece that makes the house appear as if its top and bottom floors were split. Many times the top and bottom halves of the street-facing side are covered in different material or painted with different colors. It is also popular to see a balcony on the second floor.

This project was originally created for a Machine Learning course at Simon Fraser University, but has been updated and worked on since. Furthermore, it was partly inspired through the website [vancouverspecial.com](vancouverspecial.com) (now referred to as the VS website), where its creator (K Higgins) seems to have wanted to log the locations of Vancouver Specials within the Vancouver civic area.  

The website is designed in such a way that each logged house address (house number and street name) is contained in a separate page along with the corresponding image of the house. The creator of the website also logged the location of the Vancouver Special house locations. The original goal for this project was to be able to speed up the logging process, where instead of physically looking for Vancouver specials, the final product would be able to gather images from Google Streetview, sift through them, and classify each image based on whether it contains a Vancouver Special or not. The target location for this was Burnaby, BC.

Two datasets were collected - one with house images collected from [vancouverspecial.com](vancouverspecial.com) (for positive examples) and [imagenet.com](imagenet.com) (for negative examples) and one with house images collected from the Google Streetview API. Data collection and results are discussed below. The code to run this project is discussed after.

# Data Collection
## VS website and Imagenet Dataset 

This data set consists of 1760 training images and 752 validation images. Each set has around half positive and half negative examples. Two different datasets were ultimately implemented. The first dataset was of the Vancouver Special house images were retrieved from the VS website.

As discussed previously, the website VS contains many different types of Vancouver Special examples. With this in mind, the image classification model that uses the VS website dataset would hopefully learn the decision boundaries between the classes in such a way that difference in features is accounted for. Furthermore, the images aren't all uniform, that is most images are taken from different angles with a number of houses obscured by objects such as trees, fences, etc. 

In order to retrieve the Vancouver Special House images, I wrote a program in Python using the package [Scrapy](https://scrapy.org/) to scrape the house images from the VS website. Luckily, the website creator  ordered the houses such that each has a unique number ID that corresponds to its place within an imaginary list of houses on the website. This number is contained in the page URL, which has the form [http://www.vancouverspecial.com/?num=1](www.vancouverspecial.com/?num=*), where * represents the position of the house within this list. This made the process very easy to iterate through the images and download them. In total, the website (as of December 2018) contains 1240 images of Vancouver Specials.

The negative examples, those that did not contain a Vancouver Special, were images of housing structures, including houses, apartments, etc. These images were retrieved from [Imagenet](http://image-net.org), which is a database that contains the URLs of around 14 million images. There are many different classes and subclasses of images - each class and their subclasses contain datasets for the images corresponding to said (sub)class. As said previously, the ImageNet database is used many times to train neural network image classifiers. 

Once retrieved, the images were shrunk down to dimensions <a href="https://www.codecogs.com/eqnedit.php?latex=$128\times&space;128$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$128\times&space;128$" title="$128\times 128$" /></a>. This was done to lighten the load on the model and decrease the dataset memory requirement. The images were split up randomly into the test and validation sets. 

## Google Streetview Image Datsets
A second training set was created after first running the dataset above. This dataset contains images from Google Streetview, which were retrieved using the addresses from the VS website. The addresses were retrieved using the Python module Scrapy. Images that corresponded to the addresses were retrieved using the Google Streetview API. Although in theory this idea seems perfect, a good portion of images retrieved in this manor did not contain a Vancouver Special. This could be caused by an error in the address logging by the VS website creator or an error from Google Streetview's algorithm for retrieving images. Although a Vancouver Special was not always in the retrieved image, the target house might have existed within the 360-panoramic like image that the address belongs to. With this in mind, I wrote a program that would retrieve an image from Google Streetview corresponding to an address. If the image contained a Vancouver Special, then it was saved immediately and another image corresponding to the next address was retrieved. If the image did not contain a Vancouver Special, then the program would 'rotate' about the address point by forty-five degrees and the rotated image was then retrieved. If after having rotated a total of 315 degrees, then the image was placed in a separate directory for later scrutiny. Because of this issue, only 1200 images could be used for the training set and 512 used for the validation set.

The negative examples were retrieved using the Vancouver address list from [openaddresses.io](openaddresses.io). Of course the images were searched through and those that contained Vancouver Specials were discarded. Lastly, to remedy the blurriness of smaller images, this dataset was reshaped to <a href="https://www.codecogs.com/eqnedit.php?latex=$256\times&space;256$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$256\times&space;256$" title="$256\times 256$" /></a>.

## Test Set
A test dataset was assembled using images from Google Streeview corresponding to addresses form Burnaby, the addresses uses were also  retrieved from [openaddresses.io](openaddresses.io). The Google Streetview test set was assembled by using the Python package [urllib](https://docs.python.org/3/library/urllib.html), which allows the user to retrieve an image given a URL that leads to said image. Google Inc. has created an API that allows one to easily access Google Streetview images, the user only needs to specify the location of the desired image as well as image size, orientation, and several other categories within the URL. The website [openaddresses.io]{openaddresses.io} was used to retrieve addresses of properties within Vancouver. Fifty-seven Google Streetview images that contained Vancouver Special houses and forty-nine Google Streetview images that didn't contain a Vancouver Special house were collected. The original dimensions were 600 x 600. The images were resized to <a href="https://www.codecogs.com/eqnedit.php?latex=$128\times&space;128$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$128\times&space;128$" title="$128\times 128$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=$256\times&space;256$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$256\times&space;256$" title="$256\times 256$" /></a>, depending on the training set. 

# Model and Experiment

The model used for this project was a pre-trained VGG16 network. Keras' pretrained VGG16 CNN provides for a powerful image classification model while also being manageable for computers that run on CPU power. The model was previously trained on the ImageNet database - this is especially good because the model should be able to hopefully already be able to distinguish house images from other images. The model was run in Python using Tensorflow and Keras. The pre-trained model, from Keras, had its fully connected layer replaced so that the layer could train fresh on the new training dataset previously described. The fully connected layer sends the output from the pre trained model first through a ReLU activation, then performs dropout with <a href="https://www.codecogs.com/eqnedit.php?latex=$p=.5$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$p=.5$" title="$p=.5$" /></a>, and then through a sigmoid activation function. The output can then be interpreted as a probability of how likely that the input image contains a Vancouver Special. The optimization process used is RMSprop with a binary cross entropy (also referred to as softmax) loss function. 

The procedure was to first compute the weights of all the datasets on the model excluding the fully connected layer. The weights were saved and then those that corresponded to the training set  served as input for the fully connected layer to be trained. The model was also validated and tested on in the same fashion - using the weights corresponding to the appropriate dataset. The model was trained using total epoch numbers <a href="https://www.codecogs.com/eqnedit.php?latex=$\epsilon&space;=&space;5,10,20,50$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\epsilon&space;=&space;5,10,20,50$" title="$\epsilon = 5,10,20,50$" /></a>. 

# VS Wesbsite Dataset Results

The model performs well on the training and validation sets retrieved from the VS website and Imagenet. A plot is given below that illustrates accuracy of the model with respect to the number of epochs used. 

<img src="https://i.imgur.com/nCgfW8g.jpg" width="480" align="center">


The accuracy is plotted with respect to the number of epochs used. Notice that the accuracies seem to level out, though there is still a noticeable and drastic change in certain steps for the validation accuracy. While the model performed spectacular on the training and validation sets, there is quite a different story when the model is tested on images retrieved from Google Streetview. A plot illustrating the accuracy of the model on the two Google Streetview image datasets with respect to epoch <a href="https://www.codecogs.com/eqnedit.php?latex=$\epsilon&space;=&space;5,10,20,50$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\epsilon&space;=&space;5,10,20,50$" title="$\epsilon = 5,10,20,50$" /></a> is given below.

<img src="https://i.imgur.com/U8a901Q.jpg" width="480">


The model did not perform nearly as well on the cropped images as those in the validation set. While this is true, the model seems to improve gradually with increasing epoch number. Below is a sample of a few images that were the model failed to classify along with their corresponding predicted output and true output (we denote the class of positive examples by 1 and 0 otherwise). The corresponding probability is given as well - this is the probability that the model thinks that the input is a Vancouver Special. 

<img src="https://i.imgur.com/5hvf458.png" width="530">


Moreover, the model does much worse on the images that were not cropped. After checking the test performance on each input, the model seems to fail completely. The models  predicts all non-cropped Google Streetview images as not Vancouver Specials. 


# Results of Google Streetview Dataset

The model trained and validated on the dataset retrieved from Google Streetview improved in the sense that it performed better on Google Streetview images as compared to the model trained on the dataset retrieved from the VS website and Imagenet. The test and validation accuracy plots corresponding to epochs of 5, 10, 20, and 50 are given below. 

<img src="https://i.imgur.com/HgYLOoi.jpg" width="530">


For epoch number of 50, the model performs well considering the smaller dataset size. Notice that the model seems to fail entirely for epoch number of 20. This result seems to be puzzling and occurs at random. This instability will be studied further. 

Although this model performed somewhat well and better than chance, one might want to achieve better accuracy. This should be fixed by implementing data augmentation to increase the dataset size. 


# Analysis and Discussion 

At first glance, one can see a few reasons for why the dataset from the VS website and IMageNet perform poorly on the Google Streetview test images. One such reason is the image dataset itself. After taking a look through the positive example data set, one might notice that most images are zoomed in on the houses. Not much external ``noise" is present as compared to a Google Streetview image. A possible remedy for this problem could be to collect the training and validation data sets by first retrieving the addresses from \webp and then retrieve each address' corresponding Google Streetview image. This would hopefully help the model generalize more and be able to "look" for Vancouver Special houses that might be tucked away in images. While this remedy could help, it is not necessarily ideal as the images will have to also be looked through to see whether they actually do contain the Vancouver Special. Secondly the negative examples will have to be collected in the same manner and one could only assemble this dataset by looking to see whether an image contains a Vancouver Special house or not. Although if time presents no issue, this route might prove successful. 

A second fix to the problem of overfitting is through data augmentation. Each image can be augmented through different ways such as rotation, a shear mapping, etc. This increases the size of the data set and should aid in keeping the model's variance down. While this is attractive, a large dataset can take much longer to train and test - especially on a computer without a GPU installed. 

One other problem that arises in this model is from the input resolution. While one can reasonably distinguish large objects in a <a href="https://www.codecogs.com/eqnedit.php?latex=$128\times&space;128$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$128\times&space;128$" title="$128\times 128$" /></a> image, smaller objects become harder to recognize. For example, the image below was retrieved from Google Streetview and contains a Vancouver Special in a yellow bounding box.


![alt text](https://i.imgur.com/aiaMdIP.jpg)

Notice that when zooming in on the Vancouver Special, it becomes extremely pixelated. This can cause major problems for the classifier, as it might lose the ability to tell the features apart. Thus it would most likely be more useful to scale images to have better resolution than <a href="https://www.codecogs.com/eqnedit.php?latex=$128\times&space;128$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$128\times&space;128$" title="$128\times 128$" /></a>. But just as above, this method will increase the model size by a large amount, causing memory and time problems. We lastly address the model's failure on non-cropped Google Streeview images as mentioned in the Results section. 

When applied to images with multiple houses, e.g. Google Streetview images, the model consistently predicts that the input does not contain a Vancouver Special. Possible reasons for this are described above - it is possible that the model overfit the training dataset. As previously discussed, the non-cropped Google Streetview images are almost ``expert level" datapoints - the model would have to be able to classify based on if it can detect the features corresponding to a Vancouver Special. Currently it seems that the model more so predicts whether a test image is simply of a Vancouver Special, not whether it necessarily contains one.

# Thoughts and Future Updates

This project served as a major lesson in data assimilation as well as general neural networks. The main challenge in this project was collecting images in order to form the data set. 

A future update will include data augmentation as well as tuning of some of the convolutional layers. A nice extension would be to implement localization such that the program draws a bounding box around all Vancouver Specials in an image, if they are contained within it. 

# How to Run

The programs were run in [Python 3.6](https://www.python.org/downloads/release/python-360/) on CPU power. Modules used were Keras, matplotlib, numpy, pandas, PIL , shutil, Tensorflow, and urllib.
## Collecting the VS website pictures
In order to download the house images from the [vancouverspecial.com](vancouverspecial.com), I implemented the Python module Scrapy to scrape the site. One can easily implement this by using the vspecial.py program attached. One must first create a Scrapy project, which can be done by typing the following into the command line.
```
scrapy startproject vspec_crawler
```
The user must then change directories into the new vspec_crawler directory. One can then initialize the Scrapy spider as follows.
```
scrapy genspider vspec_img_crawl vancouverspecial.com
```
Before the user decides to crawl the page, they must replace the vspec_img_crawl.py program (located in vspec_crawler/vspec_crawler/spiders) with vspec_img_crawl.py on this page. The spider can then crawl the page using the following command.
```
scrapy crawl vspec_img_crawl
```

## Collecting the Vancouver Special addresses

The second data set uses pictures from the Google Streetview API with positive training exmples corresponding to addresses from the VS website. The Python module Scrapy is used to scrape the VS website for house addresses of Vancouver Specials in Vancouver, BC. This program can be run similar to above.

```
scrapy startproject vspec_address_crawler
```
The user must then change directories into the new vanspec directory. One can then initialize the Scrapy spider as follows.
```
scrapy genspider address_crawler vancouverspecial.com
```
Before the user decides to crawl the page, they must replace the address_crawler.py program (located in vanspec/vanspec/spiders) with address_crawler.py on this page. The spider can then crawl the page using the following command.
```
scrapy crawl address_crawler
```

## Collecting Images from Google Streetview

The program get_ngoogle_examples.py downloads images from the Google Streetview API. It requires the number of images to be downloaded (this number is stored in the num_imgs variable), the path to a csv file containing addresses (path string is stored in addresses variable), and a string with the user's Google Streetview API key. The program was run to read csv files from [openaddresses.io](openaddresses.io). The images will be saved in the newly created directory, google_examples.

## Collecting the Google Streetview Dataset
In order to download the the Google Streetview images corresponding to addresses from the VS website, I wrote the get_vanspec_google.py program.  It requires the path to a csv file containing addresses (path string is stored in addresses variable) and a string with the user's Google Streetview API key.

For each address, the program will doanload the corresponding Google Streetview image and ask the user if the image corresponding
to the address is sufficient. All images are saved in the directory specified by the variable dir. If the image is sufficient, then it is saved in the subdirectory image_lib. If not, then the user specifies 'n'. The program retrieves another image that is rotated 45 degrees about the address point from the last image. Once the camera has been rotated 315 degrees, then the image is saved the directory specified by the variable qimage_dir. 

## Running the Model

Once the dataset as been prepared, the user will run model.py. The images must be split up into a training set and a validation set. The paths to these are specified by the variables train_data_dir and validation_data_dir. Furthermore, these directories must contain a separate directory for each class. For example, the positive examples were in a directory titled '1' and negative examples in a directory titles '-1'. Also note that each dataset should be divisible by the batch number, which is given by the variable batch_size. 

The program first passes the datasets through the convolutional layers and saves the outputs in npy files corresponding to each dataset path. The program also saves the sorted list of paths of images in the test set from Google Streetview, this is to account for the order with which Keras retrieves files in a directory. The paths are stored in a csv file, which is then used to view specific test examples that the model was tested on. Note that the validaiton set is not the test set. 

The model is then trained on the specified training set and is also validated. It is then tested on Google Streetview images, both cropped and non cropped. The accuracy for each epoch is plotted as well. The program runs four times through total epic numbers 5, 10, 20, and 50. 
