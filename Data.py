
import os
from os import listdir
from shutil import copyfile
from random import randint
from random import seed
from random import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

#This class handles all the importation, setup, seperation, and transformation of the data
#for the models to use.
class Data:

    #const values
    DATASET_DIR = "asl_dataset/new_asl_dataset/train/"
    DATASET_HOME = 'asl_dataset/Data/'
    SUB_DIRS = ['train/', 'test/']
    LABEL_DIRS = [#'0/', '1/', '2/', '3/', '4/', '5/', '6/', '7/', '8/', '9/',
             'a/', 'b/', 'c/', 'd/', 'e/', 'f/', 'g/', 'h/', 'i/', 'j/', 'k/',
             'l/', 'm/', 'n/', 'o/', 'p/', 'q/', 'r/', 's/', 't/', 'u/', 'v/',
             'w/', 'x/', 'y/', 'z/', 'Blank/'
    ]


    def __init__(self, batch_size, img_shape, rotation=0, zoom_range=0, shift_range=0, val_ratio=0.2, seperate_data=False):
        """
        
        """
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.val_ratio = val_ratio

        #if data is wanted to be seperated
        if seperate_data:
            self.seperate_Images(val_ratio)

        self.train_image_gen = self.image_Generator(subset='train', batch_size=batch_size, new_img_shape=img_shape, rotation=rotation, zoom_range=zoom_range, shift_range=shift_range)
        self.test_image_gen = self.image_Generator(subset='test', batch_size=batch_size, new_img_shape=img_shape, rotation=rotation, zoom_range=zoom_range, shift_range=shift_range)


    #Copies the data from the dataset into the Data folder with it seperated into test and train dataset.
    def seperate_Images(self, val_ratio):
        """
        seperates the DATASET_DIR contents into a train and test folder
        """

        #Remove the DATASET_HOME directory if it exists
        if os.path.exists(self.DATASET_HOME):
            shutil.rmtree(self.DATASET_HOME)

        #creates the train and test directories
        #in both of those creates a folder for each label
        for subdir in self.SUB_DIRS:
            for labldir in self.LABEL_DIRS:
                newdir = self.DATASET_HOME + subdir + labldir
                os.makedirs(newdir, exist_ok=True)

        # seed random number generator
        seed()

        # copy training dataset images into subdirectories
        for src_directory in self.LABEL_DIRS:
            for file in os.listdir(self.DATASET_DIR + src_directory):
                src = self.DATASET_DIR + src_directory + '/' + file
                dst_dir = 'train/'
                if random() < val_ratio:
                    dst_dir = 'test/'

                #copies file from current location to the Data file location
                dst = self.DATASET_HOME + dst_dir + src_directory + '/' + file
                copyfile(src, dst)



    
    def image_Generator(self, subset, batch_size, new_img_shape, rotation=0, zoom_range=1, shift_range=0):
        """
        Preprocesses the test and train data and augments it using the keras image data generator.

        Parameters:
        -subset: test or train
        -batch_size: size of the batches 
        -new_img_shape: the shape to scale the images to
        -rotation, zoom_range, shift_range: Augment parameters for training image generator

        Returns:
        - the image data generator 
        """

        #if this is for training  run all augments on it
        if subset == 'train':
            # expand our training dataset -- randomly transform the image datasets
            data_gen = ImageDataGenerator(
                            rescale=1.0 / 255.0,  # scales image pixel values between 0 and 1
                            rotation_range=rotation,  # randomly rotate images in the range (degrees, 0 to 180)
                            zoom_range=zoom_range,  # Randomly zoom image
                            horizontal_flip=True,
                            width_shift_range=shift_range,   # randomly shift images horizontally (fraction of total width)
                            height_shift_range=shift_range  # randomly shift images vertically (fraction of total height)
            )
        else:
            data_gen = ImageDataGenerator(rescale=1.0 / 255.0)

        #shuffles all the different labels together within each of the test and train so it isn't fed each letter in order
        image_gen = data_gen.flow_from_directory(self.DATASET_HOME + '/' + subset + '/',
                        target_size=new_img_shape[:2],
                        color_mode='rgb',
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=True
        )

        # pixel value should be between 0~1, the shape should be 200x200x3
        # -- decided by the ImageDataGenerator class
        return image_gen



