
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os

class Model:
    """
    Handles creation and fitting of the model on the given data object from Data.py.
    """

    #global value for location of the saved models
    MODEL_DIR = 'Saved_Models/'

    
    def __init__(self, data, model_name=None):
        """
        creates or loads a model and saves the data object
        """

        self.data = data

        #if there is no model to load create a new one
        if model_name is None:
            self.new_Model()
        else:
            self.model = tf.keras.models.load_model(self.MODEL_DIR+model_name)


    def new_Model(self):
        """
        Builds a new CNN Model
        """
        # building a linear stack of layers with the sequential model
        # build a sequential model
        self.model = Sequential()
        #model.add(InputLayer(input_shape=new_img_shape))

        #1st conv block
        self.model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=self.data.img_shape))
        self.model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
        self.model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
        #2nd conv block
        self.model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
        self.model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
        self.model.add(BatchNormalization())
        #3rd conv block
        self.model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
        self.model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
        self.model.add(BatchNormalization())
        #Dense block
        self.model.add(Flatten())
        self.model.add(Dense(units=100, activation='relu'))
        self.model.add(Dense(units=100, activation='relu'))
        self.model.add(Dropout(0.25))
        #output layer
        self.model.add(Dense(len(self.data.LABEL_DIRS), activation='softmax'))

        #compiling the sequential model
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        print(self.model.summary())


    def fit_Model(self, epochs):
        """
        fits the model and then saves it to the file location increasing the number it is saved as by 1
        Uses an early stopping and reduced learning rate.
        Runs for the passed number of epochs at most.
        Saves the model history as part of the model object.

        Parameters:
        -epochs: max number of epochs 
        """

        #early stopping runs off of the val_loss and restores back to the best epoch
        early_stopping = EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.001,
                            patience=5,
                            restore_best_weights=True,
                            verbose=1
        )

        #reduces the learning rate based off of the val_accuracy by half.
        reduce_learning_rate = ReduceLROnPlateau(
                            monitor='val_accuracy',
                            patience=2,
                            factor=0.5,
                            verbose=1
        )

        #saves the model_history within the model object so it can be used for graphs later
        self.model_history = self.model.fit(
                                self.data.train_image_gen,
                                epochs=epochs,
                                validation_data=self.data.test_image_gen,
                                callbacks=[early_stopping,reduce_learning_rate],
                                verbose=1
        )


        #saving the newly trained model
        #this finds the largest number to make the new model 1 larger
        largeNum = 0
        for file in os.listdir(self.MODEL_DIR):
            if (int(file[5:file.index('.keras')]) > largeNum):
                largeNum = int(file[5:file.index('.')])

        model_name = 'Model' + str(largeNum + 1) + '.keras'

        self.model.save(self.MODEL_DIR+model_name)