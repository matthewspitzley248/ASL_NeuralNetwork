
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential
from keras.src.layers import BatchNormalization, Dropout
from keras.src.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os

class Model:

    MODEL_DIR = 'Models/'

    #
    def __init__(self, data, model_name=None):

        self.data = data

        if model_name is None:
            self.new_Model()
        else:
            self.model = tf.keras.models.load_model('Models/'+model_name)


    def new_Model(self):
        """
        Builds a new CNN Model
        """
        # building a linear stack of layers with the sequential model
        # build a sequential model
        self.model = Sequential()
        #model.add(InputLayer(input_shape=new_img_shape))

        # 1st conv block
        self.model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=self.data.img_shape))
        self.model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
        self.model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
        # 2nd conv block
        self.model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
        self.model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
        self.model.add(BatchNormalization())
        # 3rd conv block
        self.model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
        self.model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
        self.model.add(BatchNormalization())
        # ANN block
        self.model.add(Flatten())
        self.model.add(Dense(units=100, activation='relu'))
        self.model.add(Dense(units=100, activation='relu'))
        self.model.add(Dropout(0.25))
        # output layer
        self.model.add(Dense(len(self.data.LABEL_DIRS), activation='softmax'))

        # compiling the sequential model
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        print(self.model.summary())


    def fit_Model(self, epochs):
        """
        """
        early_stopping = EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.001,
                            patience=5,
                            restore_best_weights=True,
                            verbose=1
        )

        reduce_learning_rate = ReduceLROnPlateau(
                            monitor='val_accuracy',
                            patience=2,
                            factor=0.5,
                            verbose=1
        )

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