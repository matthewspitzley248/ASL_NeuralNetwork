import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class Plot:
    """
    Handles all the graphs and plotting of images.
    """

    def __init__(self, model, data):
        """
        Saves the data and model object 

        Parameters:
        -model: model object
        -data: data object that the model was run on
        """
        self.model = model
        self.data = data

    def plot_training_history(self):
        """
        Creates two graphs side by side that show the Accuracy and Loss
        over the course of the model being trained.
        """

        model_history = self.model.model_history

        train_acc = model_history.history['accuracy']
        epochs = [i for i in range(len(train_acc))]
        fig, ax = plt.subplots(1, 2)
        #train_acc = model_history.history['accuracy']
        train_loss = model_history.history['loss']
        val_acc = model_history.history['val_accuracy']
        val_loss = model_history.history['val_loss']
        fig.set_size_inches(16, 9)

        ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
        ax[0].plot(epochs, val_acc, 'ro-', label='Testing Accuracy')
        ax[0].set_title('Training & Validation Accuracy')
        ax[0].legend()
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")

        ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
        ax[1].plot(epochs, val_loss, 'r-o', label='Testing Loss')
        ax[1].set_title('Testing Accuracy & Loss')
        ax[1].legend()
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        plt.show()

    def plot_image(self, predictions_array, predicted_label, true_label, img):
        """
        Shows an image that it was predicted on and if it is right or not

        Parameter:
        -predictions_array: the weights array on the predicted value
        -predicted_label: the predicted value
        -true_label: the actual value for the image
        -img: image to plot
        """
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                    100*np.max(predictions_array),
                    true_label),
                    color=color)

    def plot_value_array(self, predictions_array, true_label):
        """
        Displays a bar graph on the weights of what it thought it was.

        Parameters:
        -predictions_array: the weights array on the predicted value
        -true_label: actual value for the image
        """
        plt.grid(False)
        plt.xticks(range(len(predictions_array)), self.data.label_info)
        plt.yticks([])
        thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color('red')
        #thisplot[true_label].set_color('blue')

    def gen_image_array(self, predictions, img_prediction, img_labels, images):
        """
        Creates a 3x3 of images with their predicted labels, and the true labels.
        Color correct predictions in blue and incorrect predictions in red.
        Shows a bar graph next to it with the weight of what it thought it was.

        Parameters:
        -predictions: array of predictions of the images
        -img_prediction: array of readable values of the predictions
        -img_labels: array of readable values of the actual value
        -images: array of images that were run through the model
        """
        num_rows = 3
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(predictions[i], img_prediction[i], img_labels[i], images[i])
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(predictions[i], img_labels[i])
        plt.tight_layout()


    def test_model(self):
        """
        Runs the model through some tests to show how well it does and how it is performing.
        """
        images, img_labels = self.data.get_next_labeled_batch()

        predictions = self.model.model.predict(images)
        base_pred = np.argmax(predictions, axis=1)
        #turning base_pred into matching the img_labels
        img_prediction = []
        for item in base_pred:
            img_prediction.append(self.data.label_to_class[item])
        #print(img_prediction)


        print('Classification Report: \n', classification_report(img_prediction, img_labels))
        #print('Confusion Matrix: \n', confusion_matrix(img_prediction, img_labels))

        
        #Creates a heatmap on the tests.
        plt.figure(figsize=(15, 15))
        sns.heatmap(confusion_matrix(img_labels, img_prediction), annot=True, xticklabels=self.data.label_info, yticklabels=self.data.label_info)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        #plt.show()

        #Shows a grid of pictures with what was predicted and how sure it is as a bar graph.
        self.gen_image_array(predictions, img_prediction, img_labels, images)
        plt.show()