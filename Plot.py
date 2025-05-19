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

    def plot_image(self, i, predictions_array, predicted_label, true_label, img):
        true_label, img, predicted_label = true_label[i], img[i], predicted_label[i]
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

    def plot_value_array(self, i, labels, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(len(predictions_array)), labels)
        plt.yticks([])
        thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color('red')
        #thisplot[true_label].set_color('blue')

    def gen_image_array(self, predictions, img_prediction, img_labels, images):
        # Plot the first X test images, their predicted labels, and the true labels.
        # Color correct predictions in blue and incorrect predictions in red.
        num_rows = 3
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(i, predictions[i], img_prediction, img_labels, images)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(i, self.data.label_info, predictions[i], img_labels)
        plt.tight_layout()


    def test_model(self):
        """

        """

        images, img_labels = self.data.get_next_labeled_batch()

        #print(img_labels)

        #Model.show_images(images, img_labels, labelInfo, 25)

        predictions = self.model.model.predict(images)
        base_pred = np.argmax(predictions, axis=1)
        #turning base_pred into matching the img_labels
        img_prediction = []
        for item in base_pred:
            img_prediction.append(self.data.label_to_class[item])
        #print(img_prediction)


        print('Classification Report: \n', classification_report(img_prediction, img_labels))
        #print('Confusion Matrix: \n', confusion_matrix(img_prediction, img_labels))

        
        plt.figure(figsize=(15, 15))
        sns.heatmap(confusion_matrix(img_labels, img_prediction), annot=True, xticklabels=self.data.label_info, yticklabels=self.data.label_info)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        #plt.show()

        self.gen_image_array(predictions, img_prediction, img_labels, images)
        fig = plt.gcf()
        #fig.show()
        plt.show()