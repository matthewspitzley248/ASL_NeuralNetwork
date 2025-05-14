import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import Model
import tensorflow as tf

def plot_image(i, predictions_array, predicted_label, true_label, img):
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

def plot_value_array(i, labels, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(len(predictions_array)), labels)
  plt.yticks([])
  thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  #thisplot[true_label].set_color('blue')

def gen_image_array(predictions, img_prediction, img_labels, images, labelInfo):
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 3
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], img_prediction, img_labels, images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, labelInfo, predictions[i], img_labels)
    plt.tight_layout()

def TestModel(modelName, batchSize, rotation, zoom, shift):

    model = tf.keras.models.load_model('Models/'+modelName)
    model.summary()


    #print(model.layers[0].input_shape[1])
    imgShape = (model.layers[0].input_shape[1],model.layers[0].input_shape[2],3)

    test_image_gen = Model.normalizeImages(batchSize, 'test', True, imgShape,  rotation, zoom, shift)

    images, labels = next(test_image_gen)
    labelInfo = test_image_gen.class_indices
    label_to_class = {v: k for k, v in labelInfo.items()}
    img_labels = Model.array_to_labels(labels, label_to_class)

    #print(img_labels)

    #Model.show_images(images, img_labels, labelInfo, 25)

    predictions = model.predict(images)
    base_pred = np.argmax(predictions, axis=1)
    #turning base_pred into matching the img_labels
    img_prediction = []
    for item in base_pred:
        img_prediction.append(label_to_class[item])
    #print(img_prediction)


    print('Classification Report: \n', classification_report(img_prediction, img_labels))
    #print('Confusion Matrix: \n', confusion_matrix(img_prediction, img_labels))

    import seaborn as sns
    plt.figure(figsize=(15, 15))
    sns.heatmap(confusion_matrix(img_labels, img_prediction), annot=True, xticklabels=labelInfo, yticklabels=labelInfo)
    #plt.show()

    gen_image_array(predictions,img_prediction,img_labels,images,labelInfo)
    fig = plt.gcf()
    #fig.show()
    plt.show()
