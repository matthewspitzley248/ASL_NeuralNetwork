import cv2
import numpy as np
import screeninfo
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt

import Model

def plot_value_array(i, labels, predictions_array):
    plt.style.use('dark_background')
    plt.grid(False)
    plt.xticks(range(len(predictions_array)), labels)
    plt.yticks([])
    thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    plt.title("Predicted : {}\n{:2.0f}%".format(str.upper(prediction), (100 * np.max(prediction_nums[0]))))
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')

def graph():
    plot_value_array(0, labelInfo, prediction_nums[0])
    # redraw the canvas
    fig = plt.gcf()
    fig.canvas.draw()

    plt.clf()
    # convert canvas to image
    graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                          sep='')
    #print('-------------------'+str(fig.canvas.get_width_height()[::-1]+ (3,)))
    graph = graph.reshape(fig.canvas.get_width_height()[::-1]+ (3,))
    graph = cv2.resize(graph,(int(screen.width/2),int(screen.width/2)))
    return graph

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized


# Load the saved model
model = tf.keras.models.load_model('Models/Model29.keras')

screen_id = 0

# get the size of the screen
screen = screeninfo.get_monitors()[screen_id]
print(screeninfo.get_monitors())


# load model info
model.summary()
imgShape = (model.layers[0].input_shape[1], model.layers[0].input_shape[2], 3)
print(imgShape)
test_image_gen = Model.normalizeImages(1, 'test', True, imgShape, 0, 0, 0)
temp, labels = next(test_image_gen)
labelInfo = test_image_gen.class_indices
label_to_class = {v: k for k, v in labelInfo.items()}
img_labels = Model.array_to_labels(labels, label_to_class)

video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()

    # cropping image
    #frame = frame[0:height, 0:height]

    # Convert the captured frame into RGB
    im = Image.fromarray(frame, 'RGB')


    # Resizing image - makes it into a square then scales it down
    showim = crop_square(frame, int(screen.width/2))
    im = cv2.resize(showim, imgShape[:2])
    img = tf.reshape(im, [1, imgShape[0], imgShape[1], 3])
    # img_array = np.array(im)

    # Calling the predict method
    prediction_nums = model.predict(img)
    prediction = label_to_class[np.argmax(prediction_nums, axis=1)[0]]
    #print(prediction_nums[0])

    #graph info
    graphImg = graph()

    # Displays the predicted asl
    frame = cv2.rectangle(frame, (0, 0), (130, 35), (0, 0, 0), -1)
    frame = cv2.putText(frame, "{} {:2.0f}%".format(prediction, (100 * np.max(prediction_nums[0]))), (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    window_name = 'projector'
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    #cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #show webcam
    #cv2.imshow(window_name, graph)
    vis = np.concatenate((showim, graphImg), axis=1)
    cv2.imshow(window_name, vis)
    #make sure what is being passed into NN is same as being displayed
    cv2.imshow('im', im)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
