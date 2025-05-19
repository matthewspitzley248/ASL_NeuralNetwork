import cv2
import numpy as np
import screeninfo
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt


class Camera:
    """
    Handles the setup and running of the camera for the live use of the model.
    """

    def __init__(self, data, model, plot, screen_id=0):
        """

        """
        self.data = data
        self.model = model
        self.plot = plot
        self.screen_id = screen_id
        self.screen = screeninfo.get_monitors()[screen_id]

    def run(self):
        """
        contains the main run loop of the camera functionality to run the model through the camera.
        """
        video = cv2.VideoCapture(0)

        while True:

            #break out statement, when q is pressed quit out.
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            #gets the frame from the camera to predict on
            _, frame = video.read()

            # Resizing image - makes it into a square then scales it down
            display_image, resized_img, model_input_img = self.resize(frame)

            # Calling the predict method
            prediction_nums = self.model.model.predict(model_input_img)
            prediction = self.data.label_to_class[np.argmax(prediction_nums, axis=1)[0]]
            #print(prediction_nums[0])

            #graph info
            graphImg = self.graph(prediction_nums)

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
            vis = np.concatenate((display_image, graphImg), axis=1)
            cv2.imshow(window_name, vis)
            #make sure what is being passed into NN is same as being displayed
            cv2.imshow('resized_img', resized_img)
        video.release()
        cv2.destroyAllWindows()




    def graph(self, prediction_nums):
        """
        Uses the plot_value_array and turns it into an image to display on the screen
        """
        self.plot.plot_value_array(prediction_nums[0])
        # redraw the canvas
        fig = plt.gcf()
        fig.canvas.draw()

        plt.clf()
        # convert canvas to image
        graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                              sep='')
        #print('-------------------'+str(fig.canvas.get_width_height()[::-1]+ (3,)))
        graph = graph.reshape(fig.canvas.get_width_height()[::-1]+ (3,))
        graph = cv2.resize(graph,(int(self.screen.width/2),int(self.screen.width/2)))

        return graph


    def resize(self,frame):
        """
        """
        #the camera image cropped to a square
        display_image = self.crop_square(frame)
        #the image resized to the shape the model is trained on
        resized_img = cv2.resize(display_image, self.data.img_shape[:2])
        #the resized image turned into a tensor for the model to predict off of
        model_input_img = tf.reshape(resized_img, [1, self.data.img_shape[0], self.data.img_shape[1], 3])

        return display_image, resized_img, model_input_img


    def crop_square(self, img, interpolation=cv2.INTER_AREA):
        """
        """
        h, w = img.shape[:2]
        min_size = np.amin([h,w])
        size = int(self.screen.width/2)

        # Centralize and crop
        crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
        resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

        return resized
