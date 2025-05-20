
from Model import Model
from Data import Data
from Plot import Plot
from Camera import Camera

img_shape = (300, 300, 3)


data = Data(batch_size=100, img_shape=img_shape)#, rotation=20, zoom_range=[0.65, 1.35], shift_range=0.2, seperate_data=False)

model = Model(data=data, model_name='Model30.keras')

#model.fit_Model(epochs=50)

plot = Plot(data=data, model=model)

#plot.plot_training_history(model.model_history)
#plot.test_model()


camera = Camera(data, model, plot)

camera.run()


