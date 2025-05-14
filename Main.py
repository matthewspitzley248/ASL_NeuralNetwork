import os

from matplotlib import pyplot as plt

import Model
import ModelTest
#Model.seperateImages()
Model.GenModel('Model28.keras', 50, 30)

#Model.infoOnDataset()

#ModelTest.TestModel('Model27.keras', 200, 20, [0.7, 1.3], 0.2)


