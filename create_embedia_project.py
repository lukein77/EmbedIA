from tensorflow.keras.models import load_model
from sklearn.datasets import load_digits

from embedia.project_options import *
from embedia.project_generator import ProjectGenerator
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from embedia.project_options import *
from embedia.project_generator import ProjectGenerator

from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import larq as lq

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

############# Settings to create the project #############

                    
OUTPUT_FOLDER = 'outputs/'
PROJECT_NAME  = 'person_detectionnnnnnnnnnnn__'
MODEL_FILE    = 'models/cifar10_binarias_model.h5'

#con larq
MODEL_FILE_BINARY = 'models/person_detection_with_coco_XNORNET_threshold05.h5'


model = load_model(MODEL_FILE)
model_binary = load_model(MODEL_FILE_BINARY)    


num_classes = 10

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")
test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")



# Normalize pixel values to be between -1 and 1
train_images2, test_images2 = train_images / 127.5 - 1, test_images / 127.5 - 1

'''
from PIL import Image

img = Image.open("R.jpg")
imgArray = np.array(img)
imga = np.array(tf.image.resize(imgArray, [96,96]))

imagenn = np.zeros((1,96,96,1))
asdasd = tf.image.rgb_to_grayscale(imga)
imagenn[0] = asdasd
# Normalize pixel values to be between -1 and 1
imagenn [0] = imagenn [0] / 127.5 - 1

#plt.imshow(imagenn [0])





example_number = 1003
#sample = test_images2[example_number]






_fila,_col,_can = imagenn [0].shape
arr = np.zeros((_can,_fila,_col),dtype="float32")
for fila,elem in enumerate(imagenn [0]):
  for columna,elem2 in enumerate(elem):
    for canal,valor in enumerate(elem2):
        
        arr[canal,fila,columna] = valor   
'''
ll = np.zeros(96*96)
sample = ll
#sample =arr
comment= "clase %d example for test" % 1
#ress = model_binary.predict(imagenn)
#print(np.argmax(ress))



options = ProjectOptions()

options.project_type = ProjectType.ARDUINO
# options.project_type = ProjectType.C
#options.project_type = ProjectType.CODEBLOCK
# options.project_type = ProjectType.CPP

#options.data_type = ModelDataType.FLOAT
# options.data_type = ModelDataType.FIXED32
# options.data_type = ModelDataType.FIXED16
#options.data_type = ModelDataType.FIXED8
options.data_type = ModelDataType.BINARY

options.debug_mode = DebugMode.DISCARD
#options.debug_mode = DebugMode.DISABLED
#options.debug_mode = DebugMode.HEADERS
#options.debug_mode = DebugMode.DATA

#options.tamano_bloque = BinaryBlockSize.Bits8
#options.tamano_bloque = BinaryBlockSize.Bits16
options.tamano_bloque = BinaryBlockSize.Bits32


options.example_data=sample
options.example_comment=comment

options.files = ProjectFiles.ALL
# options.files = {ProjectFiles.MAIN}
# options.files = {ProjectFiles.MODEL}
# options.files = {ProjectFiles.LIBRARY}

generator = ProjectGenerator()
generator.create_project(OUTPUT_FOLDER, PROJECT_NAME, model, model_binary, options)

print("Project", PROJECT_NAME, "exported in", OUTPUT_FOLDER)
print("\n"+comment)






