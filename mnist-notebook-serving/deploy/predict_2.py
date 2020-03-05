import pandas as pd 
import numpy as np

import tensorflow as tf
import argparse, sys
import json

import PIL 
from PIL import Image

# Label mapping
labels = '''T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot'''.split("\n")


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default="")
  
  args, _ = parser.parse_known_args()
    
  data = args.data
  if (data == ""):
    print('Error: data argument is empty. ') 
    sys.exit(255)

  print("File name:", data)


  # from https://opensource.com/life/15/2/resize-images-python 
  # basewidth = 28 
  # baseheight = 28

  # img = Image.open(data)

  # print(img) 

  # wpercent = (basewidth / float(img.size[0]))
  # hsize = int( ( float(img.size[1]) * float(wpercent) ) )
  # img = img.resize( (basewidth, hsize), PIL.Image.ANTIALIAS)


  # From https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/

  image = Image.open(data)

  print("Image format: {}".format(image.format))
  print("Image mode: {}".format(image.mode))
  print("Image size: {}".format(image.size))

  img_resized = image.resize((28,28))
  #img_resized = np.array(img_resized, dtype='float32')
  print("Resized image: {}".format(img_resized.size))

  imagearr = np.asarray(img_resized)
  imagearr = imagearr / 255.0
  #imagearr = imagearr.reshape(28, 28)
  imagearr = imagearr.reshape(-1, 28, 28, 1)

  print("Image array shape: {}".format(imagearr.shape))
  #print("+++++++++++++++++++++")
  #print(imagearr)
  #print("+++++++++++++++++++++")
  # create Pillow image
  # image2 = Image.fromarray(imagearr)
  # print("Image2 format: {}".format(image2.format))
  # print("Image2 mode: {}".format(image2.mode))
  # print("Image2 size: {}".format(image2.size))

  model = tf.keras.models.load_model('model/1')
 
  res = model.predict(imagearr)
  
  print(res.shape)
  pred = res.argmax(axis=1)
  print(pred[0])
  predstr = labels[pred[0]]   
  print("PREDICTION:", predstr)