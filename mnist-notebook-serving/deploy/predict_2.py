import pandas as pd 
import numpy as np

import tensorflow as tf
import argparse, sys
import json

# import PIL 
# from PIL import Image

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

  print("Image format: {}".format(image.format))
  print("Image mode: {}".format(image.mode))
  print("Image size: {}".format(image.size))
 

  imagearr = np.asarray(image)
  imagearr = imagearr / 255.0
  
  imagearr = imagearr.reshape(-1, 28, 28, 1)
 

  model = tf.keras.models.load_model('model/1')
 
  res = model.predict(imagearr)
  
  print(res.shape)
  pred = res.argmax(axis=1)
  print(pred[0])
  predstr = labels[pred[0]]   
  print("PREDICTION:", predstr)