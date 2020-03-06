import pandas as pd 
import numpy as np
import tensorflow as tf
import argparse, sys
import json
import requests

import PIL 
from PIL import Image

import cv2

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


class NumpyArrayEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default="")
  
  args, _ = parser.parse_known_args()
    
  image_file = args.data
  if (image_file == ""):
    print('Error: image_fileargument is empty. ') 
    sys.exit(255)

  print("File name:", image_file)
  
  image = cv2.imread(image_file)
  print(image.shape)
 
  print("Image size: {}".format(image.size)) 
  
  # transform colored image to grayscale
  image = np.mean(image, axis=2)
  print("Shape after grayscale conversion: {}".format(image.shape))
  
  image = cv2.resize(image, (28,28) )
  print("Resized Image size: {}".format(image.size))
  print(image.shape)

  imagearr = np.asarray(image)
  imagearr = imagearr / 255.0
   
  imagearr = imagearr.reshape(-1, 28, 28, 1)

   
  data = json.dumps({"signature_name": "serving_default", "instances": imagearr }, cls=NumpyArrayEncoder)
  #print(data)

  headers = {"content-type": "application/json"} 
  r = requests.post('http://localhost:8080/invocations', data=data, headers=headers)
  j = r.json()
  #print(j.keys())
  #print(j)

  # It looks like a 2-D array, let's check its shape
  pred = np.array(j['predictions'])
  #print(pred.shape)

  # This is the N x K output array from the model
  # pred[n,k] is the probability that we believe the nth sample belongs to the kth class

  # Get the predicted classes
  pred = pred.argmax(axis=1)

  # Map them back to strings
  pred = [labels[i] for i in pred]
  print("Predicted label: {}".format(pred)) 