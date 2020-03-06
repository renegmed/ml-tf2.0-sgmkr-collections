import pandas as pd 
import numpy as np
import tensorflow as tf
import argparse, sys
import json
import requests

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

  # parser = argparse.ArgumentParser()
  # parser.add_argument('--data', type=str, default="")
  
  # args, _ = parser.parse_known_args()
    
  # data = args.data
  # if (data == ""):
  #   print('Error: data argument is empty. ') 
  #   sys.exit(255)

  # print("File name:", data)
 

 # Load in the data
  fashion_mnist = tf.keras.datasets.fashion_mnist

  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  print("x_train.shape:", x_train.shape)
  print("x_test.shape:", x_test.shape)

  # the data is only 2D!
  # convolution expects height x width x color
  x_train = np.expand_dims(x_train, -1)
  x_test = np.expand_dims(x_test, -1)
  print(x_train.shape)

  start_index = 100
  last_item_index = start_index + 5

  data = json.dumps({"signature_name": "serving_default", "instances": x_test[start_index:last_item_index].tolist()})
  #print(data)

  headers = {"content-type": "application/json"}
  #r = requests.post('http://localhost:8501/v1/models/mnist_notebook:predict', data=data, headers=headers)
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
  print("Predicted labels: {}".format(pred))

  # Get the true labels
  actual = [labels[i] for i in y_test[start_index:last_item_index]]
  print("Actual labels: {}".format(actual))

 