import requests
import json
import logging
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

print(tf.__version__)

# sample_image = tf.keras.preprocessing.image.load_img(r'images/cat.282.jpg', target_size = (224, 224))

# sample_image = tf.keras.preprocessing.image.img_to_array(sample_image)

# sample_image = np.expand_dims(sample_image, axis = 0)

# sample_image = tf.keras.applications.resnet50.preprocess_input(sample_image)

# #print(sample_image)


# url='http://localhost:8080/invocations'

# head = {'Content-type':'application/json',
#              'Accept':'application/json'}
# payload = {'instances': sample_image}

# #payld = json.dumps(payload)

# resp = requests.post(url,data=payload,headers=head)

# print('reponse text:'.format(resp.text))

# predictions = json.loads(resp.text)['predictions']
# print('predictions'.format(predictions))


# #pingurl = 'http://localhost:8080/ping'

# #ret = requests.post(pingurl)
# print("Status code: {}".format(resp.status_code))

