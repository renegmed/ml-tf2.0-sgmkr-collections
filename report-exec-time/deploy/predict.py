import pandas as pd 
import numpy as np
import tensorflow as tf
import requests
import json
from json import JSONEncoder

import urllib.request
import urllib.parse


column_names = ['report_id','report_params','day_part','exec_time']
raw_dataframe = pd.read_csv('../train/local-test/test_dir/input/report_exec_times.csv')
dataframe = raw_dataframe.copy()

print(dataframe.head())

# report_id and day_part are categorical features. This means we need to encode these two attributes

report_id = dataframe.pop('report_id')
day_part = dataframe.pop('day_part')

# Encoding categorical attributes (creating as many columns as there are unique values and assigning 1 for the column from current row value)

dataframe['report_1'] = (report_id == 1)*1.0
dataframe['report_2'] = (report_id == 2)*1.0
dataframe['report_3'] = (report_id == 3)*1.0
dataframe['report_4'] = (report_id == 4)*1.0
dataframe['report_5'] = (report_id == 5)*1.0

dataframe['day_morning'] = (day_part == 1)*1.0
dataframe['day_midday'] = (day_part == 2)*1.0
dataframe['day_afternoon'] = (day_part == 3)*1.0

print(dataframe.head())
# Splitting training dataset into train (80%) and test data

train_dataset = dataframe.sample(frac=0.8,random_state=0)
test_dataset = dataframe.drop(train_dataset.index)

# Describe train dataset, without target feature - exec_time. Mean and std will be used to normalize training data

train_stats = train_dataset.describe()
train_stats.pop("exec_time")
train_stats = train_stats.transpose()
print(train_stats)

# Neural network learns better, when data is normalized (features look similar to each other)

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']




# Construct test data row with "unseen" feature values - report_params = 15 

headers = ['report_id', 'report_params', 'day_part']
dataset_input = pd.DataFrame([[1, 15, 3]],
                                columns=headers, 
                                dtype=float,
                                index=['input'])

print("+++++ 1.")
print(dataset_input)

# Encode categorical features for test data row

report_id = dataset_input.pop('report_id')
day_part = dataset_input.pop('day_part')

dataset_input['report_1'] = (report_id == 1)*1.0
dataset_input['report_2'] = (report_id == 2)*1.0
dataset_input['report_3'] = (report_id == 3)*1.0
dataset_input['report_4'] = (report_id == 4)*1.0
dataset_input['report_5'] = (report_id == 5)*1.0

dataset_input['day_morning'] = (day_part == 1)*1.0
dataset_input['day_midday'] = (day_part == 2)*1.0
dataset_input['day_afternoon'] = (day_part == 3)*1.0

print("+++++ 2.")
dataset_input.tail()

 

normed_dataset_input = norm(dataset_input)
normed_dataset_input = np.array(normed_dataset_input)

print("+++++ 3. normed_dataset_input")
print(normed_dataset_input)

 
# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)

# url='http://localhost:8080/ping'

# resp = requests.get(url)

# print('reponse text:'.format(resp))




 
# f = urllib.request.urlopen(url)
# print(f.read().decode('utf-8'))




# url='http://127.0.0.1:8080/metadata'

# resp = requests.get(url)

# print('reponse text:'.format(resp.text))



# url='http://localhost:8080/invocations'

# head = {'Content-type':'application/json',
#              'Accept':'application/json'}

# payload = {"instances": normed_dataset_input }
# #payload = {"instances": [[ 3.19179609, 2.05277296, -0.51536518, -0.4880486, -0.50239337, -0.50629114, -0.74968743, -0.68702182, 1.45992522]] }

# #payld = json.dumps(payload,cls=NumpyArrayEncoder)

# resp = requests.post(url,data=payload,headers=head)

# print('response text:'.format(resp))

# predictions = json.loads(resp.text)['predictions']
# print('predictions'.format(predictions))



# f = urllib.request.urlopen(url)
# print(f.read().decode('utf-8'))

#model = tf.keras.models.load_model('model_report_exec_time/1583331696')
model = tf.keras.models.load_model('model/1')

res = model.predict(normed_dataset_input)

print("PREDICTION:", res)