import pandas as pd 
import numpy as np
import tensorflow as tf
import requests
import json
from json import JSONEncoder

# Construct test data row with "unseen" feature values - report_params = 15 

headers = ['report_id', 'report_params', 'day_part']
dataframe_input = pd.DataFrame([[1, 15, 3]],
                                columns=headers, 
                                dtype=float,
                                index=['input'])

print("+++++ 1.")
print(dataframe_input.head())


eps=0.001 # 0 => 0.1¢
dataframe_input['report_params'] = np.log(dataframe_input.pop('report_params')+eps)

print("+++++ 2.")
print(dataframe_input.head())


dataframe_input['report_id'] = dataframe_input['report_id'].apply(str)
dataframe_input['day_part'] = dataframe_input['day_part'].apply(str)

print("+++++ 3.")
dataframe_input.head()

#input_ds = df_to_dataset(dataframe_input, shuffle=False)
input_ds = tf.data.Dataset.from_tensor_slices(dict(dataframe_input))
input_ds = input_ds.batch(1)

for feature_batch in input_ds.take(1):
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

print(input_ds)

# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)

# url='http://localhost:8080/ping'

# resp = requests.get(url)

# print('reponse text:'.format(resp)


# url='http://127.0.0.1:8080/metadata'

# resp = requests.get(url)

# print('reponse text:'.format(resp.text))
 
url='http://127.0.0.1:8080/invocations'

head = {'Content-type':'application/json',
             'Accept':'application/json'}
payload = {"instances": input_ds}

#payld = json.dumps(payload,cls=NumpyArrayEncoder)

resp = requests.post(url,data=payload,headers=head)

print('response text:'.format(resp))

predictions = json.loads(resp.text)['predictions']
print('predictions'.format(predictions))