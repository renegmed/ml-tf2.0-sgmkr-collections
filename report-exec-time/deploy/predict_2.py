import pandas as pd 
import numpy as np
import tensorflow as tf
import argparse, sys
import json

def build_trainstat():
  # column_names = ['report_id','report_params','day_part','exec_time']
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

  # print(dataframe.head())
  # Splitting training dataset into train (80%) and test data

  train_dataset = dataframe.sample(frac=0.8,random_state=0) 
  #test_dataset = dataframe.drop(train_dataset.index)

  # Describe train dataset, without target feature - exec_time. Mean and std will be used to normalize training data

  train_stats = train_dataset.describe()
  train_stats.pop("exec_time")
  train_stats = train_stats.transpose()
  # print(train_stats)

  return train_stats


# Neural network learns better, when data is normalized (features look similar to each other)

def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default="")
  
  args, _ = parser.parse_known_args()
    
  data = args.data
  if (data == ""):
    print('Error: data argument is empty. ') 
    sys.exit(255)

  print(data)

  obj = json.loads(data)

  print("REport ID:", obj["report_id"])
  print("REport Params:", obj["report_params"])
  print("Day Part:", obj["day_part"])

  
  train_stats = build_trainstat()
   
  # Construct test data row with "unseen" feature values - report_params = 15 

  headers = ['report_id', 'report_params', 'day_part']
  dataset_input = pd.DataFrame([[obj["report_id"], obj["report_params"], obj["day_part"]]],
                                columns=headers, 
                                dtype=float,
                                index=['input'])

  # print("+++++ 1.")
  # print(dataset_input)

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

  # print("+++++ 2.")
  # dataset_input.tail() 

  normed_dataset_input = norm(dataset_input, train_stats)
  normed_dataset_input = np.array(normed_dataset_input)

  # print("+++++ 3. normed_dataset_input")
  # print(normed_dataset_input)
  

  model = tf.keras.models.load_model('model_report_exec_time/1583331696')
  #model = tf.keras.models.load_model('model/1')

  res = model.predict(normed_dataset_input)

  print("PREDICTION:", res[0][0])