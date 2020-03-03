 
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import argparse, os, subprocess, sys

# Script mode doesn't support requirements.txt
# Here's the workaround ;)
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

    
if __name__ == '__main__':

    print("TensorFlow version", tf.__version__)
    #print("Keras version", keras.__version__)

    # Keras-metrics brings additional metrics: precision, recall, f1
    #install('keras-metrics')
    #import keras_metrics
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--learning-rate', type=float, default=0.01)
#     parser.add_argument('--batch-size', type=int, default=128)
#     parser.add_argument('--dense-layer', type=int, default=512)
#     parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
#     parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    parser.add_argument('--input-file', type=str, default=os.environ['SM_INPUT_FILE'])
        
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
#     lr         = args.learning_rate
#     batch_size = args.batch_size
#     dense_layer = args.dense_layer
#     dropout    = args.dropout
    
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    input_file  = args.input_file
     
    print("+++++++ model_dir:", model_dir)
    print("+++++++ input_file:", input_file)
    

    # read training data
    # report_id - ID to identify report
    # report_params - number of parameters to execute report (when more params specified - report will be generated faster)
    # day_part - when report is executed (morning, midday or afternoon) - there is less load in the morning and in the afternoon reports are generated slower
    # exec_time - time spent to produce report

    column_names = ['report_id','report_params','day_part','exec_time']
    raw_dataframe = pd.read_csv(input_file)
    dataframe = raw_dataframe.copy()

    #dataframe.head()

    eps=0.001 # 0 => 0.1¢
    dataframe['report_params'] = np.log(dataframe.pop('report_params')+eps)

    # dataframe.head()

    dataframe['report_id'] = dataframe['report_id'].apply(str)
    dataframe['day_part'] = dataframe['day_part'].apply(str)

    # dataframe.head()
 
    
    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print("++++++",len(train), 'train examples')
    print("++++++",len(val), 'validation examples')
    print("++++++",len(test), 'test examples')

    # A utility method to create a tf.data dataset from a Pandas Dataframe
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('exec_time')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    feature_columns = []

    feature_columns.append(feature_column.numeric_column('report_params'))

    report_id = feature_column.categorical_column_with_vocabulary_list('report_id', ['1', '2', '3', '4', '5'])
    report_id_one_hot = feature_column.indicator_column(report_id)
    feature_columns.append(report_id_one_hot)

    day_part = feature_column.categorical_column_with_vocabulary_list('day_part', ['1', '2', '3'])
    day_part_one_hot = feature_column.indicator_column(day_part)
    feature_columns.append(day_part_one_hot)

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    
    # Construct neural network with Keras API on top of TensorFlow. SGD optimizer and 
    # mean squared error loss to check training quality

    def build_model(feature_layer):
        model = keras.Sequential([
            feature_layer,
            layers.Dense(16, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
   
        return model

    model = build_model(feature_layer)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs,
              callbacks=[early_stop])

    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1')) 
    
    print(".... TRAINING COMPLETED ....")
