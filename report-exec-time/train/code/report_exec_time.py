 
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

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

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    parser.add_argument('--input-file', type=str, default=os.environ['SM_INPUT_FILE'])
        
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    
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

    #dataframe.head()

    # Splitting training dataset into train (80%) and test data

    train_dataset = dataframe.sample(frac=0.8,random_state=0)
    test_dataset = dataframe.drop(train_dataset.index)

    # train_dataset.shape
    # test_dataset.shape
 
    # Describe train dataset, without target feature - exec_time. Mean and std will be used to normalize training data

    train_stats = train_dataset.describe()
    train_stats.pop("exec_time")
    train_stats = train_stats.transpose()
    # train_stats

    # Remove exec_time feature from training data and keep it as a target for both training and testing

    train_labels = train_dataset.pop('exec_time')
    test_labels = test_dataset.pop('exec_time')

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Neural network learns better, when data is normalized (features look similar to each other)

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    normed_train_data = np.array(normed_train_data)
    normed_test_data = np.array(normed_test_data)

    # Construct neural network with Keras API on top of TensorFlow. SGD optimizer and 
    # mean squared error loss to check training quality

    def build_model():
        model = keras.Sequential([
            layers.Dense(16, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
        return model

    model = build_model()
    #model.summary()
    
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=epochs,
                    validation_split=0.2, batch_size=40, verbose=1, callbacks=[early_stop])

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

    print("Testing set Mean Abs Error: {:5.2f} Report Execution Time".format(mae))


    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1')) 
    
    print(".... TRAINING COMPLETED ....")
