 
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.models import Model

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
    
    # Load in the data
#     df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
    df = pd.read_csv(input_file, encoding='ISO-8859-1')
    
    # drop unnecessary columns
    df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

    # rename columns to something better
    df.columns = ['labels', 'data']
    
    # create binary labels
    df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
    Y = df['b_labels'].values
    
    # split up the data
    df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size=0.33)
    
    # Convert sentences to sequences
    MAX_VOCAB_SIZE = 20000
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(df_train)
    sequences_train = tokenizer.texts_to_sequences(df_train)
    sequences_test = tokenizer.texts_to_sequences(df_test)
    
    # get word -> integer mapping
    word2idx = tokenizer.word_index
    V = len(word2idx)
    print('+++++ Found %s unique tokens.' % V)
    
    # pad sequences so that we get a N x T matrix
    data_train = pad_sequences(sequences_train)
    print('+++++ Shape of data train tensor:', data_train.shape)

    # get sequence length
    T = data_train.shape[1]
    
    data_test = pad_sequences(sequences_test, maxlen=T)
    print('+++++ Shape of data test tensor:', data_test.shape)

    
    # Create the model

    # We get to choose embedding dimensionality
    D = 20

    # Hidden state dimensionality
    M = 15

    # Note: we actually want to the size of the embedding to (V + 1) x D,
    # because the first index starts from 1 and not 0.
    # Thus, if the final index of the embedding matrix is V,
    # then it actually must have size V + 1.

    i = Input(shape=(T,))
    x = Embedding(V + 1, D)(i)
    x = LSTM(M, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(i, x)
    if gpu_count > 1:
       model = multi_gpu_model(model, gpus=gpu_count)
        
    # Compile and fit
    model.compile(
      loss='binary_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
    )
        
    print('Training model...epochs:', epochs)
    
    r = model.fit(
      data_train,
      Ytrain,
      epochs=epochs,
      validation_data=(data_test, Ytest)
    )

    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1')) 
    
    print(".... TRAINING COMPLETED ....")
