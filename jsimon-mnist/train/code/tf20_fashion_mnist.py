import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model 
 
import argparse, os, subprocess, sys

# Script mode doesn't support requirements.txt
# Here's the workaround ;)
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# class FMNISTModel(keras.Model):
#     # Create the different layers used by the model
#     def __init__(self, dense_layer, dropout):
#         super(FMNISTModel, self).__init__(name='fmnist_model')
#         self.conv2d_1   = Conv2D(64, 3, padding='same', activation='relu',input_shape=(28,28))
#         self.conv2d_2   = Conv2D(64, 3, padding='same', activation='relu')
#         self.max_pool2d = MaxPooling2D((2, 2), padding='same')
#         self.batch_norm = BatchNormalization()
#         self.flatten    = Flatten()
#         self.dense1     = Dense(dense_layer, activation='relu')
#         self.dense2     = Dense(10)
#         self.dropout    = Dropout(dropout)
#         self.softmax    = Softmax()

#     # Chain the layers for forward propagation
#     def call(self, x):
#         # 1st convolution block
#         x = self.conv2d_1(x)
#         x = self.max_pool2d(x)
#         x = self.batch_norm(x)
#         # 2nd convolution block
#         x = self.conv2d_2(x)
#         x = self.max_pool2d(x)
#         x = self.batch_norm(x)
#         # Flatten and classify
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.dropout(x)
#         x = self.dense2(x)
#         return self.softmax(x)

    
if __name__ == '__main__':

    print("TensorFlow version", tf.__version__)
    #print("Keras version", keras.__version__)

    # Keras-metrics brings additional metrics: precision, recall, f1
    #install('keras-metrics')
    #import keras_metrics
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dense-layer', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    dense_layer = args.dense_layer
    dropout    = args.dropout
    
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    #print("++++++ model_dir:", model_dir)
    #training_dir   = args.training
    #print("+++++++ training_dir:", training_dir)
    #validation_dir = args.validation
    #print("+++++++ validation_dir:", validation_dir)
    
    # Load in the data
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print("x_train.shape:", x_train.shape)

         
    
    # Add extra dimension for channel: (28,28) --> (28, 28, 1)
    # the data is only 2D!
    # convolution expects height x width x color
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("++++ x_train.shape: ".format(x_train.shape))

    K = len(set(y_train))
    print("++++ number of classes:", K)
    
    # Build the model using the functional API
    i = Input(shape=x_train[0].shape)
    x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
    x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(K, activation='softmax')(x)
    
    model = Model(i, x)
    
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    # Compile and fit
    # Note: make sure you are using the GPU for this!
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)
    
    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1')) 
