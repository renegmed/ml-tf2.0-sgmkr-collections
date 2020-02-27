import tensorflow as tf
from tensorflow import keras

import numpy as np
import argparse, os, subprocess, sys


# Script mode doesn't support requirements.txt
# Here's the workaround ;)
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
if __name__ == '__main__':

    print("TensorFlow version", tf.__version__)
    print("Keras version", keras.__version__)

    # Keras-metrics brings additional metrics: precision, recall, f1
    install('keras-metrics')
    import keras_metrics
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_DATA_DIR'])
    parser.add_argument('--train-steps', type=int, default=100)
    
    args, _ = parser.parse_known_args()
    
    gpu_count = args.gpu_count
    model_dir  = args.model_dir
    print("++++++ model_dir:", model_dir)
    data_dir   = args.data_dir
    print("+++++++ data_dir:", data_dir)
    train_steps = args.train_steps
    print("+++++++ train steps:", train_steps)
    



    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top = False)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
    x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
    x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    preds = tf.keras.layers.Dense(2, activation = 'softmax')(x)

    model = tf.keras.models.Model(inputs = base_model.input, outputs = preds)

    # the original resnet layers are excluded in the training
    for layer in model.layers[:175]:
        layer.trainable = False

    # the newly added layers are in the training
    for layer in model.layers[175:]:
        layer.trainable = True

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

    train_generator = train_datagen.flow_from_directory(data_dir,
                                                    target_size = (224, 224),
                                                    color_mode = 'rgb',
                                                    batch_size = 32,
                                                    class_mode = 'categorical',
                                                    shuffle = True)



    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    history = model.fit_generator(generator = train_generator, steps_per_epoch=train_generator.n//train_generator.batch_size, epochs=5)

    
    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1'))

    print("++++ Training complete ++++")

