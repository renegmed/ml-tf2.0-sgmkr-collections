import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.utils import shuffle
import argparse, os, subprocess, sys

# Script mode doesn't support requirements.txt
# Here's the workaround ;)
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

if __name__ == '__main__':

    print("TensorFlow version", tf.__version__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1024)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--input-file', type=str, default=os.environ['SM_INPUT_FILE'])

    args, _ = parser.parse_known_args()

    epochs     = args.epochs
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    input_file  = args.input_file
    print("++++++ epochs:", epochs)
    print("++++++ batch_size:", batch_size)
    print("++++++ model_dir:", model_dir)
    print("+++++++ input_file:", input_file)
    print("+++++++ gpu_count:", gpu_count)

    df = pd.read_csv(input_file)

    # We can't trust the userId and movieId to be numbered 0...N-1
    # Let's just set our own ids

    # current_user_id = 0
    # custom_user_map = {} # old user id > new user id
    # def map_user_id(row):
    #   global current_user_id, custom_user_map
    #   old_user_id = row['userId']
    #   if old_user_id not in custom_user_map:
    #     custom_user_map[old_user_id] = current_user_id
    #     current_user_id += 1
    #   return custom_user_map[old_user_id]

    # df['new_user_id'] = df.apply(map_user_id, axis=1)

    df.userId = pd.Categorical(df.userId)
    df['new_user_id'] = df.userId.cat.codes

    # Now do the same thing for movie ids
    # current_movie_id = 0
    # custom_movie_map = {} # old movie id > new movie id
    # def map_movie_id(row):
    #   global current_movie_id, custom_movie_map
    #   old_movie_id = row['movieId']
    #   if old_movie_id not in custom_movie_map:
    #     custom_movie_map[old_movie_id] = current_movie_id
    #     current_movie_id += 1
    #   return custom_movie_map[old_movie_id]

    # df['new_movie_id'] = df.apply(map_movie_id, axis=1)

    df.movieId = pd.Categorical(df.movieId)
    df['new_movie_id'] = df.movieId.cat.codes


    # Get user IDs, movie IDs, and ratings as separate arrays
    user_ids = df['new_user_id'].values
    movie_ids = df['new_movie_id'].values
    ratings = df['rating'].values

    # Get number of users and number of movies
    N = len(set(user_ids))
    M = len(set(movie_ids))

    # Set embedding dimension
    K = 10


    # Make a neural network

    # User input
    u = Input(shape=(1,))

    # Movie input
    m = Input(shape=(1,))

    # User embedding
    u_emb = Embedding(N, K)(u) # output is (num_samples, 1, K)

    # Movie embedding
    m_emb = Embedding(M, K)(m) # output is (num_samples, 1, K)

    # Flatten both embeddings
    u_emb = Flatten()(u_emb) # now it's (num_samples, K)
    m_emb = Flatten()(m_emb) # now it's (num_samples, K)

    # Concatenate user-movie embeddings into a feature vector
    x = Concatenate()([u_emb, m_emb]) # now it's (num_samples, 2K)

    # Now that we have a feature vector, it's just a regular ANN
    x = Dense(1024, activation='relu')(x)
    # x = Dense(400, activation='relu')(x)
    # x = Dense(400, activation='relu')(x)
    x = Dense(1)(x)


    # Build the model and compile
    model = Model(inputs=[u, m], outputs=x)
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)

    model.compile(
      loss='mse',
      optimizer=SGD(lr=0.08, momentum=0.9),
    )

    # split the data
    user_ids, movie_ids, ratings = shuffle(user_ids, movie_ids, ratings)
    Ntrain = int(0.8 * len(ratings))
    train_user = user_ids[:Ntrain]
    train_movie = movie_ids[:Ntrain]
    train_ratings = ratings[:Ntrain]

    test_user = user_ids[Ntrain:]
    test_movie = movie_ids[Ntrain:]
    test_ratings = ratings[Ntrain:]

    # center the ratings
    avg_rating = train_ratings.mean()
    train_ratings = train_ratings - avg_rating
    test_ratings = test_ratings - avg_rating

    r = model.fit(
        x=[train_user, train_movie],
        y=train_ratings,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0, # goes a little faster when you don't print the progress bar
        validation_data=([test_user, test_movie], test_ratings),
    )

    # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir, '1'))

    print("Training Completed")
