#!/usr/bin/env python

"""
This script trains a convolutional neural network (CNN) with a LeNet architecture 
on impressionist paintings, with the aim of classifying the artist of a painting. 

Input:
  - -a, --artists: list of str, <list-of-artists>, e.g. Matisse VanGogh Gauguin (optional, default: ALL)
  - -train, --train_data: str, <path-to-training-data> (optional, default: ../data/impressionist_subset/training/)
  - -test, --test_data: str, <path-to-test-data> (optional, default: ../data/impressionist_subset/validation/)
  - -e, --epochs: int, <number-of-epochs> (optional, default: 10)
  - -b, --batch_size: int, <batch-size> (optional, default: 50)

Output (saved in out/)
  - model_summary.txt: summary of model architecture
  - model_plot.png: visualisation of model architecture
  - model_history.png: model training history
  - model_report.txt: classification report of model, also printed to command line 
"""

# LIBRARIES ------------------------------------

# Basics
import os
import sys
import argparse

# Utility functions
sys.path.append(os.path.join(".."))
from utils.cnn_utils import (get_min_dim, preprare_Xy,
                             save_model_info, save_model_history, save_model_report)
             

# CNN Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report


# MAIN FUNCTION ------------------------------------

def main():
    
    # --- ARUGMENT PARSER ---
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--artists", nargs="+", help = "List of artists to train model on",
                    required = False, default = ["ALL"])
    # Input option for path to training data
    ap.add_argument("-train", "--train_directory", type = str, help = "Path to the training data directory",
                    required = False, default = "../data/impressionist/training/")
    # Input option for path to test data
    ap.add_argument("-test", "--test_directory", type = str, help = "Path to the test data directory",
                    required = False, default = "../data/impressionist/validation/")
    # Input option option for number of epoch
    ap.add_argument("-e", "--epochs", type = int, help = "The number of epochs to train the model",
                    required = False, default = 10)
    # Input option for batch size
    ap.add_argument("-b", "--batch_size", type = int, help = "Size of batch of images to train the model",
                    required = False, default = 50)
    # Parse arguments
    args = vars(ap.parse_args())
    artists = args["artists"]
    train_directory = args["train_directory"]
    test_directory = args["test_directory"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
        
    # --- PREPARE DATA ---
    
    # Get names of artists to load data, if ALL, use all names, otherwise only those specified
    if artists == ["ALL"]:
        label_names = os.listdir(os.path.join(train_directory))
    else:
        label_names = artists
    
    # Print message
    print(f"\n[INFO] Initialising classifcation of paintings from {label_names}.")
    
    # Get minimum dimension of smallest image, used to resize all images accordingly
    img_dim = get_min_dim(train_directory, test_directory, label_names)
    
    # Print message
    print(f"\n[INFO] Preparing data: all images will be resized to {img_dim}x{img_dim}.")
    
    # Prepare data, returns resized, normalised array of images, and binarised labels
    X_train, y_train = preprare_Xy(train_directory, img_dim, label_names)
    X_test, y_test = preprare_Xy(test_directory, img_dim, label_names)
   
    # --- CNN WITH LENET ARCHITECTURE ---
    
    # Print message
    print(f"\n[INFO] Training CNN with LeNet Architecture with {epochs} epochs and a batch size of {batch_size}.")
    
    # Get number of classes
    n_labels = len(label_names)
    
    # Define CNN: LeNet architecture
    model = Sequential()

    # Set 1: Conv2D-Activation-MaxPool
    model.add(Conv2D(32, (3, 3), padding = "same", input_shape = (img_dim, img_dim, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

    # Set 2: Con2D-Activation-MaxPool
    model.add(Conv2D(50, (5, 5), padding = "same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

    # Set 3: Fully connected layer, with relu activation
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # Set 4: Output layer, with softmax classification, to predict 10 classes (10 artists)
    model.add(Dense(n_labels))
    model.add(Activation("softmax"))

    # Compile CNN: using categorigical corrsentropy and stochastic gradient descent
    model.compile(loss = "categorical_crossentropy", 
                  optimizer = SGD(lr=0.01), metrics = ["accuracy"])
    
    # Train CNN: use training data to learn weights, using defined batch_size and epochs
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), 
                        batch_size = batch_size, epochs = epochs, verbose = 1)
    
    # Evaluate CNN: generate predictions and compare to true labels
    predictions = model.predict(X_test, batch_size)
    report = classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names = label_names)
    
    # --- OUTPUT ---
    
    # Create output directory
    output_directory = os.path.join("..", "out")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    # Save model summary, model history and classification report
    save_model_info(model, output_directory, "model_summary.txt", "model_plot.png")
    save_model_history(history, epochs, output_directory, "model_history.png")
    save_model_report(report, epochs, batch_size, output_directory, "model_report.txt")
    
    # Print classification report
    print(f"Classification report:\n {report}")
    
    # Print message
    print(f"\n[INFO] All done!")
     
    
if __name__=="__main__":
    main()