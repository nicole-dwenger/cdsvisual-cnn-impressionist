#!/usr/bin/env python

"""
Utility script with functions used in cnn_impressionist.py script 

DATA PREPARATION
  - get_min_dim(): get minimum dimension of images in train and test directory
  - preprocess_data(): preprocess data in train and test directory to retrieve images (X) and labels (y)

MODEL HELPER FUNCTIONS
  - save_model_summary(): save model summary and visualisation of model summary
  - save_model_history(): save plot of training history
  - save_model_report(): save file of classifcation report
"""

# DEPENDENCIES ------------------------------------

# Basics
import os
import sys
import glob
from tqdm import tqdm
from contextlib import redirect_stdout

# Data, Images and plots
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Tensorflow tools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide warnings
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K


# MODEL HELPER FUNCTIONS ------------------------------------

def get_min_dim(train_directory, test_directory, names):
    """Getting dimension of the smallest image in test and train data
    
    Input:
    - train_dir, test_dir: paths to training and test directory, containing subdirectories with images
    - names: names of subdirectories in train_dir and test_dir, here: refers to artist names
    
    Returns:
    - Minimum dimension of smallest image
    """
    # Create empty lists for dimensions
    dimensions = []
    
    # Walk through directories 
    for name in names:
        
        img_paths = (glob.glob(os.path.join(train_directory, name, "*.jpg")) +
                     glob.glob(os.path.join(test_directory, name, "*.jpg")))
        
        for path in img_paths:
            img = Image.open(path)
            dimensions.extend(img.size)                               
    
    return min(dimensions)


def preprocess_data(directory, img_dim, names):
    """Create array of resized and normalised images (X) and binarised labels (y) 
    
    Input: 
    - directory: directory with subdirectories called names, containing images 
    - img_dim: target size of images 
    - names: names of subdirectories in directory, here: refers to artist names
    
    Returns:
    - X: array of resized, normalised images, y: binarised labels
    """
    # Create empty array for images, with dimensions to which all images will be resized and 3 color channels
    X = np.empty((0, img_dim, img_dim, 3))
    # Create empty list for corresponding labels
    y = []
    
    # For each label name (artist)
    for name in names:
        # Get the paths of all images 
        img_paths = glob.glob(os.path.join(directory, name, "*.jpg"))
        
        # For each image for the given artist, load the image and append image array and label
        for img_path in tqdm(img_paths):
            img = load_img(img_path, target_size=(img_dim,img_dim))
            img_array = np.array([img_to_array(img)])
            X = np.vstack([X, img_array])
            y.append(name)

    # Normalize images
    X_norm = (X.astype("float") / 255.)
    
    # Binarize labels
    lb = LabelBinarizer()
    y_binary = lb.fit_transform(y)
   
    return X_norm, y_binary


def save_model_summary(model, output_directory):
    """Save model summary in .txt file and plot of model in .png
    
    Input:
    - model: compiled model
    - output_directory: path to output directory
    """
    # Define path fand filename for model summary
    out_summary = os.path.join(output_directory, "model_summary.txt")
    # Save model summary in defined file
    with open(out_summary, "w") as file:
        with redirect_stdout(file):
            model.summary()

    # Define path and filename for model plot
    out_plot = os.path.join(output_directory, "model_plot.png")
    # Save model plot in defined file
    plot_model(model, to_file = out_plot, show_shapes = True, show_layer_names = True)
    
    
def save_model_history(history, epochs, output_directory):
    """Plotting the model history, i.e. loss/accuracy of the model during training
    
    Input: 
    - history: model history
    - epochs: number of epochs the model was trained on 
    - output_directory: desired output directory
    """
    # Define output path
    out_history = os.path.join(output_directory, "model_history.png")

    # Visualize history
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_history)


def save_model_report(report, epochs, batch_size, output_directory):
    """Save report to output directory
    
    Input: 
    - report: model classifcation report
    - output_directory: final output_directory
    """
    # Define output path and file for report
    report_out = os.path.join(output_directory, "classification_report.txt")
    # Save report in defined path
    with open(report_out, 'w', encoding='utf-8') as file:
        file.writelines(f"Classification report for CNN trained with {epochs} epochs and batchsize of {batch_size}:")
        file.writelines(report) 
    

if __name__=="__main__":
    pass