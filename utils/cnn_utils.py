#!/usr/bin/env python

"""
Utility script with functions used in cnn_impressionist.py script 

Functions in this script:
  - get_label_names(): get all the names of subdirectories corresponding to artists in the train directory
  - get_min_dim(): get minimum dimension of images in train and test directory
  - unique_path(): enumerates filename, if file exists already 
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

# Tensorflow tools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide warnings
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K


# UTILITY FUNCTIONS ------------------------------------

def get_label_names(directory):
    """
    Get names of labels (artists) by finding the directory names
    Input:
      - directory: directory where subdirectories of artists are stored
    Returns:
      - label_names: list of names of subdirectories (corresponding to artists here)
    """
    # Create empty target list for label names
    label_names = []
    # Get all file names in the directory
    for name in os.listdir(directory):
        # If it is not a hidden file
        if not name.startswith('.'):
            # Append name to label_names
            label_names.append(name)
            
    return label_names

def get_min_dim(train_directory, test_directory, names):
    """
    Getting dimension of the smallest image in test and train data
    Input:
      - train_dir, test_dir: paths to training and test directory, containing subdirectories with images
      - names: names of subdirectories in train_dir and test_dir, here: refers to artist names
    Returns: 
      - Minimum dimension of smallest image
    """
    # Create empty lists for dimensions
    dimensions = []
    
    # For each name, i.e. artist 
    for name in names:
        # Get all image paths in train and test directory
        img_paths = (glob.glob(os.path.join(train_directory, name, "*.jpg")) +
                     glob.glob(os.path.join(test_directory, name, "*.jpg")))
        
        # Get the size of each image and append to list
        for path in img_paths:
            img = Image.open(path)
            dimensions.extend(img.size)                               
    
    return min(dimensions)

def unique_path(filepath):
    """
    Creating unique filepath/filename by incrementing filename, if the filename exists already 
    Input:
      - desired filepath
    Returns:
      - if path does not exist: desired filepath
      - if path exists: desired filepath, enumerated
    """ 
    # If the path does not exist, keep the original filepath
    if not os.path.exists(filepath):
        return filepath
    
    # Otherwise, split the path, and append a number that does not exist already, return the new path
    else:
        i = 1
        path, ext = os.path.splitext(filepath)
        new_path = "{}_{}{}".format(path, i, ext)
        
        while os.path.exists(new_path):
            i += 1
            new_path = "{}_{}{}".format(path, i, ext)
            
        return new_path

def save_model_info(model, output_directory, filename_summary, filename_plot):
    """
    Save model summary in .txt file and plot of model in .png
    Input:
      - model: compiled model
      - output_directory: path to output directory
      - filename_summary: name of file to save summary in
      - filename_plot: name of file to save visualisation of model
    """
    # Define path fand filename for model summary
    out_summary = unique_path(os.path.join(output_directory, filename_summary))
    # Save model summary in defined file
    with open(out_summary, "w") as file:
        with redirect_stdout(file):
            model.summary()

    # Define path and filename for model plot
    out_plot = unique_path(os.path.join(output_directory, filename_plot))
    # Save model plot in defined file
    plot_model(model, to_file = out_plot, show_shapes = True, show_layer_names = True)
     
def save_model_history(history, epochs, output_directory, filename):
    """
    Plotting the model history, i.e. loss/accuracy of the model during training
    Input: 
      - history: model history
      - epochs: number of epochs the model was trained on 
      - output_directory: desired output directory
      - filename: name of file to save history in
    """
    # Define output path
    out_history = unique_path(os.path.join(output_directory, filename))

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

def save_model_report(report, epochs, batch_size, output_directory, filename):
    """
    Save report to output directory
    Input: 
      - report: model classifcation report
      - output_directory: final output_directory
      - filename: name of file to save report in
    """
    # Define output path and file for report
    report_out = unique_path(os.path.join(output_directory, filename))
    # Save report in defined path
    with open(report_out, 'w', encoding='utf-8') as file:
        file.writelines(f"Classification report for CNN trained with {epochs} epochs and batchsize of {batch_size}:\n")
        file.writelines(report) 
    

if __name__=="__main__":
    pass