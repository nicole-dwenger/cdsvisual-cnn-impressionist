# 5: CNN on Cultural Image Data

> Mutli-class classification of impressionist paintings using a convolutional neural network with LeNet architechture.
> [Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) |

## Description

> This project related to Assignment 5: Multi-class Classifcation of Cultural Image Data

The purpose of this project was to classify impressionist paintings by their artist. Developing a tool to classify paintings can be used to e.g. sort a large collection of images or to predict the artist of a painting where the artist is unknown. Further, it may also be relevant in cultural research, e.g. to investigate similarities and differences between paintings. For this multi-class classification, a convolutional neural network (CNN) with a LeNet architecture was build. I chose to only investigate paintings of three artists, namely Matisse, Gauguin and VanGogh. 
 

## Methods
### Data 
The data used to train and evaluate the model was extracted from [kaggle](https://www.kaggle.com/delayedkarma/impressionist-classifier-data). This original dataset contains 400 training and 100 test images for 10 artists (Pisarro, Hassam, Monet, Degas, Matisse, Singer-Sargent, Cezanne, Gauguin, Renoir, van Gogh). To reduce processing time, I chose to only investigate three of these artists in this project, namely: Matisse, Gaugin, VanGogh. 

### Preprocessing
To train the CNN, it was required to resize all images to be of same size. To preserve as much as possible, the smallest dimension of the smallest image was found and used to resize all images to a square of this dimension. Further, images were saved in arrays and normalised. The corresponding labels of the images (i.e. their artists) were extracted, and binarised to be used in the CNN. 

### Convolutional Neural Network with LeNet Architecture
Convolutional Neural Networks are especially useful for visual analytics, as they can take into account the three channels of coloured images, and identify more complex features (through convolutional kernels), while reducing dimensionality (through pooling layers). A LeNet CNN consists of two sets of  *Convolutional Layer - Activation Layer - Pooling layer*, and a fully connected layer, to uses the extracted features to predict the possible lables (here: artists). Below is the visualisation of the model architecture used in this project: 

![https://github.com/nicole-dwenger/cdsvisual-cnn-paintings/blob/master/out/model_plot.png]


## Repository Structure

```
|-- data/                        # Directory containing data
    |-- impressionist_sample/    # Example dataset to test functionality of script
        |-- training/            # Training data
            |-- Matisse/         # Subdirectory for artist 1
                |-- img1
                |-- img2
                |-- ...
            |-- Gauguin/         # Subdirectory for artist 2
            |-- Van Gogh/        # Subdirectory for artist 3
            |-- ...
        |-- validation/          # Validation data, with same structure as training data directory

|-- out/                         # Output directory
    |-- model_summary.txt        # Summary of model architecture
    |-- model_plot.png           # Visualisation of model architectur
    |-- model_history.png        # Plot of training history of model
    |-- model_report.txt         # Classification report

|-- src/                         # Directory containing main script for classification
    |-- cnn_impressionist.py     # Main script for preparing data, training and evaluating CNN
    
|-- utils/                       # Directory containing utility script
    |-- cnn_utils.py             # Utility script, with functions used in cnn_impressionist.py

|-- README.md
|-- create_venv.sh               # Bash script to create virtual environment
|-- requirements.txt             # Requirements, installed in virtual environment
```

## Usage

**!** The scripts have only been tested on Linux, using Python 3.6.9. 

### 1. Cloning the Repository and Installing Dependencies

To run the scripts, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create this virtual environment with all necessary dependencies, listed in the `requirements.txt` file. The following commands can be used:

```bash
# cloning the repository
git clone https://github.com/nicole-dwenger/cdsvisual-cnn-artists.git

# move into directory
cd cdsvisual-cnn-artists/

# install virtual environment
bash create_venv.sh

# activate virtual environment 
source venv_cnn/bin/activate
```

### 2. Data
If you wish to reproduce the results of this project, I recommend downloading the dataset of impressionist paintings from  [kaggle](https://www.kaggle.com/delayedkarma/impressionist-classifier-data), storing it in the `data` directory and adjusting the  `-train` and `-test` parameters accordingly. The `-train` and `-test` parameters should specify the directory containing the subdirectories names by the artist (i.e. in the repository structure above this would be `../data/impressionist_subset/training/` and `../data/impressionist_subset/validation/`). I have trained the model on the entire set of training and testing images for the three chosen artists. 
If you simply want to run the script to test its functionality, I have provided a sample of images in the `impressionist_sample` directory. 


### 3. Running the Script 

The script `cnn_impressionist.py` preprocesses the data, trains the model and evaluates the model. The script should be called from the `src/` directory. Example commands are provided below. 

__Example:__
```bash
# moving into src
cd src/

# running script with default parameters
python3 cnn_impressionist.py

# running script with specified parameters
python3 cnn_impressionist.py -a Monet VanGogh Matisse -e 20
```

__Parameters__:
- *-a, --artists : list of str, seperated with space, optional, default:* `Matisse Gaugin VanGogh`\
   List of artist names for which the paintings should be included in the classifier. Should be at least 3, and should their 
   corresponding train and test images should be stored in a directory names by the artist in `training` and `validation`. 
   
- *-train, --train_directory : str, optional, default:* `../data/impressionist_subset/training/`\
   Path to directory containing training data. In this directory, images should be stored in subdirectories, one for each artist. 

- *-test, --train_directory : str, optional, default:* `../data/impressionist_subset/validation/`\
   Path to directory containing training data. In this directory, images should be stored in subdirectories, one for each artist. 

- *-e, --epochs : int, optional, default:* `10`\
   Number of epochs to train the model. 

- *-b, --batch_size : int, optional, default:* `30`\
   Size of batch to train model. 

__Output:__
- *model_summary.txt*\
    Summary of model architecture and number of parameters.

- *model_plot.png*\
   Visualisation of model architecture, i.e. layers. 

- *model_history.png*\
   Training history of model, i.e. training and validation loss and accuracy over epochs. 

- *model_report.txt*\
   Classification report of the model. Also printed to command line. 

## Results and Discussion

The classification report indicates, that the model achieved an weighted F1 score of X when classifying paintings from  Matisse, Gauguin and VanGogh. The model history plot inidicates, that ...




