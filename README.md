# CNN on Cultural Image Data

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion)

## Description

> This project relates to Assignment 5: Multi-class Classification of Cultural Image Data
> of the course Visual Analytics.

The aim of this project was to classify complex, cultural image data. Specifically, a dataset of impressionist paintings was used to investigate whether it is possible to classify paintings by their artist. This may be relevant to e.g. sort a large collection of images or to predict the artist of an image where the artist is unknown. 

For the multi-class classification task, a Convolutional Neural Network (CNN) with a LeNet Architecture was build, trained and evaluated. Convolutional Neural Networks are especially useful for visual analytics, as they can take into account the three channels of coloured images, and identify more complex features (through convolutional kernels), while reducing dimensionality of the feature space (through pooling layers).  
 

## Methods
### Data and Preprocessing
The data used to train and evaluate the model was extracted from [Kaggle](https://www.kaggle.com/delayedkarma/impressionist-classifier-data). This dataset contains 400 training and 100 test images for 10 artists (Pisarro, Hassam, Monet, Degas, Matisse, Singer-Sargent, Cezanne, Gauguin, Renoir, van Gogh). 
To train the CNN, it was required to resize all images to be of same size. To preserve as much as possible, the smallest dimension of the smallest image was found and used to resize all images to a square of this dimension. For the subset of data used for this project, this meant that all images were resized to be of size 266x266. Further, images were scaled using min-max-regularisation. The corresponding labels of the images (artists) were extracted, and binarised.
 
### Convolutional Neural Network with LeNet Architecture
A Convolutional Neural Network with a LeNet Architecture consists of an input layer, followed by 2 sets of **Convolutional Layer - Activation Layer - Pooling layer**, a flattening layer, two fully connected layers and finally a softmax classifier to predict the artist. The model was trained on a batch size of 50, once over 10 epochs and once over 20 epochs. As optimiser, stochastic gradient descent with a learning rate of 0.01 was used. The [model summary](https://github.com/nicole-dwenger/cdsvisual-cnn-impressionist/blob/master/out/model_summary.txt) and a [visualisation of the model](https://github.com/nicole-dwenger/cdsvisual-cnn-impressionist/blob/master/out/model_plot.png) used for this project can be found in the `out/` directory in this repository.


## Repository Structure

Note, that the `data/` directory only contains empty folders. The data is too large to be stored on GitHub. The directories only serve to illustrate how the data should be stored to run the script. Directions for downloading the data are provided below. 

```
|-- data/                        # Directory containing data (see note above)
    |-- impressionist_example/   # Example directory of how data should be structured
        |-- training/            # Example training data directory
            |-- Matisse/         # Example subdirectory for artist 1
            |-- Gauguin/         # Example subdirectory for artist 2
            |-- Van Gogh/        # Example subdirectory for artist 2
            |-- ...
        |-- validation/          # Example validation data directory, structure identical to training

|-- out/                         # Output directory, with example output
    |-- model_summary.txt        # Summary of model architecture     
    |-- model_plot.png           # Visualisation of model architecture        
    |-- model_history.png        # Plot of training history of model (over 10 epochs)
    |-- model_history_1.png      # Plot of training history of model (over 20 epochs)
    |-- model_report.txt         # Classification report (over 10 epochs)
    |-- model_report_1.txt.      # Classification report (over 20 epochs)

|-- src/                         # Directory containing main script for classification
    |-- cnn_impressionist.py     # Main script for preparing data, training and evaluating CNN
    
|-- utils/                       # Directory containing utility script
    |-- cnn_utils.py             # Utility script, with functions used in cnn_impressionist.py

|-- README.md
|-- create_venv.sh               # Bash script to create virtual environment
|-- requirements.txt             # Requirements, installed in virtual environment
```

## Usage

**!** The script has only been tested on Linux, using Python 3.6.9. 

### 1. Cloning the Repository and Installing Dependencies

To run the scripts, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create this virtual environment with all necessary dependencies, listed in the `requirements.txt` file. The following commands can be used:

```bash
# cloning the repository
git clone https://github.com/nicole-dwenger/cdsvisual-cnn-impressionist.git

# move into directory
cd cdsvisual-cnn-impressionist/

# install virtual environment
bash create_venv.sh

# activate virtual environment 
source venv_cnn/bin/activate
```

### 2. Data
The data was too large to be stored in this repository. If you wish to reproduce the results of this project, I recommend downloading the *training* and *validation* folders of the impressionist paintings dataset from [kaggle](https://www.kaggle.com/delayedkarma/impressionist-classifier-data), and saving these in `data/impressionist`. Optimally, the structure should be identical to the structure illustrated in the repository structure displayed above. When running the script, you can adjust the `-train` and `-test` parameters to represent the directories containing the training and validation data, i.e., in the repository structure above this would be `../data/impressionist_example/training/` and `../data/impressionist_example/validation/`. For this project, I have trained the model on the entire set of 400 training and 100 testing images for all 10 artists. 


### 3. Running the Script 

The script `cnn_impressionist.py` preprocesses the data, trains the model and evaluates the model. The script should be called from the `src/` directory:

```bash
# moving into src
cd src/

# running script with default parameters
python3 cnn_impressionist.py 

# running script with specified parameters
python3 cnn_impressionist.py -train data/training/ -test data/validation/ -a Monet VanGogh Matisse -e 20
```

__Parameters__:
- `-a, --artists`: *list of str, seperated with space, optional, default:* `ALL`\
   List of artist names for which the paintings should be included in the classifier. Should be at least 3, and should their 
   corresponding train and test images should be stored in a directory named by the artist in `training/` and `validation/` directories. Use `ALL` to take all artists for which directories are in `training/` (and should also be in `validation/`.
   
- `-train, --train_directory`: *str, optional, default:* `../data/impressionist/training/`\
   Path to directory containing training data. In this directory, images should be stored in subdirectories, one for each artist.

- `-test, --test_directory`: *str, optional, default:* `../data/impressionist/validation/`\
   Path to directory containing training data. In this directory, images should be stored in subdirectories, one for each artist.

- `-e, --epoch`: *int, optional, default:* `10`\
   Number of epochs to train the model. 

- `-b, --batch_size`: *int, optional, default:* `50`\
   Size of batch to train model. 

__Output__ saved in `out/`:
- `model_summary.txt`\
    Summary of model architecture and number of parameters. If the file exists already, a number will be added to the filename to avoid overwriting. 

- `model_plot.png`\
   Visualisation of model architecture, i.e. layers. If the file exists already, a number will be added to the filename to avoid overwriting. 

- `model_history.png`\
   Training history of model, i.e. training and validation loss and accuracy over epochs. If the file exists already, a number will be added to the filename to avoid overwriting. 

- `model_report.txt`\
   Classification report of the model. If the file exists already, a number will be added to the filename to avoid overwriting. Also printed to command line. 

## Results and Discussion

Outputs of the model can be found in the `out/` directory of this repository. The [classification report](https://github.com/nicole-dwenger/cdsvisual-cnn-impressionist/blob/master/out/model_report.txt) of the model trained over 10 epochs indicated a weighted F1 score of 0.35, while the [classification report](https://github.com/nicole-dwenger/cdsvisual-cnn-impressionist/blob/master/out/model_report_1.txt) of the model trained over 20 epochs indicated a weighted F1 score of 0.38, when classifying paintings of the 10 artists.

Looking at the model history, it seems that the model trained over 20 epochs (displayed on the right) started overfitting on the training data, as the accuracy does not increase for the test data, while it keeps increasing for the training data. Similarly, the loss for the test data actually starts increasing from epoch 10, while it decreases for the training data. This is likely the reason why the model did not improve much when running it for 20 epochs. Thus, in early stopping after 10 epochs (displayed on the left) could prevent this overfitting. Other measures, such as drop out layers could be also be considered to further prevent overfitting.


 10 epochs | 15 epochs
:-------------------------:|:-------------------------:
![](https://github.com/nicole-dwenger/cdsvisual-cnn-impressionist/blob/master/out/model_history.png)  |  ![](https://github.com/nicole-dwenger/cdsvisual-cnn-impressionist/blob/master/out/model_history_1.png)


Rather than using training a CNN from scratch, it could improve classification to use transfer learning, meaning pretrained models, such as VGG16 to classify these images. The pretrained models have been trained to extract relevant features from a large collection of images. Thus, the CNN (as in this project) does not need to be trained from scratch, but only the last fully connected layers would need to be re-trained to classify the artists. The aim of this project, however was to also indicate how a CNN can be build and trained from scratch. 





