Multi-Class U-Net Model for Image Segmentation
Overview

This protocol provides instructions for setting up and running a multi-class U-Net model designed for image segmentation tasks. The U-Net model is a convolutional neural network that excels in biomedical image segmentation, but it can be adapted for various image segmentation tasks involving multiple classes.
Requirements

    Python 3.x
    TensorFlow 2.x
    Keras (included with TensorFlow 2.x)
    NumPy
    OpenCV
    Matplotlib
    Scikit-learn
    Pandas

Please ensure all the required libraries are installed before running the script. You can install these packages using pip:

bash

pip install tensorflow numpy opencv-python matplotlib scikit-learn pandas

Dataset Preparation

Your dataset should consist of pairs of images and their corresponding segmentation masks. The images and masks are expected to be in .tiff/.tif format. Masks should be encoded with integer values, where each value represents a different class. Prepare a CSV file with two columns: Input and Mask, pointing to the respective file paths of the images and their masks.

Example of a CSV file structure:
Input,Mask
path/to/image1.tif,path/to/mask1.tif
path/to/image2.tif,path/to/mask2.tif
...

Model Training

    Configuration: Adjust the parameters such as n_classes, IMG_HEIGHT, IMG_WIDTH, and IMG_CHANNELS in the script to match your dataset characteristics.

    Dataset Loading: Update the path in the script to your dataset CSV file.

    Training: Run the script to start the training process. The script will automatically split the data into training and validation sets, train the model, and save the best-performing model based on validation loss.

Evaluation and Prediction

After training, the script evaluates the best model on the test set and displays the accuracy. It also includes a section for making predictions on test images, visualizing the original images, their true masks, and the predicted masks.
Saving the Model

The script saves the best model based on validation loss during training. You can load this model for further evaluation or to make predictions on new data.