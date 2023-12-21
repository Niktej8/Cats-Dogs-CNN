# Cat vs Dog Image Classifier

This repository contains code for a Convolutional Neural Network (CNN) designed to classify images as either a cat or a dog. The model is trained on separate datasets of cat and dog images, leveraging data augmentation for enhanced performance.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Usage](#usage)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Predictions](#predictions)



## Introduction

This project focuses on building a CNN to solve the binary classification problem of distinguishing between cat and dog images. The model is trained on a large dataset of labeled cat and dog images to learn the patterns and features associated with each class.

## Dependencies

To run this code, you'll need the following dependencies:

- TensorFlow
- Keras
- Matplotlib
- Seaborn
- ...

Make sure to install the required dependencies before running the code.

## Dataset

You can download the dataset for the Cats and Dogs Classifier from the following link:

[Download Dataset](https://cats-and-dogs-classifier.s3.ap-south-1.amazonaws.com/dataset.zip)

Make sure to unzip the downloaded file before using it for training or testing.

## Usage

Provide instructions on how to use the code, including any setup steps and commands.

## Data Augmentation

Data augmentation is a technique used to increase the diversity of the training dataset by applying various transformations to the existing images. This helps the model generalize better and improves its performance on unseen data.

### Techniques Used:

- **Shearing:** A shearing transformation is applied to the images, changing their shapes by shifting one part of the image along the horizontal or vertical axis.
- **Random Zoom:** Randomly zooming into images helps the model become more robust by learning features at different scales.
- **Horizontal Flips:** Randomly flipping images horizontally introduces variations in the orientation, enhancing the model's ability to recognize objects from different angles.

## Model Architecture

The Convolutional Neural Network (CNN) architecture is designed to effectively capture features for cat and dog classification.

### Layers:

- **Convolutional Layer 1:** Filters: 32, Kernel Size: 3x3, Activation Function: ReLU, Input Shape: (64, 64, 3)
- **Max Pooling Layer 1:** Pool Size: 2x2, Strides: 2
- **Convolutional Layer 2:** Filters: 32, Kernel Size: 3x3, Activation Function: ReLU
- **Max Pooling Layer 2:** Pool Size: 2x2, Strides: 2
- **Flattening Layer:** Converts the 2D feature maps to a 1D vector for input to the Dense layers.
- **Fully Connected Layer (Dense Layer 1):** Units: 128, Activation Function: ReLU
- **Output Layer (Dense Layer 2):** Units: 1 (Binary classification), Activation Function: Sigmoid

## Training

The model is trained on separate datasets for cats and dogs. Adjust hyperparameters such as the number of epochs and batch size as needed.

## Evaluation

- **Accuracy:** The proportion of correctly classified instances.
- **Precision:** The proportion of true positive predictions out of all positive predictions.
- **Recall:** The proportion of true positive predictions out of all actual positive instances.
- **F1 Score:** The harmonic mean of precision and recall.

## Predictions
You can make predictions on new images using the trained model by changing their path

```bash
cat_and_dog_classifier.ipynb
