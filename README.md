# machineLearning-visually-impaired-
A ML project on visually impaired

Project Idea: Object Recognition for Assistive Apps for Blind Users using the VizWiz Classification Dataset.

Description: Modern object recognition models are trained on clean datasets like ImageNet, but visually impaired users take pictures that are often blurry, poorly framed or low lighted. The goal is to classify objects in bad images to power an assistive app that could speak out what's in front of the user. We will use the VizWiz classification dataset which contains images taken by blind people and labeled with object categories.

Dataset: VizWiz Classification (https://vizwiz.org)
Images taken by blind users, labeled with presence/absence of 200 ImageNet style object cateogries, designed to test classifiers trained on ImageNet. 
Scale: about 8,900 images (test set) plus associated metadata and category list
This is perfect because:
-It's real data from visually impaired users not synthetic
-Difficult: blue, off center subjects, bad lighting are good for analysis
-Directly tied to assistive tech: captioning/recognition for blind people

Problem Formulation:
Input: An RGB image x ∈ R^(H×W×3) captured by a blind user
Output: A probability distribution over K object categories, p(y|x) where y ∈ {1,...,K}. We will assume it's a single-label classification even though some images may contain multiple objects. 
Training data: {(xi​,yi​)}i=1-N​ whre xi is an image and yi is the ground truth label.
Preprocessing: Resize to 224x224. Normalize with ImageNet means/stds

Loss function: Cross entropy loss for multi class classification

Training procedure: 
Optimizer: Adam or SGD with momentum
Learning rate schedule: step decay or cosine schedule
Epochs: 10-30 with early stopping on validation loss
Batch size: 32-128

ML approaches to compare:
1. Model 1
